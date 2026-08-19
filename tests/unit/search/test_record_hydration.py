import asyncio
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

import pytest

from searchkernel.domain import Record, RecordIdentity, SearchResultProvenance
from searchkernel.ports.search_results import RecordSearchFailure
from searchkernel.runtime import HydrationCache, HydrationCacheKey
from searchkernel.search.record_hydration import RecordHydrationCoordinator
from searchkernel.search.record_pipeline import (
    RecordSearchCandidate,
    RecordSearchError,
)

pytestmark = pytest.mark.asyncio


def _candidate(record_id: str) -> RecordSearchCandidate:
    return RecordSearchCandidate(
        identity=RecordIdentity(None, "note", record_id),
        score=1.0,
        provenance=SearchResultProvenance(),
    )


def _record(record_id: str) -> Record:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    return Record(
        source_kind="note",
        source_id=record_id,
        title=record_id,
        body=f"body for {record_id}",
        created_at=timestamp,
        updated_at=timestamp,
    )


def _coordinator(
    hydrator: object,
    *,
    hydrate_record,
    cache: HydrationCache[Record | None] | None = None,
    policy_version: str | None = None,
    hydration_version: object | None = None,
    hydration_version_provider=None,
    max_batch_size: int = 100,
    max_concurrency: int = 8,
    handle_error=None,
) -> RecordHydrationCoordinator:
    return RecordHydrationCoordinator(
        hydrator=hydrator,
        hydrate_record=hydrate_record,
        hydration_cache=cache,
        policy_version=policy_version,
        hydration_version=hydration_version,
        hydration_version_provider=hydration_version_provider,
        max_batch_size=max_batch_size,
        max_concurrency=max_concurrency,
        handle_error=handle_error or _lenient_error_handler,
    )


def _lenient_error_handler(
    stage: str, error: Exception, failures: list[RecordSearchFailure]
) -> None:
    failures.append(RecordSearchFailure(stage, str(error), type(error).__name__))


async def test_batch_hydrator_is_preferred_and_respects_batch_boundaries() -> None:
    """Batch hydration is preferred and receives bounded ordered slices."""
    candidates = [_candidate(str(index)) for index in range(5)]

    class BatchHydrator:
        def __init__(self) -> None:
            self.batches: list[list[str]] = []

        async def hydrate_records(
            self, identities: Sequence[RecordIdentity]
        ) -> Mapping[str, Record]:
            self.batches.append([identity.source_id for identity in identities])
            return {
                identity.storage_key: _record(identity.source_id)
                for identity in identities
            }

    hydrator = BatchHydrator()
    coordinator = _coordinator(
        hydrator,
        hydrate_record=lambda identity: pytest.fail(
            f"scalar hydration used for {identity.source_id}"
        ),
        max_batch_size=2,
    )

    result = await coordinator.hydrate_candidates(candidates, [], [])

    assert hydrator.batches == [["0", "1"], ["2", "3"], ["4"]]
    assert [candidate.record_id for candidate, _ in result] == [
        candidate.record_id for candidate in candidates
    ]


async def test_scalar_hydration_is_bounded_and_preserves_candidate_order() -> None:
    """Scalar hydration runs concurrently without changing ranked order."""
    candidates = [_candidate(str(index)) for index in range(4)]
    started = asyncio.Event()
    release = asyncio.Event()
    active = 0
    maximum_active = 0

    async def hydrate(identity: RecordIdentity) -> Record:
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        if maximum_active == 2:
            started.set()
        await release.wait()
        active -= 1
        return _record(identity.source_id)

    coordinator = _coordinator(
        object(), hydrate_record=hydrate, max_concurrency=2
    )
    task = asyncio.create_task(coordinator.hydrate_candidates(candidates, [], []))
    await started.wait()
    release.set()
    result = await task

    assert maximum_active == 2
    assert [candidate.record_id for candidate, _ in result] == [
        candidate.record_id for candidate in candidates
    ]


async def test_hydration_cache_reports_hit_miss_and_version_fallback() -> None:
    """Cache diagnostics distinguish hits, misses, and scalar version fallback."""
    candidate = _candidate("cached")
    missed = _candidate("missed")
    cache = HydrationCache()
    key = HydrationCacheKey.build(
        candidate.identity, record_version=3, policy_version="policy/v1"
    )
    cache.set(key, _record("cached"))

    class VersionProvider:
        def __init__(self) -> None:
            self.batch_calls = 0
            self.scalar_calls = 0

        def hydration_versions(self, identities: Sequence[RecordIdentity]):
            self.batch_calls += 1
            raise RuntimeError("batch version unavailable")

        def __call__(self, identity: RecordIdentity) -> int:
            self.scalar_calls += 1
            return 3

    class Hydrator:
        async def hydrate_records(
            self, identities: Sequence[RecordIdentity]
        ) -> Mapping[str, Record]:
            return {
                identity.storage_key: _record(identity.source_id)
                for identity in identities
            }

    provider = VersionProvider()
    diagnostics: list[str] = []
    result = await _coordinator(
        Hydrator(),
        hydrate_record=lambda identity: pytest.fail("batch hydrator expected"),
        cache=cache,
        policy_version="policy/v1",
        hydration_version_provider=provider,
    ).hydrate_candidates([candidate, missed], [], diagnostics)

    assert result[0][1] == _record("cached")
    assert result[1][1] == _record("missed")
    assert provider.batch_calls == 1
    assert provider.scalar_calls == 2
    assert diagnostics == [
        "hydration_cache:batch_version_fallback:RuntimeError",
        "hydration_cache:hit",
        "hydration_cache:miss",
    ]


async def test_hydration_cache_coalesces_concurrent_misses() -> None:
    """An in-flight miss is coalesced and receives the completed record."""
    candidate = _candidate("shared")
    joined = asyncio.Event()

    class TrackingCache(HydrationCache[Record | None]):
        async def async_wait_for_miss(self, key: HydrationCacheKey):
            joined.set()
            leader, shared = await super().async_wait_for_miss(key)
            return leader, shared

    cache = TrackingCache()
    key = HydrationCacheKey.build(
        candidate.identity, record_version=1, policy_version="policy/v1"
    )
    leader, _ = await cache.async_wait_for_miss(key)
    assert leader
    joined.clear()
    first_diagnostics: list[str] = []
    coordinator = _coordinator(
        object(),
        hydrate_record=lambda identity: pytest.fail("coalesced miss hydrated"),
        cache=cache,
        policy_version="policy/v1",
        hydration_version=1,
    )
    task = asyncio.create_task(
        coordinator.hydrate_candidates([candidate], [], first_diagnostics)
    )
    await joined.wait()
    cache.set(key, _record("shared"))
    result = await task

    assert result[0][1] == _record("shared")
    assert first_diagnostics == ["hydration_cache:coalesced"]


@pytest.mark.parametrize("failure_mode", ["strict", "lenient"])
async def test_batch_failures_follow_handler_mode_and_mutate_failures(
    failure_mode: str,
) -> None:
    """Hydration failures either raise or append one caller-owned failure."""
    error = RuntimeError("hydration unavailable")

    failures: list[RecordSearchFailure] = []

    async def hydrate(identity: RecordIdentity) -> Record:
        raise error

    def handle_error(
        stage: str, failure: Exception, target: list[RecordSearchFailure]
    ) -> None:
        if failure_mode == "strict":
            raise RecordSearchError(stage, failure)
        _lenient_error_handler(stage, failure, target)

    coordinator = _coordinator(
        object(),
        hydrate_record=hydrate,
        handle_error=handle_error,
    )

    if failure_mode == "strict":
        with pytest.raises(RecordSearchError, match="hydration retrieval failed"):
            await coordinator.hydrate_candidates([_candidate("broken")], failures, [])
        assert failures == []
    else:
        assert await coordinator.hydrate_candidates(
            [_candidate("broken")], failures, []
        ) == []
        assert failures == [
            RecordSearchFailure("hydration", str(error), "RuntimeError")
        ]


async def test_cache_bypass_diagnostic_and_missing_or_none_records_are_preserved() -> None:
    """Cache bypasses still hydrate, while missing and explicit None remain visible."""
    candidates = [_candidate("missing"), _candidate("none")]

    class Hydrator:
        async def hydrate_records(
            self, identities: Sequence[RecordIdentity]
        ) -> Mapping[str, Record | None]:
            return {identities[1].storage_key: None}

    diagnostics: list[str] = []
    result = await _coordinator(
        Hydrator(),
        hydrate_record=lambda identity: pytest.fail("batch hydrator expected"),
        cache=HydrationCache(),
    ).hydrate_candidates(candidates, [], diagnostics)

    assert [(candidate.record_id, record) for candidate, record in result] == [
        ("missing", None),
        ("none", None),
    ]
    assert diagnostics == ["hydration_cache:bypass:missing_policy_version"]
