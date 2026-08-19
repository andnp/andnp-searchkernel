import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Event, Thread
from typing import Any

import pytest

from searchkernel.domain import (
    Record,
    RecordHit,
    RecordIdentity,
    SearchResultProvenance,
)
from searchkernel.runtime import (
    CandidateCacheKey,
    CandidateResultCache,
    HydrationCache,
    HydrationCacheKey,
    QueryEmbeddingCache,
    SearchEpochs,
    UnstableCacheKey,
)
from searchkernel.search.record_pipeline import (
    RecordSearchCandidate,
    RecordSearchPipeline,
)


@dataclass
class _Clock:
    value: float = 0.0

    def __call__(self) -> float:
        return self.value


def _candidate_key(query: str = "query") -> CandidateCacheKey:
    return CandidateCacheKey.build(
        query=query,
        filters={},
        requested_limit=1,
        acquisition_limit=1,
        adaptive_limit=None,
        routing_fingerprint="r",
        encoder_namespace=None,
        epochs=SearchEpochs(),
        policy_version=None,
    )


@pytest.mark.asyncio
async def test_async_embedding_misses_single_flight_and_warm_hits() -> None:
    cache = QueryEmbeddingCache()
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def compute() -> list[float]:
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return [1.0, 2.0]

    first = asyncio.create_task(
        cache.async_get_or_compute(
            encoder_namespace="model/v1|dim=2",
            query="  Same \t query  ",
            compute=compute,
        )
    )
    await started.wait()
    second = asyncio.create_task(
        cache.async_get_or_compute(
            encoder_namespace="model/v1|dim=2",
            query="Same query",
            compute=compute,
        )
    )
    release.set()
    assert await asyncio.gather(first, second) == [[1.0, 2.0], [1.0, 2.0]]
    assert await cache.async_get_or_compute(
        encoder_namespace="model/v1|dim=2",
        query="Same query",
        compute=compute,
    ) == [1.0, 2.0]
    assert calls == 1
    assert cache.metrics.coalesced_waiters == 1
    assert cache.metrics.hits == 1


def test_embedding_cache_isolates_namespace_case_and_returns_copies() -> None:
    cache = QueryEmbeddingCache()
    calls = 0

    def compute() -> list[float]:
        nonlocal calls
        calls += 1
        return [float(calls)]

    first = cache.get_or_compute(
        encoder_namespace="model/v1|prompt=a",
        query="  Case  Sensitive ",
        compute=compute,
    )
    first.append(99.0)
    assert cache.get_or_compute(
        encoder_namespace="model/v1|prompt=a",
        query="Case Sensitive",
        compute=compute,
    ) == [1.0]
    assert cache.get_or_compute(
        encoder_namespace="model/v1|prompt=a",
        query="case sensitive",
        compute=compute,
    ) == [2.0]
    assert cache.get_or_compute(
        encoder_namespace="model/v1|prompt=b",
        query="Case Sensitive",
        compute=compute,
    ) == [3.0]


def test_embedding_cache_bounds_ttl_and_failures() -> None:
    clock = _Clock()
    cache = QueryEmbeddingCache(ttl_seconds=2.0, max_entries=1, clock=clock)
    calls = 0

    def compute() -> list[float]:
        nonlocal calls
        calls += 1
        return [float(calls)]

    assert cache.get_or_compute(model_name="m", query="a", compute=compute) == [1.0]
    assert cache.get_or_compute(model_name="m", query="b", compute=compute) == [2.0]
    assert cache.get_or_compute(model_name="m", query="a", compute=compute) == [3.0]
    clock.value = 2.0
    assert cache.get_or_compute(model_name="m", query="a", compute=compute) == [4.0]
    assert cache.metrics.evictions >= 2

    def fail() -> list[float]:
        raise RuntimeError("failed")

    with pytest.raises(RuntimeError):
        cache.get_or_compute(model_name="m", query="failure", compute=fail)
    with pytest.raises(RuntimeError):
        cache.get_or_compute(model_name="m", query="failure", compute=fail)


@pytest.mark.asyncio
async def test_cancelled_embedding_is_not_cached() -> None:
    cache = QueryEmbeddingCache()
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def compute() -> list[float]:
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return [1.0]

    task = asyncio.create_task(
        cache.async_get_or_compute(
            encoder_namespace="m",
            query="cancel",
            compute=compute,
        )
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    release.set()
    assert await cache.async_get_or_compute(
        encoder_namespace="m",
        query="cancel",
        compute=lambda: [2.0],
    ) == [2.0]
    assert calls == 1


def test_candidate_key_is_stable_and_epoch_sensitive() -> None:
    first = CandidateCacheKey.build(
        query="  query   text ",
        filters={"tags": {"b", "a"}, "status": "active"},
        requested_limit=5,
        acquisition_limit=25,
        adaptive_limit=100,
        routing_fingerprint="routing/v1",
        encoder_namespace="model/v1|dim=3",
        policy_version="policy/v2",
        epochs=SearchEpochs(1, 2, 3),
    )
    second = CandidateCacheKey.build(
        query="  query   text ",
        filters={"tags": {"b", "a"}, "status": "active"},
        requested_limit=5,
        acquisition_limit=25,
        adaptive_limit=100,
        routing_fingerprint="routing/v1",
        encoder_namespace="model/v1|dim=3",
        policy_version="policy/v2",
        epochs=SearchEpochs(2, 2, 3),
    )
    assert first == CandidateCacheKey.build(
        query="  query   text ",
        filters={"tags": {"b", "a"}, "status": "active"},
        requested_limit=5,
        acquisition_limit=25,
        adaptive_limit=100,
        routing_fingerprint="routing/v1",
        encoder_namespace="model/v1|dim=3",
        policy_version="policy/v2",
        epochs=SearchEpochs(1, 2, 3),
    )
    assert first.query == "query text"
    assert first != second
    with pytest.raises(UnstableCacheKey):
        CandidateCacheKey.build(
            epochs=SearchEpochs(),
            filters={"unstable": object()},
            query="query",
            requested_limit=5,
            acquisition_limit=25,
            adaptive_limit=100,
            routing_fingerprint="routing/v1",
            encoder_namespace="model/v1|dim=3",
            policy_version="policy/v2",
        )


def test_candidate_key_separates_source_scoped_authorization_variants() -> None:
    """Distinct authorization claims cannot share a candidate-cache entry."""
    common = {
        "query": "query",
        "requested_limit": 1,
        "acquisition_limit": 5,
        "adaptive_limit": None,
        "routing_fingerprint": "routing/v1",
        "encoder_namespace": None,
        "policy_version": "policy/v1",
        "epochs": SearchEpochs(),
    }
    first = CandidateCacheKey.build(
        **common,
        filters={
            "source_scoped_filters": {
                "note": {"metadata_contains_any": {"acl": ["first"]}}
            }
        },
    )
    second = CandidateCacheKey.build(
        **common,
        filters={
            "source_scoped_filters": {
                "note": {"metadata_contains_any": {"acl": ["second"]}}
            }
        },
    )

    assert first != second
    assert first.filters != second.filters


def test_candidate_cache_is_bounded_and_defensive() -> None:
    cache: CandidateResultCache[list[str]] = CandidateResultCache(max_entries=1)
    key = CandidateCacheKey.build(
        query="query",
        filters={},
        requested_limit=1,
        acquisition_limit=5,
        adaptive_limit=None,
        routing_fingerprint="r",
        encoder_namespace=None,
        epochs=SearchEpochs(),
        policy_version=None,
    )
    value = ["a"]
    cache.set(key, value)
    value.append("changed")
    assert cache.get(key) == ["a"]
    assert cache.metrics.hits == 1


def test_candidate_cache_isolates_mutable_candidate_provenance() -> None:
    """Protect cached candidate provenance from mutations after retrieval."""
    cache: CandidateResultCache[tuple[RecordSearchCandidate, ...]] = CandidateResultCache()
    key = CandidateCacheKey.build(
        query="query",
        filters={},
        requested_limit=1,
        acquisition_limit=5,
        adaptive_limit=None,
        routing_fingerprint="r",
        encoder_namespace=None,
        epochs=SearchEpochs(),
        policy_version=None,
    )
    identity = RecordIdentity("workspace", "note", "1")
    provenance = SearchResultProvenance(record_identity=identity)
    provenance.add_strategy("keyword", 1, 1.0)
    cache.set(
        key,
        (
            RecordSearchCandidate(
                identity=identity,
                score=1.0,
                provenance=provenance,
            ),
        ),
    )

    cached = cache.get(key)
    assert cached is not None
    cached[0].provenance.add_strategy("mutated", 2, 0.5)

    stored = cache.get(key)
    assert stored is not None
    assert "mutated" not in stored[0].provenance.strategies


def test_hydration_cache_requires_version_in_key_and_expires_missing() -> None:
    clock = _Clock()
    cache: HydrationCache[Record] = HydrationCache(
        missing_ttl_seconds=1.0,
        clock=clock,
    )
    identity = RecordIdentity("workspace", "note", "1")
    old_key = HydrationCacheKey.build(
        identity,
        record_version=datetime(2026, 1, 1, tzinfo=UTC),
        policy_version="policy/v1",
    )
    new_key = HydrationCacheKey.build(
        identity,
        record_version=datetime(2026, 1, 2, tzinfo=UTC),
        policy_version="policy/v1",
    )
    cache.set(old_key, None)
    assert cache.lookup(old_key) == (True, None)
    clock.value = 1.0
    assert cache.lookup(old_key) == (False, None)
    record = Record(
        source_kind="note",
        source_id="1",
        title="title",
        body="body",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=datetime(2026, 1, 2, tzinfo=UTC),
        workspace_id="workspace",
    )
    cache.set(new_key, record)
    assert cache.get(new_key) == record
    assert cache.get(old_key) is None


@pytest.mark.asyncio
async def test_candidate_cache_single_flight_isolates_failures() -> None:
    cache: CandidateResultCache[list[str]] = CandidateResultCache()
    key = CandidateCacheKey.build(
        query="query",
        filters={},
        requested_limit=1,
        acquisition_limit=1,
        adaptive_limit=None,
        routing_fingerprint="r",
        encoder_namespace=None,
        epochs=SearchEpochs(),
        policy_version=None,
    )
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def compute() -> list[str]:
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return ["candidate"]

    first = asyncio.create_task(cache.async_get_or_compute(key, compute))
    await started.wait()
    second = asyncio.create_task(cache.async_get_or_compute(key, compute))
    release.set()
    assert await asyncio.gather(first, second) == [["candidate"], ["candidate"]]
    assert calls == 1
    assert cache.metrics.coalesced_waiters == 1

    async def fail() -> list[str]:
        raise RuntimeError("isolated")

    failed_key = CandidateCacheKey.build(
        query="failure",
        filters={},
        requested_limit=1,
        acquisition_limit=1,
        adaptive_limit=None,
        routing_fingerprint="r",
        encoder_namespace=None,
        epochs=SearchEpochs(),
        policy_version=None,
    )
    with pytest.raises(RuntimeError, match="isolated"):
        await cache.async_get_or_compute(failed_key, fail)
    assert cache.metrics.misses == 2


@pytest.mark.asyncio
async def test_async_wait_does_not_starve_a_single_worker_executor() -> None:
    """A follower must complete while the default executor is occupied."""
    cache: CandidateResultCache[list[str]] = CandidateResultCache()
    key = _candidate_key("executor-starvation")
    assert await cache.async_wait_for_miss(key) == (True, None)

    executor = ThreadPoolExecutor(max_workers=1)
    loop = asyncio.get_running_loop()
    loop.set_default_executor(executor)
    blocker_started = Event()
    release_blocker = Event()

    def block_executor() -> None:
        blocker_started.set()
        release_blocker.wait()

    blocker = loop.run_in_executor(None, block_executor)
    assert blocker_started.wait(timeout=1.0)
    follower = asyncio.create_task(cache.async_wait_for_miss(key))
    loop.call_soon(cache.set, key, ["candidate"])
    try:
        assert await asyncio.wait_for(follower, timeout=1.0) == (
            False,
            ["candidate"],
        )
    finally:
        release_blocker.set()
        await blocker
        executor.shutdown(wait=True)


@pytest.mark.asyncio
async def test_cancelled_follower_does_not_cancel_shared_completion() -> None:
    """Cancelling one follower leaves another follower able to receive data."""
    cache: CandidateResultCache[list[str]] = CandidateResultCache()
    key = _candidate_key("follower-cancellation")
    assert await cache.async_wait_for_miss(key) == (True, None)

    cancelled = asyncio.create_task(cache.async_wait_for_miss(key))
    survivor = asyncio.create_task(cache.async_wait_for_miss(key))
    loop = asyncio.get_running_loop()
    loop.call_soon(cancelled.cancel)
    loop.call_soon(cache.set, key, ["candidate"])
    cancelled_result, survivor_result = await asyncio.gather(
        cancelled,
        survivor,
        return_exceptions=True,
    )

    assert isinstance(cancelled_result, asyncio.CancelledError)
    assert survivor_result == (False, ["candidate"])


@pytest.mark.asyncio
async def test_cross_thread_completion_reaches_another_event_loop() -> None:
    """A synchronous completion in one thread wakes a waiter on another loop."""
    cache: CandidateResultCache[list[str]] = CandidateResultCache()
    key = _candidate_key("cross-thread")
    owner_started = Event()
    release_owner = Event()
    owner_errors: list[BaseException] = []

    def run_owner() -> None:
        async def owner() -> None:
            try:
                assert await cache.async_wait_for_miss(key) == (True, None)
                owner_started.set()
                if not release_owner.wait(timeout=1.0):
                    raise AssertionError("owner release was not signaled")
                cache.set(key, ["cross-thread"])
            except AssertionError as error:
                owner_errors.append(error)

        asyncio.run(owner())

    owner_thread = Thread(target=run_owner)
    owner_thread.start()
    assert owner_started.wait(timeout=1.0)
    follower = asyncio.create_task(cache.async_wait_for_miss(key))
    asyncio.get_running_loop().call_soon(release_owner.set)
    try:
        assert await asyncio.wait_for(follower, timeout=1.0) == (
            False,
            ["cross-thread"],
        )
    finally:
        owner_thread.join(timeout=1.0)
    assert not owner_thread.is_alive()
    assert owner_errors == []


@pytest.mark.asyncio
async def test_stale_owner_failure_does_not_remove_new_flight() -> None:
    """A delayed old-owner failure cannot release a newer single flight."""
    cache: CandidateResultCache[list[str]] = CandidateResultCache()
    key = _candidate_key("stale-flight")
    claimed = asyncio.Event()
    release = asyncio.Event()

    async def old_owner() -> None:
        assert await cache.async_wait_for_miss(key) == (True, None)
        claimed.set()
        await release.wait()
        raise RuntimeError("old owner failed")

    old_owner_task = asyncio.create_task(old_owner())
    await claimed.wait()
    cache.fail(key, RuntimeError("external failure"))
    assert await cache.async_wait_for_miss(key) == (True, None)
    release.set()
    with pytest.raises(RuntimeError, match="old owner failed"):
        await old_owner_task

    follower = asyncio.create_task(cache.async_wait_for_miss(key))
    asyncio.get_running_loop().call_soon(cache.set, key, ["new flight"])
    assert await follower == (False, ["new flight"])


@pytest.mark.asyncio
async def test_hydration_cache_single_flight_caches_missing_records() -> None:
    """Concurrent missing-record hydration runs once and caches ``None``."""
    cache: HydrationCache[Record] = HydrationCache()
    key = HydrationCacheKey.build(
        RecordIdentity("workspace", "note", "missing"),
        record_version=1,
        policy_version="policy/v1",
    )
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def compute() -> None:
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()

    first = asyncio.create_task(cache.async_get_or_compute(key, compute))
    await started.wait()
    second = asyncio.create_task(cache.async_get_or_compute(key, compute))
    release.set()
    assert await asyncio.gather(first, second) == [None, None]
    assert calls == 1
    assert cache.lookup(key) == (True, None)
    assert cache.metrics.coalesced_waiters == 1


@pytest.mark.asyncio
async def test_cache_failure_releases_single_flight_waiters() -> None:
    candidate_cache: CandidateResultCache[list[str]] = CandidateResultCache()
    candidate_key = CandidateCacheKey.build(
        query="candidate-failure",
        filters={},
        requested_limit=1,
        acquisition_limit=1,
        adaptive_limit=None,
        routing_fingerprint="r",
        encoder_namespace=None,
        epochs=SearchEpochs(),
        policy_version=None,
    )
    assert await candidate_cache.async_wait_for_miss(candidate_key) == (True, None)
    candidate_waiter = asyncio.create_task(
        candidate_cache.async_wait_for_miss(candidate_key)
    )
    candidate_error = RuntimeError("candidate failure")
    asyncio.get_running_loop().call_soon(
        candidate_cache.fail,
        candidate_key,
        candidate_error,
    )
    with pytest.raises(RuntimeError, match="candidate failure"):
        await candidate_waiter

    hydration_cache: HydrationCache[Record] = HydrationCache()
    hydration_key = HydrationCacheKey.build(
        RecordIdentity(None, "note", "hydration-failure"),
        record_version=1,
        policy_version="policy/v1",
    )
    assert await hydration_cache.async_wait_for_miss(hydration_key) == (True, None)
    hydration_waiter = asyncio.create_task(
        hydration_cache.async_wait_for_miss(hydration_key)
    )
    hydration_error = RuntimeError("hydration failure")
    asyncio.get_running_loop().call_soon(
        hydration_cache.fail,
        hydration_key,
        hydration_error,
    )
    with pytest.raises(RuntimeError, match="hydration failure"):
        await hydration_waiter


@pytest.mark.asyncio
async def test_owner_failure_releases_unfinished_single_flight_waiters() -> None:
    cache: CandidateResultCache[list[str]] = CandidateResultCache()
    key = CandidateCacheKey.build(
        query="owner-failure",
        filters={},
        requested_limit=1,
        acquisition_limit=1,
        adaptive_limit=None,
        routing_fingerprint="r",
        encoder_namespace=None,
        epochs=SearchEpochs(),
        policy_version=None,
    )

    claimed = asyncio.Event()
    release = asyncio.Event()

    async def owner() -> None:
        assert await cache.async_wait_for_miss(key) == (True, None)
        claimed.set()
        await release.wait()
        raise RuntimeError("owner failed")

    first = asyncio.create_task(owner())
    await claimed.wait()
    second = asyncio.create_task(cache.async_wait_for_miss(key))
    asyncio.get_running_loop().call_soon(release.set)
    with pytest.raises(RuntimeError, match="owner failed"):
        await first
    with pytest.raises(RuntimeError, match="owner failed"):
        await second


@pytest.mark.asyncio
async def test_record_pipeline_warm_candidates_skip_retrieval_until_epoch_changes() -> None:
    class Keyword:
        def __init__(self) -> None:
            self.calls = 0
            self.current_epoch = 0

        def search(
            self,
            query: str,
            k: int,
            filters: dict[str, Any] | None = None,
        ) -> list[RecordHit]:
            self.calls += 1
            return [RecordHit(RecordIdentity(None, "note", "record"), 1.0)]

        def index(self, records: list[Record]) -> None:
            pass

        def keyword_epoch(self) -> int:
            return self.current_epoch

    record = Record(
        source_kind="note",
        source_id="record",
        title="title",
        body="body",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    keyword = Keyword()
    pipeline = RecordSearchPipeline(
        keyword_store=keyword,
        hydrator=lambda _identity: record,
    )
    first = await pipeline.async_search(" query ")
    second = await pipeline.async_search("query")
    assert len(first.results) == len(second.results) == 1
    assert keyword.calls == 1
    assert "candidate_cache:hit" in second.cache_diagnostics
    keyword.current_epoch = 1
    await pipeline.async_search("query")
    assert keyword.calls == 2


@pytest.mark.asyncio
async def test_record_pipeline_hydration_cache_uses_version_and_policy() -> None:
    class Hydrator:
        record_epoch = 1

        def __init__(self, record: Record) -> None:
            self.record = record
            self.calls = 0

        def hydrate_record(self, record_id: RecordIdentity) -> Record:
            assert record_id.source_id == "record"
            self.calls += 1
            return self.record

    record = Record(
        source_kind="note",
        source_id="record",
        title="title",
        body="body",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    hydrator = Hydrator(record)

    class Keyword:
        def index(self, records: list[Record]) -> None:
            pass

        def search(
            self,
            query: str,
            k: int,
            filters: dict[str, Any] | None = None,
        ) -> list[RecordHit]:
            return [RecordHit(RecordIdentity(None, "note", "record"), 1.0)]

    keyword = Keyword()
    pipeline = RecordSearchPipeline(
        keyword_store=keyword,
        hydrator=hydrator,
        hydration_cache=HydrationCache(),
        policy_version="policy/v1",
    )
    await pipeline.async_search("query")
    await pipeline.async_search("query")
    assert hydrator.calls == 1
    hydrator.record_epoch = 2
    await pipeline.async_search("query")
    assert hydrator.calls == 2


@pytest.mark.asyncio
async def test_record_pipeline_uses_batch_hydration_versions_before_scalar_provider() -> None:
    record = Record(
        source_kind="note",
        source_id="record",
        title="title",
        body="body",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )

    class Hydrator:
        def __init__(self) -> None:
            self.hydrate_calls = 0

        def hydrate_record(self, identity: RecordIdentity) -> Record:
            self.hydrate_calls += 1
            return record

    class VersionProvider:
        def __init__(self) -> None:
            self.batch_calls = 0
            self.scalar_calls = 0

        def hydration_versions(
            self, identities: list[RecordIdentity]
        ) -> dict[str, object]:
            self.batch_calls += 1
            return {identity.storage_key: 1 for identity in identities}

        def __call__(self, identity: RecordIdentity) -> object:
            self.scalar_calls += 1
            return 99

    class Keyword:
        def search(
            self,
            query: str,
            k: int,
            filters: dict[str, Any] | None = None,
        ) -> list[RecordHit]:
            return [RecordHit(RecordIdentity(None, "note", "record"), 1.0)]

    hydrator = Hydrator()
    version_provider = VersionProvider()
    pipeline = RecordSearchPipeline(
        keyword_store=Keyword(),
        hydrator=hydrator,
        hydration_cache=HydrationCache(),
        policy_version="policy/v1",
        hydration_version_provider=version_provider,
    )

    first = await pipeline.async_search("query")
    second = await pipeline.async_search("query")

    assert len(first.results) == len(second.results) == 1
    assert version_provider.batch_calls == 2
    assert version_provider.scalar_calls == 0
    assert hydrator.hydrate_calls == 1
