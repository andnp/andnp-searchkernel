import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pytest

from searchkernel.domain import Record, RecordHit, RecordIdentity
from searchkernel.runtime import (
    CandidateCacheKey,
    CandidateResultCache,
    HydrationCache,
    HydrationCacheKey,
    QueryEmbeddingCache,
    SearchEpochs,
    UnstableCacheKey,
)
from searchkernel.search.record_pipeline import RecordSearchPipeline


@dataclass
class _Clock:
    value: float = 0.0

    def __call__(self) -> float:
        return self.value


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
