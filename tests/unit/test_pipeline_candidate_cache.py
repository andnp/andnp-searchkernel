import asyncio
from datetime import UTC, datetime

import pytest

from searchkernel.domain import Record, RecordHit, RecordIdentity
from searchkernel.runtime import CandidateResultCache, HydrationCache
from searchkernel.search.pipeline_candidate_cache import CandidateCachePolicy
from searchkernel.search.record_pipeline import (
    RecordSearchConfig,
    RecordSearchPipeline,
    RecordSearchPolicy,
)


def _policy(store: object) -> CandidateCachePolicy[object]:
    return CandidateCachePolicy(
        CandidateResultCache(),
        config=RecordSearchConfig(),
        policy=RecordSearchPolicy(),
        keyword_store=store,
        vector_store=None,
        graph_store=None,
        embedding_provider=None,
        embedding_model_name=None,
        embedding_dim=None,
        encoder_namespace=None,
        routing_fingerprint="test",
        policy_version=None,
    )


def test_candidate_epochs_prefer_bulk_protocol() -> None:
    class BulkStore:
        bulk_calls = 0
        scalar_calls = 0

        def epochs(self) -> dict[str, int]:
            self.bulk_calls += 1
            return {"keyword": 3, "vector": 4, "graph": 5}

        def keyword_epoch(self) -> int:
            self.scalar_calls += 1
            return 99

    store = BulkStore()
    assert _policy(store)._cache_epochs().keyword == 3
    assert store.bulk_calls == 1
    assert store.scalar_calls == 0


def test_candidate_epochs_fall_back_to_scalar_protocol() -> None:
    class ScalarStore:
        def epochs(self) -> dict[str, int]:
            raise RuntimeError("bulk unavailable")

        def keyword_epoch(self) -> int:
            return 7

    assert _policy(ScalarStore())._cache_epochs().keyword == 7


@pytest.mark.asyncio
async def test_pipeline_coalesces_candidate_and_hydration_misses() -> None:
    record = Record(
        source_kind="note",
        source_id="record",
        title="title",
        body="body",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )

    class Keyword:
        calls = 0
        started = asyncio.Event()
        release = asyncio.Event()

        def keyword_epoch(self) -> int:
            return 0

        def search(self, query, k, filters=None):
            async def delayed():
                self.calls += 1
                self.started.set()
                await self.release.wait()
                return [RecordHit(RecordIdentity(None, "note", "record"), 1.0)]

            return delayed()

    class Hydrator:
        record_epoch = 1
        calls = 0
        started = asyncio.Event()
        release = asyncio.Event()

        async def hydrate_records(self, identities):
            self.calls += 1
            self.started.set()
            await self.release.wait()
            return {identities[0].storage_key: record}

    keyword = Keyword()
    hydrator = Hydrator()
    pipeline = RecordSearchPipeline(
        keyword_store=keyword,
        hydrator=hydrator,
        hydration_cache=HydrationCache(),
        policy_version="policy/v1",
    )
    first_task = asyncio.create_task(pipeline.async_search("query"))
    await keyword.started.wait()
    second_task = asyncio.create_task(pipeline.async_search("query"))
    await asyncio.sleep(0)
    keyword.release.set()
    await hydrator.started.wait()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    hydrator.release.set()
    first, second = await asyncio.gather(first_task, second_task)

    assert keyword.calls == 1
    assert hydrator.calls == 1
    assert len(first.results) == len(second.results) == 1
    diagnostics = [*first.cache_diagnostics, *second.cache_diagnostics]
    assert "candidate_cache:coalesced" in diagnostics
