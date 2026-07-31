import asyncio
import threading
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import cast

import pytest

from searchkernel.domain import (
    GraphEdge,
    GraphNeighbor,
    Record,
    RecordHit,
    RecordIdentity,
)
from searchkernel.search.record_pipeline import (
    RecordSearchConfig,
    RecordSearchError,
    RecordSearchOutcome,
    RecordSearchPipeline,
    RecordSearchPolicy,
)

pytestmark = pytest.mark.asyncio


def _record(record_id: str) -> Record:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    return Record(
        source_kind="fake",
        source_id=record_id,
        title=record_id,
        body=f"body for {record_id}",
        created_at=timestamp,
        updated_at=timestamp,
    )


class FakeKeywordStore:
    def __init__(self, results: list[RecordHit | tuple[str, float]]) -> None:
        self.results = results
        self.queries: list[tuple[str, int, dict[str, object] | None]] = []

    def index(self, records: list[Record]) -> None:
        pass

    def search(
        self,
        query: str,
        k: int,
        filters: dict[str, object] | None = None,
    ) -> list[RecordHit | tuple[str, float]]:
        self.queries.append((query, k, filters))
        return self.results


class FakeVectorStore:
    def __init__(self, results: list[RecordHit | tuple[str, float]]) -> None:
        self.results = results
        self.filters: list[dict[str, object] | None] = []

    def upsert(self, records: list[Record], model_name: str, dim: int) -> None:
        pass

    def search(
        self,
        query_vector: list[float],
        k: int,
        *,
        model_name: str,
        dim: int,
        filters: dict[str, object] | None = None,
    ) -> list[RecordHit | tuple[str, float]]:
        assert query_vector == [1.0, 0.0]
        assert (model_name, dim) == ("fake-model", 2)
        self.filters.append(filters)
        return self.results

    def delete(self, record_ids: list[str]) -> None:
        pass

    def epoch(self) -> int:
        return 0


class FakeGraphStore:
    def __init__(
        self,
        neighbors: dict[str, list[GraphNeighbor | tuple[str, str, float]]],
    ) -> None:
        self._neighbors = neighbors

    def upsert_edges(
        self,
        edges: Sequence[GraphEdge | tuple[str, str, str, float]],
    ) -> None:
        pass

    def neighbors(
        self,
        record_id: RecordIdentity | str,
        edge_types: list[str] | None = None,
        depth: int = 1,
    ) -> list[GraphNeighbor | tuple[str, str, float]]:
        key = record_id.source_id if isinstance(record_id, RecordIdentity) else record_id
        return self._neighbors.get(key, [])


class FakeEmbedder:
    model_name = "fake-model"
    dim = 2

    def embed_query(self, query: str) -> list[float]:
        assert query == "query"
        return [1.0, 0.0]


def _hydrator(records: dict[str, Record]):
    def hydrate(record_id: RecordIdentity | str) -> Record | None:
        key = record_id.source_id if isinstance(record_id, RecordIdentity) else record_id
        return records.get(key)

    return hydrate


async def test_keyword_only_hydrates_records_with_deterministic_ties() -> None:
    records = {record_id: _record(record_id) for record_id in ("a", "b")}
    pipeline = RecordSearchPipeline(
        keyword_store=FakeKeywordStore([("b", 1.0), ("a", 1.0)]),
        hydrator=_hydrator(records),
    )

    outcome = await pipeline.async_search("query", limit=2)

    assert [result.record_id for result in outcome.results] == ["a", "b"]
    assert all(result.record.source_kind == "fake" for result in outcome.results)
    assert outcome.results[0].provenance.strategies == ("keyword",)


async def test_minimum_candidate_limit_applies_to_store_acquisition() -> None:
    records = {"a": _record("a")}
    keyword_store = FakeKeywordStore([("a", 1.0)])
    pipeline = RecordSearchPipeline(
        keyword_store=keyword_store,
        hydrator=_hydrator(records),
        config=RecordSearchConfig(minimum_candidate_limit=50),
    )

    await pipeline.async_search("query", limit=1)

    assert keyword_store.queries[0][1] == 50


async def test_hybrid_search_fuses_keyword_and_vector_rankings() -> None:
    records = {record_id: _record(record_id) for record_id in ("a", "b", "c")}
    pipeline = RecordSearchPipeline(
        keyword_store=FakeKeywordStore([("b", 10.0), ("a", 1.0)]),
        vector_store=FakeVectorStore([("a", 0.9), ("c", 0.8)]),
        embedding_provider=FakeEmbedder(),
        hydrator=_hydrator(records),
    )

    outcome = await pipeline.async_search("query", limit=3)

    assert [result.record_id for result in outcome.results] == ["a", "b", "c"]
    assert outcome.results[0].provenance.strategies == ("keyword", "vector")


async def test_policy_can_bound_vector_acquisition_to_keyword_candidates() -> None:
    records = {record_id: _record(record_id) for record_id in ("a", "b")}
    vector_store = FakeVectorStore([("a", 0.9)])
    pipeline = RecordSearchPipeline(
        keyword_store=FakeKeywordStore([("b", 1.0), ("a", 0.5)]),
        vector_store=vector_store,
        embedding_provider=FakeEmbedder(),
        hydrator=_hydrator(records),
        policy=RecordSearchPolicy(
            vector_candidate_ids=lambda ranking, filters: [record_id for record_id, _ in ranking]
        ),
    )

    await pipeline.async_search("query", limit=2, filters={"workspace_id": "workspace-1"})

    assert vector_store.filters == [
        {
            "workspace_id": "workspace-1",
            "statuses": ["active"],
            "candidate_ids": ["b", "a"],
        }
    ]


async def test_policy_can_order_vector_candidates_before_fusion() -> None:
    records = {record_id: _record(record_id) for record_id in ("a", "b")}
    vector_store = FakeVectorStore([("a", 0.9), ("b", 0.8)])
    pipeline = RecordSearchPipeline(
        vector_store=vector_store,
        embedding_provider=FakeEmbedder(),
        hydrator=_hydrator(records),
        policy=RecordSearchPolicy(
            vector_ranking_order=lambda ranking, filters: list(reversed(ranking))
        ),
    )

    outcome = await pipeline.async_search("query", limit=2)

    assert [result.record_id for result in outcome.results] == ["b", "a"]


async def test_candidate_filter_runs_before_graph_expansion() -> None:
    records = {record_id: _record(record_id) for record_id in ("seed", "blocked", "allowed")}
    pipeline = RecordSearchPipeline(
        keyword_store=FakeKeywordStore([("seed", 1.0)]),
        graph_store=FakeGraphStore(
            {"seed": [("blocked", "related", 1.0), ("allowed", "related", 0.5)]}
        ),
        hydrator=_hydrator(records),
        policy=RecordSearchPolicy(
            candidate_filter=lambda candidate: candidate.record_id != "blocked"
        ),
    )

    outcome = await pipeline.async_search("query", limit=3)

    assert [result.record_id for result in outcome.results] == ["seed", "allowed"]
    assert "blocked" not in {result.record_id for result in outcome.results}
    assert "graph" in outcome.results[-1].provenance.strategies


async def test_policy_can_adjust_scores_reject_results_and_post_process() -> None:
    records = {record_id: _record(record_id) for record_id in ("a", "b")}
    pipeline = RecordSearchPipeline(
        keyword_store=FakeKeywordStore([("a", 1.0), ("b", 0.5)]),
        hydrator=_hydrator(records),
        policy=RecordSearchPolicy(
            score_adjuster=lambda candidate: candidate.score
            + (1.0 if candidate.record_id == "b" else 0.0),
            result_filter=lambda result: result.record_id != "a",
            post_process=lambda results: list(reversed(results)),
        ),
    )

    outcome = await pipeline.async_search("query", limit=2)

    assert [result.record_id for result in outcome.results] == ["b"]
    assert outcome.results[0].score > 0


async def test_graph_expansion_is_bounded_and_missing_records_are_reported() -> None:
    records = {"seed": _record("seed")}
    pipeline = RecordSearchPipeline(
        keyword_store=FakeKeywordStore([("seed", 1.0)]),
        graph_store=FakeGraphStore(
            {
                "seed": [
                    ("missing", "related", 1.0),
                    ("other", "related", 0.9),
                ]
            }
        ),
        hydrator=_hydrator(records),
        config=RecordSearchConfig(max_neighbors_per_seed=1),
        continue_on_error=True,
    )

    outcome = await pipeline.async_search("query", limit=3)

    assert [result.record_id for result in outcome.results] == ["seed"]
    assert outcome.missing_record_ids == ("missing",)
    assert outcome.degraded


async def test_graph_expansion_reads_only_bounded_seed_neighbors() -> None:
    records = {record_id: _record(record_id) for record_id in ("a", "b", "c")}
    calls: list[RecordIdentity] = []

    class Graph:
        def upsert_edges(
            self,
            edges: Sequence[GraphEdge | tuple[str, str, str, float]],
        ) -> None:
            pass

        def neighbors(
            self,
            record_id: RecordIdentity | str,
            edge_types: list[str] | None = None,
            depth: int = 1,
        ) -> list[GraphNeighbor | tuple[str, str, float]]:
            assert isinstance(record_id, RecordIdentity)
            calls.append(record_id)
            return [("missing", "related", 1.0)]

    pipeline = RecordSearchPipeline(
        keyword_store=FakeKeywordStore([("a", 1.0), ("b", 0.9), ("c", 0.8)]),
        graph_store=Graph(),
        hydrator=_hydrator(records),
        config=RecordSearchConfig(max_graph_seeds=2),
        continue_on_error=True,
    )

    await pipeline.async_search("query", limit=3)

    assert sorted(identity.source_id for identity in calls) == ["a", "b"]


async def test_callable_embedding_provider_accepts_explicit_vector_metadata() -> None:
    records = {"a": _record("a")}
    pipeline = RecordSearchPipeline(
        vector_store=FakeVectorStore([("a", 1.0)]),
        embedding_provider=lambda query: [1.0, 0.0],
        embedding_model_name="fake-model",
        embedding_dim=2,
        hydrator=_hydrator(records),
    )

    outcome = await pipeline.async_search("query")
    assert [result.record_id for result in outcome.results] == ["a"]


async def test_store_errors_raise_by_default() -> None:
    class BrokenKeywordStore(FakeKeywordStore):
        def search(
            self,
            query: str,
            k: int,
            filters: dict[str, object] | None = None,
        ) -> list[RecordHit | tuple[str, float]]:
            raise RuntimeError("backend unavailable")

    pipeline = RecordSearchPipeline(
        keyword_store=BrokenKeywordStore([]),
        hydrator=_hydrator({}),
    )

    with pytest.raises(RecordSearchError, match="keyword retrieval failed"):

        await pipeline.async_search("query")


async def test_store_errors_can_be_explicitly_returned_as_degraded() -> None:
    class BrokenKeywordStore(FakeKeywordStore):
        def search(
            self,
            query: str,
            k: int,
            filters: dict[str, object] | None = None,
        ) -> list[RecordHit | tuple[str, float]]:
            raise RuntimeError("backend unavailable")

    pipeline = RecordSearchPipeline(
        keyword_store=BrokenKeywordStore([]),
        hydrator=_hydrator({}),
        continue_on_error=True,
    )

    outcome = await pipeline.async_search("query")

    assert not outcome.results
    assert outcome.degraded
    assert outcome.failures[0].stage == "keyword"


async def test_search_keeps_sync_compatibility_outside_event_loop() -> None:
    records = {"a": _record("a")}
    pipeline = RecordSearchPipeline(
        keyword_store=FakeKeywordStore([("a", 1.0)]),
        hydrator=_hydrator(records),
    )

    outcome = await asyncio.to_thread(
        lambda: cast(RecordSearchOutcome, pipeline.search("query"))
    )

    assert outcome.results[0].record_id == "a"


@pytest.mark.asyncio
async def test_composite_identity_reaches_async_hydrator() -> None:
    identity = RecordIdentity("workspace-a", "note", "note-1")
    expected_identity = identity
    record = Record(
        workspace_id="workspace-a",
        source_kind="note",
        source_id="note-1",
        title="note-1",
        body="body for note-1",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )

    class Hydrator:
        def hydrate_record(self, record_id: RecordIdentity) -> Record:
            assert record_id == expected_identity
            return record

    class Store:
        def index(self, records: list[Record]) -> None:
            pass

        def search(
            self,
            query: str,
            k: int,
            filters: dict[str, object] | None = None,
        ) -> list[RecordHit | tuple[str, float]]:
            return [RecordHit(identity, 1.0)]

    pipeline = RecordSearchPipeline(
        keyword_store=Store(),
        hydrator=Hydrator(),
    )

    outcome = await pipeline.async_search("query")

    assert outcome.results[0].storage_key == identity.storage_key
    assert outcome.results[0].provenance.record_identity == identity


@pytest.mark.asyncio
async def test_graph_neighbors_preserve_canonical_identity() -> None:
    seed = RecordIdentity("workspace-a", "note", "seed")
    target = RecordIdentity("workspace-b", "commit", "target")
    records = {
        "seed": Record(
            workspace_id=seed.workspace_id,
            source_kind=seed.source_kind,
            source_id=seed.source_id,
            title="seed",
            body="seed",
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            updated_at=datetime(2026, 1, 1, tzinfo=UTC),
        ),
        "target": Record(
            workspace_id=target.workspace_id,
            source_kind=target.source_kind,
            source_id=target.source_id,
            title="target",
            body="target",
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            updated_at=datetime(2026, 1, 1, tzinfo=UTC),
        ),
    }

    class Store:
        def index(self, records: list[Record]) -> None:
            pass

        def search(
            self,
            query: str,
            k: int,
            filters: dict[str, object] | None = None,
        ) -> list[RecordHit | tuple[str, float]]:
            return [RecordHit(seed, 1.0)]

    class Graph:
        def upsert_edges(
            self,
            edges: Sequence[GraphEdge | tuple[str, str, str, float]],
        ) -> None:
            pass

        def neighbors(
            self,
            record_id: RecordIdentity | str,
            edge_types: list[str] | None = None,
            depth: int = 1,
        ) -> list[GraphNeighbor | tuple[str, str, float]]:
            return [GraphNeighbor(target, "related", 1.0)]

    pipeline = RecordSearchPipeline(
        keyword_store=Store(),
        graph_store=Graph(),
        hydrator=_hydrator(records),
    )

    outcome = await pipeline.async_search("query", limit=2)

    assert [result.storage_key for result in outcome.results] == [
        seed.storage_key,
        target.storage_key,
    ]


async def test_keyword_and_embedding_work_overlap_without_candidate_gating() -> None:
    records = {"a": _record("a")}
    keyword_started = asyncio.Event()
    embedding_started = asyncio.Event()
    release = asyncio.Event()
    vector_started = asyncio.Event()

    class Keyword:
        async def search(
            self,
            query: str,
            k: int,
            filters: dict[str, object] | None = None,
        ) -> list[RecordHit | tuple[str, float]]:
            keyword_started.set()
            await release.wait()
            return [("a", 1.0)]

    class Embedder:
        model_name = "fake-model"
        dim = 2

        async def embed_query(self, text: str) -> list[float]:
            embedding_started.set()
            await release.wait()
            return [1.0, 0.0]

    class Vector:
        async def search(
            self,
            query_vector: list[float],
            k: int,
            *,
            model_name: str,
            dim: int,
            filters: dict[str, object] | None = None,
        ) -> list[RecordHit | tuple[str, float]]:
            vector_started.set()
            return [("a", 1.0)]

    pipeline = RecordSearchPipeline(
        keyword_store=Keyword(),
        vector_store=Vector(),
        embedding_provider=Embedder(),
        hydrator=_hydrator(records),
    )
    task = asyncio.create_task(pipeline.async_search("query", limit=1))
    await asyncio.wait_for(
        asyncio.gather(keyword_started.wait(), embedding_started.wait()),
        timeout=1,
    )
    assert not vector_started.is_set()
    release.set()
    await task
    assert vector_started.is_set()


async def test_candidate_gating_delays_vector_lookup_until_keyword_ids_arrive() -> None:
    keyword_started = asyncio.Event()
    keyword_finished = asyncio.Event()
    embedding_started = asyncio.Event()
    release_keyword = asyncio.Event()
    vector_started = asyncio.Event()

    class Keyword:
        async def search(
            self,
            query: str,
            k: int,
            filters: dict[str, object] | None = None,
        ) -> list[RecordHit | tuple[str, float]]:
            keyword_started.set()
            await release_keyword.wait()
            keyword_finished.set()
            return [("a", 1.0)]

    class Embedder:
        model_name = "fake-model"
        dim = 2

        async def embed_query(self, text: str) -> list[float]:
            embedding_started.set()
            return [1.0, 0.0]

    class Vector:
        async def search(
            self,
            query_vector: list[float],
            k: int,
            *,
            model_name: str,
            dim: int,
            filters: dict[str, object] | None = None,
        ) -> list[RecordHit | tuple[str, float]]:
            assert keyword_finished.is_set()
            vector_started.set()
            return [("a", 1.0)]

    pipeline = RecordSearchPipeline(
        keyword_store=Keyword(),
        vector_store=Vector(),
        embedding_provider=Embedder(),
        hydrator=_hydrator({"a": _record("a")}),
        policy=RecordSearchPolicy(
            vector_candidate_ids=lambda ranking, filters: [
                record_id for record_id, _score in ranking
            ]
        ),
    )
    task = asyncio.create_task(pipeline.async_search("query", limit=1))
    await asyncio.wait_for(
        asyncio.gather(keyword_started.wait(), embedding_started.wait()),
        timeout=1,
    )
    assert not vector_started.is_set()
    release_keyword.set()
    await task
    assert vector_started.is_set()


async def test_artifact_keyword_confidence_skips_embedding() -> None:
    class CountingEmbedder:
        model_name = "fake-model"
        dim = 2

        def __init__(self) -> None:
            self.calls = 0

        def embed_query(self, query: str) -> list[float]:
            self.calls += 1
            return [1.0, 0.0]

    embedder = CountingEmbedder()
    pipeline = RecordSearchPipeline(
        keyword_store=FakeKeywordStore([("a", 1.0)]),
        vector_store=FakeVectorStore([("a", 0.9)]),
        embedding_provider=embedder,
        hydrator=_hydrator({"a": _record("a")}),
    )

    outcome = await pipeline.async_search("src/search_kernel.py", limit=1)

    assert [result.record_id for result in outcome.results] == ["a"]
    assert embedder.calls == 0
    assert "vector:artifact_keyword_confident" in outcome.diagnostics


async def test_lane_budgets_reach_their_respective_stores() -> None:
    class AnyEmbedder(FakeEmbedder):
        def embed_query(self, query: str) -> list[float]:
            return [1.0, 0.0]

    keyword_store = FakeKeywordStore([("a", 1.0)])
    vector_store = FakeVectorStore([("a", 0.9)])
    pipeline = RecordSearchPipeline(
        keyword_store=keyword_store,
        vector_store=vector_store,
        embedding_provider=AnyEmbedder(),
        hydrator=_hydrator({"a": _record("a")}),
        config=RecordSearchConfig(
            keyword_candidate_multiplier=2,
            vector_candidate_multiplier=3,
        ),
    )

    await pipeline.async_search("what is caching?", limit=2)

    assert keyword_store.queries[0][1] == 4
    assert vector_store.filters[0] == {"statuses": ["active"]}


async def test_graph_disabled_does_not_touch_graph_store() -> None:
    class FailingGraph(FakeGraphStore):
        def neighbors(
            self,
            record_id: RecordIdentity | str,
            edge_types: list[str] | None = None,
            depth: int = 1,
        ) -> list[GraphNeighbor | tuple[str, str, float]]:
            raise AssertionError("graph should be disabled")

    pipeline = RecordSearchPipeline(
        keyword_store=FakeKeywordStore([("a", 1.0)]),
        graph_store=FailingGraph({}),
        hydrator=_hydrator({"a": _record("a")}),
        config=RecordSearchConfig(graph_enabled=False),
    )

    outcome = await pipeline.async_search("what is caching?", limit=1)

    assert outcome.results[0].record_id == "a"
    assert "query_plan:skip:graph:disabled" in outcome.diagnostics


async def test_conditional_expansion_is_called_once_after_weak_first_pass() -> None:
    class AnyEmbedder(FakeEmbedder):
        def embed_query(self, query: str) -> list[float]:
            return [1.0, 0.0]

    class ExpandingKeyword(FakeKeywordStore):
        def search(
            self,
            query: str,
            k: int,
            filters: dict[str, object] | None = None,
        ) -> list[RecordHit | tuple[str, float]]:
            self.queries.append((query, k, filters))
            return [("a", 1.0)] if query == "what is thing?" else [("b", 0.9)]

    class ExpandingVector(FakeVectorStore):
        def __init__(self) -> None:
            super().__init__([])
            self.expansion_calls = 0

        def expand_query(
            self,
            query: str,
            *,
            top_k: int,
            similarity_threshold: float,
        ) -> str:
            self.expansion_calls += 1
            return f"{query} expanded"

    keyword_store = ExpandingKeyword([])
    vector_store = ExpandingVector()
    records = {"a": _record("a"), "b": _record("b")}
    pipeline = RecordSearchPipeline(
        keyword_store=keyword_store,
        vector_store=vector_store,
        embedding_provider=AnyEmbedder(),
        hydrator=_hydrator(records),
        config=RecordSearchConfig(expansion_enabled=True),
    )

    outcome = await pipeline.async_search("what is thing?", limit=2)

    assert vector_store.expansion_calls == 1
    assert [result.record_id for result in outcome.results] == ["a", "b"]
    assert "expansion:applied" in outcome.diagnostics


async def test_conditional_expansion_obeys_latency_budget() -> None:
    class AnyEmbedder(FakeEmbedder):
        def embed_query(self, query: str) -> list[float]:
            return [1.0, 0.0]

    class SlowExpandingVector(FakeVectorStore):
        async def expand_query(
            self,
            query: str,
            *,
            top_k: int,
            similarity_threshold: float,
        ) -> str:
            await asyncio.sleep(0.05)
            return f"{query} expanded"

    pipeline = RecordSearchPipeline(
        keyword_store=FakeKeywordStore([]),
        vector_store=SlowExpandingVector([]),
        embedding_provider=AnyEmbedder(),
        hydrator=_hydrator({}),
        config=RecordSearchConfig(
            expansion_enabled=True,
            expansion_timeout_s=0.01,
        ),
    )

    started = time.perf_counter()
    outcome = await pipeline.async_search("what is thing?", limit=1)
    elapsed = time.perf_counter() - started

    assert elapsed < 0.05
    assert "expansion:fallback:timeout" in outcome.diagnostics


async def test_unsupported_expansion_is_reported_without_backend_call() -> None:
    class AnyEmbedder(FakeEmbedder):
        def embed_query(self, query: str) -> list[float]:
            return [1.0, 0.0]

    class UnsupportedVector(FakeVectorStore):
        query_expansion_supported = False

        def expand_query(
            self,
            query: str,
            *,
            top_k: int,
            similarity_threshold: float,
        ) -> str:
            raise AssertionError("unsupported expansion must be bypassed")

    pipeline = RecordSearchPipeline(
        keyword_store=FakeKeywordStore([]),
        vector_store=UnsupportedVector([]),
        embedding_provider=AnyEmbedder(),
        hydrator=_hydrator({}),
        config=RecordSearchConfig(expansion_enabled=True),
    )

    outcome = await pipeline.async_search("what is thing?", limit=1)

    assert "expansion:skip:unsupported" in outcome.diagnostics


async def test_trace_is_redacted_and_contains_routing_diagnostics() -> None:
    pipeline = RecordSearchPipeline(
        keyword_store=FakeKeywordStore([("a", 1.0)]),
        hydrator=_hydrator({"a": _record("a")}),
        config=RecordSearchConfig(capture_trace=True),
    )

    outcome = await pipeline.async_search("secret query", limit=1)

    assert outcome.trace is not None
    trace = outcome.trace.to_dict()
    assert "query" not in trace
    assert "diagnostics" in trace["provenance"]


async def test_rerank_runs_once_with_a_bounded_candidate_set() -> None:
    class Reranker:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def rerank(self, query: str, documents: list[str]) -> list[float]:
            self.calls.append(documents)
            return [0.2, 0.9]

    reranker = Reranker()
    records = {record_id: _record(record_id) for record_id in ("a", "b")}
    pipeline = RecordSearchPipeline(
        keyword_store=FakeKeywordStore([("a", 1.0), ("b", 0.9)]),
        hydrator=_hydrator(records),
        reranker=reranker,
        config=RecordSearchConfig(rerank_budget=2),
    )

    outcome = await pipeline.async_search("query", limit=2)

    assert len(reranker.calls) == 1
    assert len(reranker.calls[0]) == 2
    assert [result.record_id for result in outcome.results] == ["b", "a"]
    assert "rerank:applied:2" in outcome.diagnostics


async def test_rerank_failure_falls_back_deterministically_in_lenient_mode() -> None:
    class FailingReranker:
        def rerank(self, query: str, documents: list[str]) -> list[float]:
            raise RuntimeError("reranker unavailable")

    records = {record_id: _record(record_id) for record_id in ("a", "b")}
    pipeline = RecordSearchPipeline(
        keyword_store=FakeKeywordStore([("b", 1.0), ("a", 1.0)]),
        hydrator=_hydrator(records),
        reranker=FailingReranker(),
        config=RecordSearchConfig(rerank_budget=2, failure_mode="lenient"),
    )

    outcome = await pipeline.async_search("query", limit=2)

    assert [result.record_id for result in outcome.results] == ["a", "b"]
    assert outcome.failures[0].stage == "rerank"
    assert "rerank:fallback:RuntimeError" in outcome.diagnostics


async def test_batch_graph_and_hydration_use_canonical_keys_once() -> None:
    seed = RecordIdentity("workspace-a", "note", "seed")
    target = RecordIdentity("workspace-b", "commit", "target")
    records = {
        seed.storage_key: Record(
            workspace_id=seed.workspace_id,
            source_kind=seed.source_kind,
            source_id=seed.source_id,
            title="seed",
            body="seed",
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            updated_at=datetime(2026, 1, 1, tzinfo=UTC),
        ),
        target.storage_key: Record(
            workspace_id=target.workspace_id,
            source_kind=target.source_kind,
            source_id=target.source_id,
            title="target",
            body="target",
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            updated_at=datetime(2026, 1, 1, tzinfo=UTC),
        ),
    }

    class Store:
        def index(self, records: list[Record]) -> None:
            pass

        def search(
            self,
            query: str,
            k: int,
            filters: dict[str, object] | None = None,
        ) -> list[RecordHit]:
            return [RecordHit(seed, 1.0)]

    class Graph:
        def upsert_edges(
            self,
            edges: Sequence[GraphEdge | tuple[str, str, str, float]],
        ) -> None:
            pass

        def __init__(self) -> None:
            self.calls = 0
            self.identities: list[RecordIdentity] = []

        async def neighbors_many(
            self,
            identities: Sequence[RecordIdentity],
            *,
            depth: int,
        ) -> dict[str, list[GraphNeighbor]]:
            self.calls += 1
            self.identities = list(identities)
            return {
                seed.storage_key: [GraphNeighbor(target, "related", 1.0)]
            }

        def neighbors(
            self,
            record_id: RecordIdentity | str,
            edge_types: list[str] | None = None,
            depth: int = 1,
        ) -> list[GraphNeighbor]:
            raise AssertionError("scalar graph lookup should not run")

    class Hydrator:
        def __init__(self) -> None:
            self.calls = 0
            self.identities: list[RecordIdentity] = []

        async def hydrate_records(
            self,
            identities: Sequence[RecordIdentity],
        ) -> dict[str, Record | None]:
            self.calls += 1
            self.identities = list(identities)
            return {identity.storage_key: records[identity.storage_key] for identity in identities}

        def hydrate_record(self, record_id: RecordIdentity) -> Record | None:
            raise AssertionError("scalar hydration should not run")

    graph = Graph()
    hydrator = Hydrator()
    pipeline = RecordSearchPipeline(
        keyword_store=Store(),
        graph_store=graph,
        hydrator=hydrator,
    )

    outcome = await pipeline.async_search("query", limit=2)

    assert graph.calls == 1
    assert graph.identities == [seed]
    assert hydrator.calls == 1
    assert [identity.storage_key for identity in hydrator.identities] == [
        seed.storage_key,
        target.storage_key,
    ]
    assert [result.storage_key for result in outcome.results] == [
        seed.storage_key,
        target.storage_key,
    ]


async def test_scalar_and_batch_hydration_have_identical_results_and_provenance() -> None:
    records = {record_id: _record(record_id) for record_id in ("a", "b", "c")}
    keyword_results: list[RecordHit | tuple[str, float]] = [
        ("b", 1.0),
        ("a", 0.9),
    ]
    vector_results: list[RecordHit | tuple[str, float]] = [
        ("a", 0.8),
        ("c", 0.7),
    ]

    class BatchHydrator:
        async def hydrate_records(
            self,
            identities: Sequence[RecordIdentity],
        ) -> dict[str, Record | None]:
            return {
                identity.storage_key: records[identity.source_id]
                for identity in identities
            }

        def hydrate_record(self, record_id: RecordIdentity) -> Record | None:
            raise AssertionError("scalar hydration should not run")

    scalar = RecordSearchPipeline(
        keyword_store=FakeKeywordStore(keyword_results),
        vector_store=FakeVectorStore(vector_results),
        embedding_provider=FakeEmbedder(),
        hydrator=_hydrator(records),
    )
    batched = RecordSearchPipeline(
        keyword_store=FakeKeywordStore(keyword_results),
        vector_store=FakeVectorStore(vector_results),
        embedding_provider=FakeEmbedder(),
        hydrator=BatchHydrator(),
    )

    scalar_outcome = await scalar.async_search("query", limit=3)
    batch_outcome = await batched.async_search("query", limit=3)

    def signature(outcome: RecordSearchOutcome) -> list[tuple[str, float, dict]]:
        return [
            (
                result.storage_key,
                result.score,
                result.provenance.to_dict(),
            )
            for result in outcome.results
        ]

    assert signature(scalar_outcome) == signature(batch_outcome)


async def test_scalar_graph_fallback_is_bounded() -> None:
    records = {record_id: _record(record_id) for record_id in ("a", "b", "c", "d")}
    lock = threading.Lock()
    active = 0
    maximum_active = 0

    class Graph:
        def upsert_edges(
            self,
            edges: Sequence[GraphEdge | tuple[str, str, str, float]],
        ) -> None:
            pass

        def neighbors(
            self,
            record_id: RecordIdentity | str,
            edge_types: list[str] | None = None,
            depth: int = 1,
        ) -> list[GraphNeighbor | tuple[str, str, float]]:
            nonlocal active, maximum_active
            with lock:
                active += 1
                maximum_active = max(maximum_active, active)
            try:
                time.sleep(0.03)
                return []
            finally:
                with lock:
                    active -= 1

    pipeline = RecordSearchPipeline(
        keyword_store=FakeKeywordStore([(record_id, 1.0) for record_id in records]),
        graph_store=Graph(),
        hydrator=_hydrator(records),
        config=RecordSearchConfig(max_graph_concurrency=2),
    )

    await pipeline.async_search("query", limit=4)

    assert maximum_active == 2


async def test_batch_graph_failures_keep_strict_and_lenient_modes() -> None:
    class BrokenGraph:
        def upsert_edges(
            self,
            edges: Sequence[GraphEdge | tuple[str, str, str, float]],
        ) -> None:
            pass

        async def neighbors_many(
            self,
            identities: Sequence[RecordIdentity],
            *,
            depth: int,
        ) -> dict[str, list[GraphNeighbor]]:
            raise RuntimeError("graph unavailable")

        def neighbors(
            self,
            record_id: RecordIdentity | str,
            edge_types: list[str] | None = None,
            depth: int = 1,
        ) -> list[GraphNeighbor]:
            raise AssertionError("scalar graph lookup should not run")

    strict = RecordSearchPipeline(
        keyword_store=FakeKeywordStore([("a", 1.0)]),
        graph_store=BrokenGraph(),
        hydrator=_hydrator({"a": _record("a")}),
    )
    with pytest.raises(RecordSearchError, match="graph retrieval failed"):
        await strict.async_search("query")

    lenient = RecordSearchPipeline(
        keyword_store=FakeKeywordStore([("a", 1.0)]),
        graph_store=BrokenGraph(),
        hydrator=_hydrator({"a": _record("a")}),
        continue_on_error=True,
    )
    outcome = await lenient.async_search("query")
    assert [result.record_id for result in outcome.results] == ["a"]
    assert outcome.failures[0].stage == "graph"


async def test_cancelling_overlapped_lanes_cancels_both_tasks() -> None:
    keyword_started = asyncio.Event()
    embedding_started = asyncio.Event()
    keyword_cancelled = asyncio.Event()
    embedding_cancelled = asyncio.Event()
    wait_forever = asyncio.Event()

    class Keyword:
        async def search(
            self,
            query: str,
            k: int,
            filters: dict[str, object] | None = None,
        ) -> list[RecordHit | tuple[str, float]]:
            keyword_started.set()
            try:
                await wait_forever.wait()
            finally:
                keyword_cancelled.set()
            return []

    class Embedder:
        model_name = "fake-model"
        dim = 2

        async def embed_query(self, text: str) -> list[float]:
            embedding_started.set()
            try:
                await wait_forever.wait()
            finally:
                embedding_cancelled.set()
            return [1.0, 0.0]

    pipeline = RecordSearchPipeline(
        keyword_store=Keyword(),
        vector_store=FakeVectorStore([]),
        embedding_provider=Embedder(),
        hydrator=_hydrator({}),
    )
    task = asyncio.create_task(pipeline.async_search("query"))
    await asyncio.wait_for(
        asyncio.gather(keyword_started.wait(), embedding_started.wait()),
        timeout=1,
    )
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert keyword_cancelled.is_set()
    assert embedding_cancelled.is_set()


async def test_hydration_completion_order_does_not_change_results() -> None:
    records = {record_id: _record(record_id) for record_id in ("a", "b", "c")}

    class Hydrator:
        async def hydrate_record(self, record_id: RecordIdentity) -> Record:
            await asyncio.sleep({"a": 0.03, "b": 0.01, "c": 0.0}[record_id.source_id])
            return records[record_id.source_id]

    pipeline = RecordSearchPipeline(
        keyword_store=FakeKeywordStore([("c", 1.0), ("b", 1.0), ("a", 1.0)]),
        hydrator=Hydrator(),
    )

    outcome = await pipeline.async_search("query", limit=3)

    assert [result.record_id for result in outcome.results] == ["a", "b", "c"]
