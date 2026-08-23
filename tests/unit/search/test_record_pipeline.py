import asyncio
import threading
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import cast

import pytest

from searchkernel.domain import (
    GraphEdge,
    GraphNeighbor,
    Record,
    RecordHit,
    RecordIdentity,
    SearchFilters,
    SearchResultProvenance,
    Vector,
)
from searchkernel.ports import KeywordStore, VectorStore
from searchkernel.ports.search_results import (
    MAX_FAILURE_DETAIL_LENGTH,
    RecordSearchOutcome,
    RecordSearchResult,
)
from searchkernel.runtime import (
    CandidateResultCache,
    HydrationCache,
    HydrationCacheKey,
)
from searchkernel.search.record_pipeline import (
    RecordSearchCandidate,
    RecordSearchConfig,
    RecordSearchError,
    RecordSearchPipeline,
    RecordSearchPolicy,
    RecordSearchQueryContext,
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


def _hit(record_id: str, score: float) -> RecordHit:
    return RecordHit(RecordIdentity(None, "fake", record_id), score)


def _source_record(source_kind: str, record_id: str) -> Record:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    return Record(
        source_kind=source_kind,
        source_id=record_id,
        title=record_id,
        body=f"body for {record_id}",
        created_at=timestamp,
        updated_at=timestamp,
    )


def _hits(results: Sequence[RecordHit | tuple[str, float]]) -> list[RecordHit]:
    return [
        result
        if isinstance(result, RecordHit)
        else _hit(result[0], result[1])
        for result in results
    ]


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("expected a mapping")
    return value


class FakeKeywordStore:
    def __init__(self, results: Sequence[RecordHit | tuple[str, float]]) -> None:
        self.results = _hits(results)
        self.queries: list[tuple[str, int, SearchFilters | None]] = []

    def index(self, records: list[Record]) -> None:
        pass

    def search(
        self,
        query: str,
        k: int,
        filters: SearchFilters | None = None,
    ) -> Sequence[RecordHit]:
        self.queries.append((query, k, filters))
        return self.results


class FakeVectorStore:
    def __init__(self, results: Sequence[RecordHit | tuple[str, float]]) -> None:
        self.results = _hits(results)
        self.filters: list[SearchFilters | None] = []

    def upsert(self, records: list[Record], model_name: str, dim: int) -> None:
        pass

    def search(
        self,
        query_vector: Vector,
        k: int,
        *,
        model_name: str,
        dim: int,
        filters: SearchFilters | None = None,
    ) -> Sequence[RecordHit]:
        assert query_vector == [1.0, 0.0]
        assert (model_name, dim) == ("fake-model", 2)
        self.filters.append(filters)
        return self.results

    def delete(self, record_ids: list[str]) -> None:
        pass

    def epoch(self) -> int:
        return 0


class FakeKeywordIndex:
    def index(self, records: list[Record]) -> None:
        pass


class FakeGraphMutations:
    def upsert_edges(
        self,
        edges: Sequence[GraphEdge | tuple[str, str, str, float]],
    ) -> None:
        pass

    def delete_edges(
        self,
        edges: Sequence[GraphEdge | tuple[str, str, str, float]],
    ) -> None:
        pass


class FakeGraphStore:
    def __init__(
        self,
        neighbors: dict[str, list[GraphNeighbor | tuple[str, str, float]]],
    ) -> None:
        self.calls: list[str] = []
        self._neighbors = {
            key: [
                neighbor
                if isinstance(neighbor, GraphNeighbor)
                else GraphNeighbor(
                    RecordIdentity(None, "fake", neighbor[0]), neighbor[1], neighbor[2]
                )
                for neighbor in values
            ]
            for key, values in neighbors.items()
        }

    def upsert_edges(
        self,
        edges: Sequence[GraphEdge | tuple[str, str, str, float]],
    ) -> None:
        pass

    def delete_edges(
        self,
        edges: Sequence[GraphEdge | tuple[str, str, str, float]],
    ) -> None:
        pass

    def neighbors(
        self,
        record_id: RecordIdentity | str,
        edge_types: list[str] | None = None,
        depth: int = 1,
        max_neighbors: int | None = None,
    ) -> Sequence[GraphNeighbor]:
        del edge_types, depth, max_neighbors
        key = (
            record_id.source_id
            if isinstance(record_id, RecordIdentity)
            else record_id
        )
        self.calls.append(key)
        return self._neighbors.get(key, [])


class FakeEmbedder:
    model_name = "fake-model"
    dim = 2

    def embed_query(self, query: str) -> list[float]:
        assert query == "query"
        return [1.0, 0.0]


def _hydrator(records: dict[str, Record]):
    def hydrate(record_id: RecordIdentity) -> Record | None:
        return records.get(record_id.source_id)

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
    assert outcome.diagnostic_evidence is not None
    assert outcome.diagnostic_evidence.enabled_lanes == ("keyword",)
    assert outcome.diagnostic_evidence.lane_budgets["keyword"] == 10
    assert outcome.diagnostic_evidence.result_provenance == {
        outcome.results[0].storage_key: ("keyword",),
        outcome.results[1].storage_key: ("keyword",),
    }
    assert outcome.diagnostic_evidence.raw_pre_fusion_overlap.available
    assert outcome.diagnostic_evidence.raw_pre_fusion_overlap.count == 0


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


@pytest.mark.parametrize("limit", [1, 3, 5])
async def test_search_never_returns_more_than_requested_limit(limit: int) -> None:
    record_ids = [f"record-{index}" for index in range(10)]
    records = {record_id: _record(record_id) for record_id in record_ids}
    pipeline = RecordSearchPipeline(
        keyword_store=FakeKeywordStore([(record_id, 1.0) for record_id in record_ids]),
        hydrator=_hydrator(records),
        config=RecordSearchConfig(
            adaptive_enabled=True,
            maximum_limit=10,
            score_ratio_floor=0.0,
            minimum_score=0.0,
            maximum_score_gap=1.0,
        ),
    )

    outcome = await pipeline.async_search("query", limit=limit)

    assert len(outcome.results) == limit


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
    assert outcome.diagnostic_evidence is not None
    assert outcome.diagnostic_evidence.enabled_lanes == ("keyword", "vector")
    assert outcome.diagnostic_evidence.result_provenance[
        outcome.results[0].storage_key
    ] == (
        "keyword",
        "vector",
    )
    assert outcome.diagnostic_evidence.raw_pre_fusion_overlap.available
    assert outcome.diagnostic_evidence.raw_pre_fusion_overlap.count == 1
    assert outcome.diagnostic_evidence.final_duplicate_count == 0


async def test_exact_identifier_outranks_nearby_keyword_match() -> None:
    records = {
        "ENG-939": _record("ENG-939"),
        "ENG-940": _record("ENG-940"),
    }
    pipeline = RecordSearchPipeline(
        keyword_store=FakeKeywordStore([("ENG-940", 10.0), ("ENG-939", 1.0)]),
        hydrator=_hydrator(records),
    )

    outcome = await pipeline.async_search("eng-939", limit=2)

    assert [result.record_id for result in outcome.results] == ["ENG-939", "ENG-940"]
    assert outcome.results[0].provenance.strategies == (
        "keyword",
        "exact_identifier",
    )


@pytest.mark.parametrize(
    "query",
    (
        "/repo/src/search.py",
        'record:[null,"fake","/repo/src/search.py"]',
        "fake:/repo/src/search.py",
    ),
)
async def test_exact_canonical_identifier_variants_outrank_nearby_match(
    query: str,
) -> None:
    exact = Record(
        source_kind="fake",
        source_id="/repo/src/search.py",
        title="search.py",
        body="search implementation",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    nearby = _record("search.py")
    records = {exact.source_id: exact, nearby.source_id: nearby}
    pipeline = RecordSearchPipeline(
        keyword_store=FakeKeywordStore(
            [(nearby.source_id, 10.0), (exact.source_id, 1.0)]
        ),
        hydrator=_hydrator(records),
    )

    outcome = await pipeline.async_search(query, limit=2)

    assert [result.record_id for result in outcome.results] == [
        exact.source_id,
        nearby.source_id,
    ]


async def test_exact_identifier_survives_reranker_reordering() -> None:
    class ReverseReranker:
        model_name = "fake-reranker"

        def rerank(self, query: str, documents: list[str]) -> list[float]:
            return [0.1, 0.9]

    exact = _record("ENG-939")
    nearby = _record("ENG-940")
    pipeline = RecordSearchPipeline(
        keyword_store=FakeKeywordStore([("ENG-939", 1.0), ("ENG-940", 10.0)]),
        hydrator=_hydrator({exact.source_id: exact, nearby.source_id: nearby}),
        reranker=ReverseReranker(),
        config=RecordSearchConfig(rerank_budget=2),
    )

    outcome = await pipeline.async_search("eng-939", limit=2)

    assert [result.record_id for result in outcome.results] == [
        exact.source_id,
        nearby.source_id,
    ]


@pytest.mark.parametrize("retrieval_mode", ["semantic", "semantic_only"])
async def test_semantic_retrieval_mode_routes_only_vector(
    retrieval_mode: str,
) -> None:
    records = {record_id: _record(record_id) for record_id in ("vector", "graph")}
    keyword_store = FakeKeywordStore([("keyword", 1.0)])
    graph_store = FakeGraphStore({"vector": [("graph", "related", 1.0)]})

    class AnyQueryEmbedder(FakeEmbedder):
        def embed_query(self, query: str) -> list[float]:
            return [1.0, 0.0]

    pipeline = RecordSearchPipeline(
        keyword_store=keyword_store,
        vector_store=FakeVectorStore([("vector", 0.9)]),
        graph_store=graph_store,
        embedding_provider=AnyQueryEmbedder(),
        hydrator=_hydrator(records),
        config=RecordSearchConfig(capture_trace=True),
    )

    outcome = await pipeline.async_search(
        "what relates to query?",
        limit=1,
        filters={"retrieval_mode": retrieval_mode},
    )

    assert [result.record_id for result in outcome.results] == ["vector"]
    assert keyword_store.queries == []
    assert graph_store.calls == []
    assert outcome.trace is not None
    trace = outcome.trace.to_dict()
    provenance = _mapping(trace["provenance"])
    query_plan = _mapping(provenance["query_plan"])
    assert query_plan["lanes"] == (
        "vector",
    )
    assert outcome.diagnostic_evidence is not None
    assert outcome.diagnostic_evidence.enabled_lanes == ("vector",)
    assert outcome.diagnostic_evidence.raw_pre_fusion_overlap.available
    assert outcome.diagnostic_evidence.raw_pre_fusion_overlap.count == 0
    assert {
        (skip.lane, skip.reason)
        for skip in outcome.diagnostic_evidence.skipped_lanes
    } == {
        ("keyword", "unavailable"),
        ("graph", "unavailable"),
    }


async def test_retrieval_mode_defaults_to_hybrid_and_rejects_unknown_values() -> None:
    pipeline = RecordSearchPipeline(
        keyword_store=FakeKeywordStore([("a", 1.0)]),
        hydrator=_hydrator({"a": _record("a")}),
        config=RecordSearchConfig(capture_trace=True),
    )

    outcome = await pipeline.async_search("query", limit=1)
    assert outcome.trace is not None
    trace = outcome.trace.to_dict()
    provenance = _mapping(trace["provenance"])
    query_plan = _mapping(provenance["query_plan"])
    assert query_plan["lanes"] == (
        "keyword",
    )

    with pytest.raises(ValueError, match="retrieval_mode"):
        await pipeline.async_search(
            "query", limit=1, filters={"retrieval_mode": "keyword"}
        )


async def test_vector_candidate_acquisition_supports_async_store_adapter() -> None:
    class AsyncVectorStore:
        def epoch(self) -> int:
            return 0

        async def search(
            self,
            query_vector: list[float],
            k: int,
            *,
            model_name: str,
            dim: int,
            filters: SearchFilters | None = None,
        ) -> Sequence[RecordHit]:
            assert query_vector == [1.0, 0.0]
            assert (model_name, dim) == ("fake-model", 2)
            assert filters == {"statuses": ["active"]}
            return _hits([("a", 0.9)])

        async_search = search

    pipeline = RecordSearchPipeline(
        vector_store=AsyncVectorStore(),
        embedding_provider=FakeEmbedder(),
        hydrator=_hydrator({"a": _record("a")}),
    )

    outcome = await pipeline.async_search("query")

    assert [result.record_id for result in outcome.results] == ["a"]


async def test_policy_can_bound_vector_acquisition_to_keyword_candidates() -> None:
    records = {record_id: _record(record_id) for record_id in ("a", "b")}
    vector_store = FakeVectorStore([("a", 0.9)])
    pipeline = RecordSearchPipeline(
        keyword_store=FakeKeywordStore([("b", 1.0), ("a", 0.5)]),
        vector_store=vector_store,
        embedding_provider=FakeEmbedder(),
        hydrator=_hydrator(records),
        policy=RecordSearchPolicy(
            vector_candidate_ids=lambda ranking, filters: [
                hit.storage_key for hit in ranking
            ]
        ),
    )

    await pipeline.async_search("query", limit=2, filters={"workspace_id": "workspace-1"})

    assert vector_store.filters == [
        {
            "workspace_id": "workspace-1",
            "statuses": ["active"],
            "candidate_ids": [
                RecordIdentity(None, "fake", "b").storage_key,
                RecordIdentity(None, "fake", "a").storage_key,
            ],
        }
    ]


async def test_vector_policy_receives_typed_query_context() -> None:
    records = {record_id: _record(record_id) for record_id in ("a", "b")}
    vector_store = FakeVectorStore([("a", 0.9), ("b", 0.8)])
    contexts: list[RecordSearchQueryContext] = []

    def select_candidates(
        ranking: Sequence[RecordHit],
        context: RecordSearchQueryContext,
    ) -> list[str]:
        contexts.append(context)
        assert context.query == "query"
        assert context.limit == 2
        assert context["workspace_id"] == "workspace-1"
        assert context["statuses"] == ["active"]
        return [hit.storage_key for hit in ranking]

    def order_candidates(
        ranking: Sequence[RecordHit],
        context: RecordSearchQueryContext,
    ) -> Sequence[RecordHit]:
        contexts.append(context)
        assert dict(context) == dict(context.filters)
        return list(reversed(ranking))

    pipeline = RecordSearchPipeline(
        vector_store=vector_store,
        embedding_provider=FakeEmbedder(),
        hydrator=_hydrator(records),
        policy=RecordSearchPolicy(
            vector_candidate_ids=select_candidates,
            vector_ranking_order=order_candidates,
        ),
    )

    outcome = await pipeline.async_search(
        "query", limit=2, filters={"workspace_id": "workspace-1"}
    )

    assert [result.record_id for result in outcome.results] == ["b", "a"]
    assert len(contexts) == 2
    assert contexts[0] is contexts[1]


async def test_missing_epoch_bypasses_candidate_cache() -> None:
    records = {"a": _record("a")}
    pipeline = RecordSearchPipeline(
        keyword_store=FakeKeywordStore([("a", 1.0)]),
        hydrator=_hydrator(records),
    )

    outcome = await pipeline.async_search("query")

    assert "candidate_cache:bypass:UnstableCacheKey" in outcome.cache_diagnostics


async def test_candidate_cache_hit_skips_source_acquisition() -> None:
    class EpochKeywordStore(FakeKeywordStore):
        def keyword_epoch(self) -> int:
            return 0

    store = EpochKeywordStore([("a", 1.0)])
    pipeline = RecordSearchPipeline(
        keyword_store=store,
        hydrator=_hydrator({"a": _record("a")}),
    )

    first = await pipeline.async_search("query")
    second = await pipeline.async_search("query")

    assert "candidate_cache:miss" in first.cache_diagnostics
    assert "candidate_cache:hit" in second.cache_diagnostics
    assert len(store.queries) == 1


async def test_candidate_cache_fingerprint_includes_routing_weights() -> None:
    class EpochKeywordStore(FakeKeywordStore):
        def keyword_epoch(self) -> int:
            return 0

    store = EpochKeywordStore([("a", 1.0)])
    cache = CandidateResultCache()
    first = RecordSearchPipeline(
        keyword_store=store,
        hydrator=_hydrator({"a": _record("a")}),
        candidate_cache=cache,
    )
    second = RecordSearchPipeline(
        keyword_store=store,
        hydrator=_hydrator({"a": _record("a")}),
        candidate_cache=cache,
        config=RecordSearchConfig(base_keyword_weight=2.0),
    )

    first_outcome = await first.async_search("query")
    second_outcome = await second.async_search("query")

    assert "candidate_cache:miss" in first_outcome.cache_diagnostics
    assert "candidate_cache:miss" in second_outcome.cache_diagnostics


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

    outcome = await pipeline.async_search("what relates to seed?", limit=3)

    assert [result.record_id for result in outcome.results] == ["seed", "allowed"]
    assert "blocked" not in {result.record_id for result in outcome.results}
    assert "graph" in outcome.results[-1].provenance.strategies


async def test_relationship_query_prioritizes_linked_records_over_unrelated_direct_hits() -> None:
    records = {
        record_id: _record(record_id)
        for record_id in ("seed", "linked", "direct")
    }
    pipeline = RecordSearchPipeline(
        keyword_store=FakeKeywordStore([("seed", 1.0), ("direct", 0.9)]),
        graph_store=FakeGraphStore({"seed": [("linked", "related", 1.0)]}),
        hydrator=_hydrator(records),
    )

    outcome = await pipeline.async_search("what relates to the seed?", limit=3)

    assert [result.record_id for result in outcome.results] == [
        "seed",
        "linked",
        "direct",
    ]
    assert outcome.results[1].provenance.strategies == ("graph",)


async def test_parent_expansion_preserves_canonical_identity_and_first_rank() -> None:
    records = {
        record_id: _record(record_id)
        for record_id in ("child-a", "child-b", "parent")
    }
    child_a = RecordIdentity(None, "fake", "child-a")
    child_b = RecordIdentity(None, "fake", "child-b")
    parent = RecordIdentity(None, "fake", "parent")

    class ParentExpander:
        def parent_identity(self, identity: RecordIdentity) -> RecordIdentity | None:
            if identity.source_id in {"child-a", "child-b"}:
                return parent
            return None

    pipeline = RecordSearchPipeline(
        keyword_store=FakeKeywordStore(
            [
                RecordHit(child_b, 1.0),
                RecordHit(child_a, 0.9),
            ]
        ),
        hydrator=_hydrator(records),
        policy=RecordSearchPolicy(parent_expander=ParentExpander()),
    )

    outcome = await pipeline.async_search("query", limit=2)

    assert [result.record_id for result in outcome.results] == ["parent"]
    provenance = outcome.results[0].provenance
    assert provenance.record_identity == parent
    assert provenance.parent_expanded_from == "child-b"
    assert provenance.parent_expanded_from_identity == child_b


async def test_parent_expansion_prefers_batch_resolution() -> None:
    records = {record_id: _record(record_id) for record_id in ("child", "parent")}
    child = RecordIdentity(None, "fake", "child")
    parent = RecordIdentity(None, "fake", "parent")

    class ParentExpander:
        def __init__(self) -> None:
            self.calls = 0

        async def parent_identities(
            self, identities: Sequence[RecordIdentity]
        ) -> dict[str, RecordIdentity | None]:
            self.calls += 1
            assert identities == [child]
            return {child.storage_key: parent}

        def parent_identity(self, identity: RecordIdentity) -> RecordIdentity | None:
            raise AssertionError("scalar parent lookup should not run")

    expander = ParentExpander()
    pipeline = RecordSearchPipeline(
        keyword_store=FakeKeywordStore([RecordHit(child, 1.0)]),
        hydrator=_hydrator(records),
        policy=RecordSearchPolicy(parent_expander=expander),
    )

    outcome = await pipeline.async_search("query", limit=1)

    assert [result.record_id for result in outcome.results] == ["parent"]
    assert expander.calls == 1


async def test_missing_chunk_parent_is_reported_without_reordering_results() -> None:
    """Report missing chunk parents while preserving lenient result ordering.

    A hydrated chunk cannot become a result when its parent is unavailable,
    but the search outcome must retain the existing missing-record contract.
    """
    parent = RecordIdentity(None, "fake", "missing-parent")
    chunk = Record(
        source_kind="fake",
        source_id=f"{parent.storage_key}#chunk:0",
        title="chunk",
        body="chunk body",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
        metadata={
            "_searchkernel_chunk": True,
            "_chunk_id": "0",
            "_chunk_parent_storage_key": parent.storage_key,
            "_chunk_metadata": {},
        },
    )
    ordinary = _record("ordinary")
    pipeline = RecordSearchPipeline(
        keyword_store=FakeKeywordStore(
            [RecordHit(chunk.identity, 1.0), RecordHit(ordinary.identity, 0.9)]
        ),
        hydrator=_hydrator({chunk.source_id: chunk, ordinary.source_id: ordinary}),
        config=RecordSearchConfig(failure_mode="lenient"),
    )

    outcome = await pipeline.async_search("query", limit=2)

    assert [result.record_id for result in outcome.results] == ["ordinary"]
    assert outcome.missing_record_ids == ("missing-parent",)
    assert outcome.diagnostic_evidence is not None
    assert outcome.diagnostic_evidence.missing_record_ids == ("missing-parent",)
    assert outcome.diagnostic_evidence.degraded


async def test_chunk_aggregation_combines_matches_and_truncates_excerpts() -> None:
    """Combine chunks for one parent while preserving best-score semantics.

    The aggregate keeps the highest parent score and returns only the
    configured number of deterministically ordered chunk matches.
    """
    parent = _record("parent")
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    chunks = [
        Record(
            source_kind="fake",
            source_id=f"{parent.storage_key}#chunk:{chunk_id}",
            title=f"chunk-{chunk_id}",
            body=f"body-{chunk_id}",
            created_at=timestamp,
            updated_at=timestamp,
            metadata={
                "_searchkernel_chunk": True,
                "_chunk_id": chunk_id,
                "_chunk_parent_storage_key": parent.storage_key,
                "_chunk_metadata": {"start_pos": start_pos},
            },
        )
        for chunk_id, start_pos in (("a", 20), ("b", 10), ("c", 0))
    ]
    provenance = SearchResultProvenance(strategies=("vector",))

    class BatchHydrator:
        async def hydrate_records(
            self, identities: Sequence[RecordIdentity]
        ) -> dict[str, Record]:
            assert identities == [parent.identity]
            return {parent.storage_key: parent}

    pipeline = RecordSearchPipeline(
        hydrator=BatchHydrator(),
        config=RecordSearchConfig(max_chunk_matches=2),
    )
    results = [
        RecordSearchResult(
            record=chunk,
            score=score,
            provenance=provenance,
        )
        for chunk, score in zip(chunks, (0.9, 0.8, 0.7), strict=True)
    ]

    aggregated = await pipeline._aggregate_chunk_results(
        results, [], [], limit=1
    )

    assert len(aggregated) == 1
    assert aggregated[0].record_id == "parent"
    assert aggregated[0].score == 0.9
    assert [match.chunk_id for match in aggregated[0].chunk_matches] == [
        "a",
        "b",
    ]
    assert aggregated[0].provenance is not provenance
    assert aggregated[0].provenance.strategies == ("vector",)


def _chunk_result(
    parent: Record,
    chunk_id: str,
    score: float,
    start_pos: int,
    strategies: tuple[str, ...] = ("vector",),
) -> RecordSearchResult:
    """Build a hydrated chunk result for direct aggregation tests."""
    chunk = Record(
        source_kind=parent.source_kind,
        source_id=f"{parent.storage_key}#chunk:{chunk_id}",
        title=f"chunk {chunk_id}",
        body=f"chunk body {chunk_id}",
        created_at=parent.created_at,
        updated_at=parent.updated_at,
        metadata={
            "_searchkernel_chunk": True,
            "_chunk_id": chunk_id,
            "_chunk_parent_storage_key": parent.storage_key,
            "_chunk_metadata": {"start_pos": start_pos},
        },
    )
    return RecordSearchResult(
        record=chunk,
        score=score,
        provenance=SearchResultProvenance(
            strategies=strategies, record_identity=chunk.identity
        ),
    )


async def test_chunk_aggregation_indexes_existing_and_synthesized_parents() -> None:
    """Aggregate several parents while preserving matches and provenance.

    Existing parents retain ordinary-result provenance, synthesized parents
    are added after ordinary results, and both match and result limits apply.
    """
    parent_a = _record("parent-a")
    parent_b = _record("parent-b")
    ordinary = RecordSearchResult(
        record=parent_a,
        score=0.4,
        provenance=SearchResultProvenance(
            strategies=("keyword",), record_identity=parent_a.identity
        ),
    )
    chunks = [
        _chunk_result(parent_a, "a-2", 0.9, 2),
        _chunk_result(parent_a, "a-1", 0.8, 1),
        _chunk_result(parent_b, "b-1", 0.7, 1),
    ]
    pipeline = RecordSearchPipeline(
        hydrator=_hydrator({parent_b.source_id: parent_b}),
        config=RecordSearchConfig(max_chunk_matches=1),
    )

    aggregated = await pipeline._aggregate_chunk_results(
        [ordinary, *chunks], [], [], limit=3
    )

    assert [result.record_id for result in aggregated] == ["parent-a", "parent-b"]
    assert aggregated[0].score == 0.9
    assert aggregated[0].provenance.strategies == ("keyword",)
    assert aggregated[0].provenance.record_identity == parent_a.identity
    assert [match.chunk_id for match in aggregated[0].chunk_matches] == ["a-2"]
    assert aggregated[1].provenance.strategies == ("vector",)
    assert [match.chunk_id for match in aggregated[1].chunk_matches] == ["b-1"]
    assert [
        result.record_id
        for result in await pipeline._aggregate_chunk_results(
            [ordinary, *chunks], [], [], limit=1
        )
    ] == ["parent-a"]


async def test_synthesized_parent_uses_strongest_chunk_provenance() -> None:
    """Use the strongest chunk for synthesized score and provenance.

    Equal scores choose the lexicographically earliest chunk identity so the
    selected provenance is deterministic regardless of retrieval order.
    """
    parent = _record("parent")
    weaker = _chunk_result(
        parent,
        "z-weaker",
        0.7,
        1,
        ("keyword",),
    )
    strongest = _chunk_result(
        parent,
        "b-strongest",
        0.9,
        2,
        ("vector",),
    )
    tied = _chunk_result(
        parent,
        "a-tied",
        0.9,
        3,
        ("graph",),
    )
    pipeline = RecordSearchPipeline(hydrator=_hydrator({parent.source_id: parent}))

    aggregated = await pipeline._aggregate_chunk_results(
        [weaker, tied, strongest], [], [], limit=1
    )

    assert aggregated[0].score == 0.9
    assert aggregated[0].provenance.strategies == ("graph",)
    assert aggregated[0].provenance.record_identity == tied.record.identity


async def test_chunk_aggregation_batches_distinct_missing_parents() -> None:
    """Report each missing parent while using the existing batch capability."""
    parent_a = RecordIdentity(None, "fake", "missing-a")
    parent_b = RecordIdentity(None, "fake", "missing-b")
    chunks = [
        _chunk_result(_record(parent_a.source_id), "a", 0.9, 1),
        _chunk_result(_record(parent_b.source_id), "b", 0.8, 1),
    ]

    class BatchHydrator:
        async def hydrate_records(
            self, identities: Sequence[RecordIdentity]
        ) -> dict[str, Record | None]:
            assert identities == [parent_a, parent_b]
            return {}

        def hydrate_record(self, identity: RecordIdentity) -> Record | None:
            raise AssertionError("scalar hydration should not run")

    pipeline = RecordSearchPipeline(hydrator=BatchHydrator())
    missing: list[str] = []

    aggregated = await pipeline._aggregate_chunk_results(
        chunks, [], missing, limit=2
    )

    assert aggregated == []
    assert missing == ["missing-a", "missing-b"]


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


async def test_query_policy_calibrates_sources_without_hardcoded_query_modes() -> None:
    document = _source_record("document", "doc")
    commit = _source_record("git_commit", "commit")
    store = FakeKeywordStore([])
    store.results = [
        RecordHit(commit.identity, 1.0),
        RecordHit(document.identity, 0.9),
    ]

    def allowed_source(
        candidate: RecordSearchCandidate,
        context: RecordSearchQueryContext,
    ) -> bool:
        source_kind = candidate.source_kind
        selected = context.filters.get("source_kinds")
        if isinstance(selected, Sequence) and not isinstance(selected, str):
            return source_kind in selected
        return source_kind != "git_commit"

    def calibrate_source(
        candidate: RecordSearchCandidate,
        context: RecordSearchQueryContext,
    ) -> float:
        if context.filters.get("source_kinds"):
            return candidate.score
        return candidate.score * (2.0 if candidate.source_kind == "document" else 0.1)

    pipeline = RecordSearchPipeline(
        keyword_store=store,
        hydrator=_hydrator(
            {"doc": document, "commit": commit},
        ),
        policy=RecordSearchPolicy(
            query_candidate_filter=allowed_source,
            query_score_adjuster=calibrate_source,
        ),
    )

    ordinary = await pipeline.async_search("what is retrieval?", limit=2)
    explicit_git = await pipeline.async_search(
        "show history",
        limit=2,
        filters={"source_kinds": ["git_commit"]},
    )

    assert [result.record_id for result in ordinary.results] == ["doc"]
    assert [result.record_id for result in explicit_git.results] == ["commit"]


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

    outcome = await pipeline.async_search("what relates to the seed?", limit=3)

    assert [result.record_id for result in outcome.results] == ["seed"]
    assert outcome.missing_record_ids == ("missing",)
    assert outcome.degraded


async def test_graph_expansion_reads_only_bounded_seed_neighbors() -> None:
    records = {record_id: _record(record_id) for record_id in ("a", "b", "c")}
    calls: list[RecordIdentity] = []

    class Graph(FakeGraphMutations):
        def upsert_edges(
            self,
            edges: Sequence[GraphEdge | tuple[str, str, str, float]],
        ) -> None:
            pass

        def delete_edges(
            self,
            edges: Sequence[GraphEdge | tuple[str, str, str, float]],
        ) -> None:
            pass

        async def neighbors(
            self,
            record_id: RecordIdentity | str,
            edge_types: list[str] | None = None,
            depth: int = 1,
            max_neighbors: int | None = None,
        ) -> Sequence[GraphNeighbor]:
            assert isinstance(record_id, RecordIdentity)
            assert max_neighbors == 1
            calls.append(record_id)
            return [
                GraphNeighbor(
                    RecordIdentity(None, "fake", "missing"),
                    "related",
                    1.0,
                )
            ]

    pipeline = RecordSearchPipeline(
        keyword_store=FakeKeywordStore([("a", 1.0), ("b", 0.9), ("c", 0.8)]),
        graph_store=Graph(),
        hydrator=_hydrator(records),
        config=RecordSearchConfig(
            max_graph_seeds=2,
            max_neighbors_per_seed=1,
            max_graph_concurrency=2,
        ),
        continue_on_error=True,
    )

    await pipeline.async_search("what relates to this module?", limit=3)

    assert sorted(identity.source_id for identity in calls) == ["a", "b"]


async def test_graph_expansion_selects_deterministic_top_neighbors_from_fallback_store() -> None:
    records = {
        record_id: _record(record_id)
        for record_id in ("seed", "target-a", "target-b", "target-c")
    }
    pipeline = RecordSearchPipeline(
        keyword_store=FakeKeywordStore([("seed", 1.0)]),
        graph_store=FakeGraphStore(
            {
                "seed": [
                    ("target-c", "related", 0.5),
                    ("target-b", "related", 0.9),
                    ("target-a", "related", 0.9),
                ]
            }
        ),
        hydrator=_hydrator(records),
        config=RecordSearchConfig(max_neighbors_per_seed=2),
    )

    outcome = await pipeline.async_search("what relates to the seed?", limit=4)

    assert [result.record_id for result in outcome.results] == [
        "seed",
        "target-a",
        "target-b",
    ]


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


async def test_async_callable_hydrator_runs_without_thread_hop() -> None:
    event_loop_thread = threading.get_ident()

    class AsyncHydrator:
        async def __call__(self, record_id: RecordIdentity) -> Record:
            assert threading.get_ident() == event_loop_thread
            return _record(record_id.source_id)

    pipeline = RecordSearchPipeline(
        keyword_store=FakeKeywordStore([("a", 1.0)]),
        hydrator=AsyncHydrator(),
    )

    outcome = await pipeline.async_search("query")

    assert [result.record_id for result in outcome.results] == ["a"]


async def test_store_errors_raise_by_default() -> None:
    class BrokenKeywordStore(FakeKeywordStore):
        def search(
            self,
            query: str,
            k: int,
            filters: SearchFilters | None = None,
        ) -> Sequence[RecordHit]:
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
            filters: SearchFilters | None = None,
        ) -> Sequence[RecordHit]:
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


async def test_malformed_candidate_data_keeps_other_lane_results_in_lenient_mode() -> None:
    """
    Malformed generic candidates fail one lane without discarding valid hits.
    """

    class MalformedKeywordStore:
        def search(
            self,
            query: str,
            k: int,
            filters: SearchFilters | None = None,
        ) -> list[object]:
            return [object()]

    pipeline = RecordSearchPipeline(
        keyword_store=cast(KeywordStore, MalformedKeywordStore()),
        vector_store=cast(
            VectorStore,
            FakeVectorStore([("vector-result", 0.9)]),
        ),
        embedding_provider=FakeEmbedder(),
        hydrator=_hydrator({"vector-result": _record("vector-result")}),
        config=RecordSearchConfig(failure_mode="lenient"),
    )

    outcome = await pipeline.async_search("query", limit=1)

    assert [result.record_id for result in outcome.results] == ["vector-result"]
    assert outcome.failures[0].stage == "keyword"
    assert outcome.failures[0].detail is not None


async def test_unavailable_provider_failure_detail_is_bounded() -> None:
    """
    Provider failures retain partial results and bounded neutral detail.
    """

    class UnavailableVectorStore(FakeVectorStore):
        def search(
            self,
            query_vector: list[float],
            k: int,
            *,
            model_name: str,
            dim: int,
            filters: SearchFilters | None = None,
        ) -> list[RecordHit]:
            raise RuntimeError("provider unavailable: " + "x" * 1000)

    pipeline = RecordSearchPipeline(
        keyword_store=FakeKeywordStore([("keyword-result", 1.0)]),
        vector_store=cast(VectorStore, UnavailableVectorStore([])),
        embedding_provider=FakeEmbedder(),
        hydrator=_hydrator({"keyword-result": _record("keyword-result")}),
        config=RecordSearchConfig(failure_mode="lenient"),
    )

    outcome = await pipeline.async_search("query", limit=1)

    assert [result.record_id for result in outcome.results] == ["keyword-result"]
    assert outcome.failures[0].stage == "vector"
    assert outcome.failures[0].detail is not None
    assert len(outcome.failures[0].detail) == MAX_FAILURE_DETAIL_LENGTH
    assert outcome.failures[0].detail.startswith("provider unavailable: ")


async def test_malformed_candidate_data_preserves_strict_failure_behavior() -> None:
    """
    Strict mode still raises the stage-specific retrieval exception.
    """

    class MalformedKeywordStore:
        def search(
            self,
            query: str,
            k: int,
            filters: SearchFilters | None = None,
        ) -> list[object]:
            return [object()]

    pipeline = RecordSearchPipeline(
        keyword_store=cast(KeywordStore, MalformedKeywordStore()),
        hydrator=_hydrator({}),
    )

    with pytest.raises(RecordSearchError, match="keyword retrieval failed"):
        await pipeline.async_search("query")


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

    class Store(FakeKeywordIndex):
        def index(self, records: list[Record]) -> None:
            pass

        def search(
            self,
            query: str,
            k: int,
            filters: SearchFilters | None = None,
        ) -> Sequence[RecordHit]:
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

    class Store(FakeKeywordIndex):
        def index(self, records: list[Record]) -> None:
            pass

        def search(
            self,
            query: str,
            k: int,
            filters: SearchFilters | None = None,
        ) -> Sequence[RecordHit]:
            return [RecordHit(seed, 1.0)]

    class Graph(FakeGraphMutations):
        def upsert_edges(
            self,
            edges: Sequence[GraphEdge | tuple[str, str, str, float]],
        ) -> None:
            pass

        def delete_edges(
            self,
            edges: Sequence[GraphEdge | tuple[str, str, str, float]],
        ) -> None:
            pass

        def neighbors(
            self,
            record_id: RecordIdentity | str,
            edge_types: list[str] | None = None,
            depth: int = 1,
            max_neighbors: int | None = None,
        ) -> Sequence[GraphNeighbor]:
            return [GraphNeighbor(target, "related", 1.0)]

    pipeline = RecordSearchPipeline(
        keyword_store=Store(),
        graph_store=Graph(),
        hydrator=_hydrator(records),
    )

    outcome = await pipeline.async_search("what relates to the seed?", limit=2)

    assert [result.storage_key for result in outcome.results] == [
        seed.storage_key,
        target.storage_key,
    ]


@pytest.mark.parametrize(
    ("depth", "expected"),
    [(1, ["one-hop"]), (2, ["one-hop", "two-hop"])],
)
async def test_graph_expansion_preserves_scoped_neighbor_provenance(
    depth: int,
    expected: list[str],
) -> None:
    seed = RecordIdentity("workspace-a", "note", "seed")
    one_hop = RecordIdentity("workspace-a", "note", "one-hop")
    two_hop = RecordIdentity("workspace-a", "note", "two-hop")
    cross_project = RecordIdentity("workspace-b", "git_commit", "noise")
    records = {
        identity.storage_key: Record(
            workspace_id=identity.workspace_id,
            source_kind=identity.source_kind,
            source_id=identity.source_id,
            title=identity.source_id,
            body=identity.source_id,
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            updated_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
        for identity in (seed, one_hop, two_hop, cross_project)
    }

    class Store(FakeKeywordIndex):
        def search(
            self,
            query: str,
            k: int,
            filters: SearchFilters | None = None,
        ) -> list[RecordHit]:
            return [RecordHit(seed, 1.0)]

    class Graph(FakeGraphMutations):
        def neighbors(
            self,
            record_id: RecordIdentity | str,
            edge_types: list[str] | None = None,
            depth: int = 1,
            max_neighbors: int | None = None,
            **kwargs: object,
        ) -> list[GraphNeighbor]:
            assert record_id == seed
            assert kwargs.get("filters") == {
                "statuses": ["active"],
                "workspace_id": "workspace-a",
                "source_kinds": ["note"],
            }
            neighbors = [GraphNeighbor(one_hop, "links_to", 1.0)]
            if depth > 1:
                neighbors.append(GraphNeighbor(two_hop, "references", 0.8))
            neighbors.append(GraphNeighbor(cross_project, "links_to", 2.0))
            return neighbors

    pipeline = RecordSearchPipeline(
        keyword_store=Store(),
        graph_store=Graph(),
        hydrator=lambda identity: records.get(identity.storage_key),
        config=RecordSearchConfig(graph_depth=depth),
    )

    outcome = await pipeline.async_search(
        "what is linked to the seed?",
        limit=3,
        filters={
            "workspace_id": "workspace-a",
            "source_kinds": ["note"],
        },
    )

    assert [result.record_id for result in outcome.results] == ["seed", *expected]
    assert all(
        result.record.workspace_id == "workspace-a" for result in outcome.results
    )
    assert all(
        result.record.source_kind == "note" for result in outcome.results
    )
    assert all(
        result.provenance.record_identity == result.record.identity
        for result in outcome.results
    )
    assert all(
        result.provenance.strategies == ("graph",)
        for result in outcome.results[1:]
    )


@pytest.mark.parametrize(
    "query",
    [
        "Which pages link to Hybrid Search Strategy?",
        "What documents are neighbors of Hybrid Search Strategy?",
        "show me notes that embed this one",
    ],
)
async def test_relationship_target_resolver_selects_canonical_neighbors(
    query: str,
) -> None:
    seed = RecordIdentity("workspace-a", "note", "explanation")
    target = RecordIdentity("workspace-a", "note", "hybrid-search-strategy")
    neighbors = (
        RecordIdentity("workspace-a", "note", "outbound"),
        RecordIdentity("workspace-a", "note", "inbound"),
    )
    outsider = RecordIdentity("workspace-b", "note", "other-project")
    identities = (seed, target, *neighbors, outsider)
    records = {
        identity.storage_key: Record(
            workspace_id=identity.workspace_id,
            source_kind=identity.source_kind,
            source_id=identity.source_id,
            title=identity.source_id,
            body=identity.source_id,
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            updated_at=datetime(2026, 1, 1, tzinfo=UTC),
            uri=f"/docs/{identity.source_id}.md",
        )
        for identity in identities
    }
    resolver_calls: list[tuple[str, Mapping[str, object]]] = []

    class Store(FakeKeywordIndex):
        def search(
            self,
            query: str,
            k: int,
            filters: SearchFilters | None = None,
        ) -> list[RecordHit]:
            return [RecordHit(seed, 1.0)]

    class Graph(FakeGraphMutations):
        def __init__(self) -> None:
            self.calls: list[RecordIdentity] = []

        def neighbors(
            self,
            record_id: RecordIdentity | str,
            edge_types: list[str] | None = None,
            depth: int = 1,
            max_neighbors: int | None = None,
            **kwargs: object,
        ) -> list[GraphNeighbor]:
            assert isinstance(record_id, RecordIdentity)
            self.calls.append(record_id)
            assert kwargs.get("filters") == {
                "statuses": ["active"],
                "workspace_id": "workspace-a",
                "project_id": "project-a",
            }
            return [
                GraphNeighbor(neighbors[0], "links_to", 1.0),
                GraphNeighbor(neighbors[1], "links_to", 1.0),
                GraphNeighbor(outsider, "links_to", 2.0),
            ]

    graph = Graph()

    async def resolve(
        query: str,
        context: RecordSearchQueryContext,
    ) -> list[RecordHit]:
        resolver_calls.append((query, context.filters))
        return [RecordHit(target, 2.0)]

    pipeline = RecordSearchPipeline(
        keyword_store=Store(),
        graph_store=graph,
        hydrator=lambda identity: records.get(identity.storage_key),
        policy=RecordSearchPolicy(graph_target_resolver=resolve),
        config=RecordSearchConfig(adaptive_graph_enabled=False),
    )

    outcome = await pipeline.async_search(
        query,
        limit=3,
        filters={"workspace_id": "workspace-a", "project_id": "project-a"},
    )

    assert resolver_calls == [
        (
            query,
            {
                "statuses": ["active"],
                "workspace_id": "workspace-a",
                "project_id": "project-a",
            },
        )
    ]
    assert graph.calls == [target]
    assert [result.record_id for result in outcome.results] == [
        "explanation",
        "inbound",
        "outbound",
    ]
    assert [result.record.uri for result in outcome.results] == [
        "/docs/explanation.md",
        "/docs/inbound.md",
        "/docs/outbound.md",
    ]
    assert [result.score for result in outcome.results] == pytest.approx(
        [1 / 61, 1 / 61, 1 / 62]
    )
    assert [result.provenance.strategies for result in outcome.results] == [
        ("keyword",),
        ("graph",),
        ("graph",),
    ]
    assert [
        result.provenance.strategy_details["graph"].rank
        for result in outcome.results[1:]
    ] == [1, 2]
    assert all(
        result.provenance.record_identity == result.record.identity
        for result in outcome.results
    )
    assert all(result.record.workspace_id == "workspace-a" for result in outcome.results)


async def test_relationship_target_resolver_preserves_no_neighbor_behavior() -> None:
    seed = _record("explanation")
    target = _record("target")
    calls: list[str] = []

    class Graph(FakeGraphMutations):
        def neighbors(
            self,
            record_id: RecordIdentity | str,
            edge_types: list[str] | None = None,
            depth: int = 1,
            max_neighbors: int | None = None,
        ) -> list[GraphNeighbor]:
            assert isinstance(record_id, RecordIdentity)
            calls.append(record_id.source_id)
            return []

    async def resolve(
        query: str,
        context: RecordSearchQueryContext,
    ) -> list[RecordHit]:
        return [RecordHit(target.identity, 2.0)]

    pipeline = RecordSearchPipeline(
        keyword_store=FakeKeywordStore([("explanation", 1.0)]),
        graph_store=Graph(),
        hydrator=_hydrator({"explanation": seed, "target": target}),
        policy=RecordSearchPolicy(graph_target_resolver=resolve),
        config=RecordSearchConfig(adaptive_graph_enabled=False),
    )

    outcome = await pipeline.async_search(
        "What documents are neighbors of target?",
        limit=2,
    )

    assert calls == ["target"]
    assert [result.record_id for result in outcome.results] == ["explanation"]
    assert outcome.results[0].provenance.strategies == ("keyword",)
    assert outcome.failures == ()
    assert outcome.missing_record_ids == ()


@pytest.mark.parametrize(
    "query",
    [
        "What documents are neighbors of Hybrid Search Strategy?",
        "What pages does Hybrid Search Strategy link to?",
    ],
)
async def test_relationship_resolver_keeps_direct_target_as_graph_seed(
    query: str,
) -> None:
    explanation = _source_record("note", "explanation")
    target = _source_record("note", "hybrid-search-strategy")
    neighbor = _source_record("note", "neighbor")
    outsider = _source_record("note", "outsider")
    for record in (explanation, target, neighbor):
        record.workspace_id = "project-a"
    outsider.workspace_id = "project-b"
    records = {
        record.source_id: record
        for record in (explanation, target, neighbor, outsider)
    }
    calls: list[RecordIdentity] = []

    class Store(FakeKeywordIndex):
        def search(
            self,
            query: str,
            k: int,
            filters: SearchFilters | None = None,
        ) -> list[RecordHit]:
            return [
                RecordHit(explanation.identity, 1.0),
                RecordHit(target.identity, 0.2),
            ]

    class Graph(FakeGraphMutations):
        def neighbors(
            self,
            record_id: RecordIdentity | str,
            edge_types: list[str] | None = None,
            depth: int = 1,
            max_neighbors: int | None = None,
            **kwargs: object,
        ) -> list[GraphNeighbor]:
            assert isinstance(record_id, RecordIdentity)
            calls.append(record_id)
            assert kwargs.get("filters") == {
                "statuses": ["active"],
                "workspace_id": "project-a",
                "source_kinds": ["note"],
            }
            return [
                GraphNeighbor(neighbor.identity, "links_to", 0.9),
                GraphNeighbor(outsider.identity, "links_to", 2.0),
            ]

    async def resolve(
        query: str,
        context: RecordSearchQueryContext,
    ) -> list[RecordHit]:
        return [RecordHit(target.identity, 2.0)]

    pipeline = RecordSearchPipeline(
        keyword_store=Store(),
        graph_store=Graph(),
        hydrator=_hydrator(records),
        policy=RecordSearchPolicy(graph_target_resolver=resolve),
        config=RecordSearchConfig(
            adaptive_graph_enabled=False,
            max_graph_seeds=1,
        ),
    )

    outcome = await pipeline.async_search(
        query,
        limit=3,
        filters={"workspace_id": "project-a", "source_kinds": ["note"]},
    )

    assert calls == [target.identity]
    assert [result.record_id for result in outcome.results] == [
        "explanation",
        "neighbor",
        "hybrid-search-strategy",
    ]
    assert outcome.results[1].provenance.strategies == ("graph",)
    assert outcome.results[1].provenance.strategy_details["graph"].rank == 1
    assert outcome.results[1].score == pytest.approx(1 / 61)
    assert all(result.record.workspace_id == "project-a" for result in outcome.results)


async def test_relationship_target_resolver_uses_incoming_graph_direction() -> None:
    seed = _source_record("note", "explanation")
    target = _source_record("note", "hybrid-search-strategy")
    inbound = _source_record("note", "inbound")
    outsider = _source_record("note", "outsider")
    seed.workspace_id = "workspace-a"
    target.workspace_id = "workspace-a"
    inbound.workspace_id = "workspace-a"
    outsider.workspace_id = "workspace-b"
    records = {
        record.source_id: record for record in (seed, target, inbound, outsider)
    }
    calls: list[str] = []

    class Graph(FakeGraphMutations):
        def neighbors(
            self,
            record_id: RecordIdentity | str,
            edge_types: list[str] | None = None,
            depth: int = 1,
            max_neighbors: int | None = None,
        ) -> list[GraphNeighbor]:
            raise AssertionError("outgoing traversal should not run")

        def incoming_neighbors(
            self,
            record_id: RecordIdentity | str,
            edge_types: list[str] | None = None,
            depth: int = 1,
            max_neighbors: int | None = None,
            **kwargs: object,
        ) -> list[GraphNeighbor]:
            assert isinstance(record_id, RecordIdentity)
            calls.append(record_id.source_id)
            assert kwargs.get("filters") == {
                "statuses": ["active"],
                "workspace_id": "workspace-a",
                "source_kinds": ["note"],
            }
            return [
                GraphNeighbor(inbound.identity, "links_to", 0.8),
                GraphNeighbor(outsider.identity, "links_to", 2.0),
            ]

    class Store(FakeKeywordIndex):
        def search(
            self,
            query: str,
            k: int,
            filters: SearchFilters | None = None,
        ) -> list[RecordHit]:
            return [RecordHit(seed.identity, 1.0)]

    async def resolve(
        query: str,
        context: RecordSearchQueryContext,
    ) -> list[RecordHit]:
        return [RecordHit(target.identity, 2.0)]

    pipeline = RecordSearchPipeline(
        keyword_store=Store(),
        graph_store=Graph(),
        hydrator=_hydrator(records),
        policy=RecordSearchPolicy(graph_target_resolver=resolve),
        config=RecordSearchConfig(adaptive_graph_enabled=False),
    )

    outcome = await pipeline.async_search(
        "Which pages link to Hybrid Search Strategy?",
        limit=3,
        filters={"workspace_id": "workspace-a", "source_kinds": ["note"]},
    )

    assert calls == ["hybrid-search-strategy"]
    assert [result.record_id for result in outcome.results] == [
        "explanation",
        "inbound",
    ]
    assert outcome.results[1].provenance.strategies == ("graph",)


async def test_chunk_target_hits_normalize_to_document_graph_neighbors() -> None:
    seed = RecordIdentity("project-a", "note", "explanation")
    target = RecordIdentity("project-a", "note", "hybrid-search-strategy")
    target_chunk_a = RecordIdentity(
        "project-a",
        "note",
        "hybrid-search-strategy_chunk_26",
    )
    target_chunk_b = RecordIdentity(
        "project-a",
        "note",
        "hybrid-search-strategy_chunk_27",
    )
    neighbors = (
        RecordIdentity("project-a", "note", "neighbor-a"),
        RecordIdentity("project-a", "note", "neighbor-b"),
    )
    outsider = RecordIdentity("project-b", "note", "other-project")

    def record(
        identity: RecordIdentity,
        *,
        doc_id: str | None = None,
    ) -> Record:
        return Record(
            workspace_id=identity.workspace_id,
            source_kind=identity.source_kind,
            source_id=identity.source_id,
            title=identity.source_id,
            body=identity.source_id,
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            updated_at=datetime(2026, 1, 1, tzinfo=UTC),
            metadata={"doc_id": doc_id} if doc_id is not None else {},
            uri=f"/docs/{identity.source_id}.md",
        )

    records = {
        identity.storage_key: record(identity)
        for identity in (seed, *neighbors, outsider)
    }
    records[target_chunk_a.storage_key] = record(
        target_chunk_a,
        doc_id=target.source_id,
    )
    records[target_chunk_b.storage_key] = record(
        target_chunk_b,
        doc_id=target.source_id,
    )
    graph_calls: list[RecordIdentity] = []

    class Store(FakeKeywordIndex):
        def search(
            self,
            query: str,
            k: int,
            filters: SearchFilters | None = None,
        ) -> list[RecordHit]:
            return [RecordHit(seed, 1.0)]

    class Graph(FakeGraphMutations):
        def neighbors(
            self,
            record_id: RecordIdentity | str,
            edge_types: list[str] | None = None,
            depth: int = 1,
            max_neighbors: int | None = None,
            **kwargs: object,
        ) -> list[GraphNeighbor]:
            assert isinstance(record_id, RecordIdentity)
            assert isinstance(record_id, RecordIdentity)
            graph_calls.append(record_id)
            assert kwargs.get("filters") == {
                "statuses": ["active"],
                "workspace_id": "project-a",
                "project_id": "project-a",
            }
            return [
                GraphNeighbor(neighbors[1], "links_to", 1.0),
                GraphNeighbor(outsider, "links_to", 2.0),
                GraphNeighbor(neighbors[0], "links_to", 1.0),
            ]

    async def resolve(
        query: str,
        context: RecordSearchQueryContext,
    ) -> list[RecordHit]:
        return [
            RecordHit(target_chunk_b, 1.5),
            RecordHit(target_chunk_a, 2.0),
        ]

    pipeline = RecordSearchPipeline(
        keyword_store=Store(),
        graph_store=Graph(),
        hydrator=lambda identity: records.get(identity.storage_key),
        policy=RecordSearchPolicy(graph_target_resolver=resolve),
        config=RecordSearchConfig(adaptive_graph_enabled=False),
    )

    outcome = await pipeline.async_search(
        "What pages does Hybrid Search Strategy link to?",
        limit=3,
        filters={"workspace_id": "project-a", "project_id": "project-a"},
    )

    assert graph_calls == [target]
    assert [result.record_id for result in outcome.results] == [
        "explanation",
        "neighbor-a",
        "neighbor-b",
    ]
    assert [result.record.uri for result in outcome.results] == [
        "/docs/explanation.md",
        "/docs/neighbor-a.md",
        "/docs/neighbor-b.md",
    ]
    assert [result.score for result in outcome.results] == pytest.approx(
        [1 / 61, 1 / 61, 1 / 62]
    )
    assert [result.provenance.strategies for result in outcome.results] == [
        ("keyword",),
        ("graph",),
        ("graph",),
    ]
    assert all(
        result.record.workspace_id == "project-a"
        for result in outcome.results
    )


async def test_chunk_target_normalization_preserves_empty_neighbor_behavior() -> None:
    seed = _record("explanation")
    target_chunk = Record(
        workspace_id="project-a",
        source_kind="note",
        source_id="isolated_chunk_1",
        title="Isolated Target",
        body="target chunk",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
        metadata={"doc_id": "isolated"},
    )
    target = Record(
        workspace_id="project-a",
        source_kind="note",
        source_id="isolated",
        title="Isolated Target",
        body="target",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    graph_calls: list[RecordIdentity] = []

    class Graph(FakeGraphMutations):
        def neighbors(
            self,
            record_id: RecordIdentity | str,
            edge_types: list[str] | None = None,
            depth: int = 1,
            max_neighbors: int | None = None,
        ) -> list[GraphNeighbor]:
            assert isinstance(record_id, RecordIdentity)
            graph_calls.append(record_id)
            return []

    async def resolve(
        query: str,
        context: RecordSearchQueryContext,
    ) -> list[RecordHit]:
        return [RecordHit(target_chunk.identity, 2.0)]

    pipeline = RecordSearchPipeline(
        keyword_store=FakeKeywordStore([("explanation", 1.0)]),
        graph_store=Graph(),
        hydrator=_hydrator(
            {
                "explanation": seed,
                target_chunk.source_id: target_chunk,
                target.source_id: target,
            }
        ),
        policy=RecordSearchPolicy(graph_target_resolver=resolve),
        config=RecordSearchConfig(adaptive_graph_enabled=False),
    )

    outcome = await pipeline.async_search(
        "What documents are neighbors of Isolated Target?",
        limit=2,
        filters={"workspace_id": "project-a"},
    )

    assert graph_calls == [target.identity]
    assert [result.record_id for result in outcome.results] == ["explanation"]
    assert outcome.results[0].provenance.strategies == ("keyword",)
    assert outcome.failures == ()
    assert outcome.missing_record_ids == ()


async def test_keyword_and_embedding_work_overlap_without_candidate_gating() -> None:
    records = {"a": _record("a")}
    keyword_started = asyncio.Event()
    embedding_started = asyncio.Event()
    release = asyncio.Event()
    vector_started = asyncio.Event()

    class Keyword(FakeKeywordIndex):
        async def search(
            self,
            query: str,
            k: int,
            filters: SearchFilters | None = None,
        ) -> Sequence[RecordHit]:
            keyword_started.set()
            await release.wait()
            return _hits([("a", 1.0)])

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
            filters: SearchFilters | None = None,
        ) -> Sequence[RecordHit]:
            vector_started.set()
            return _hits([("a", 1.0)])

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
            filters: SearchFilters | None = None,
        ) -> Sequence[RecordHit]:
            keyword_started.set()
            await release_keyword.wait()
            keyword_finished.set()
            return _hits([("a", 1.0)])

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
            filters: SearchFilters | None = None,
        ) -> Sequence[RecordHit]:
            assert keyword_finished.is_set()
            vector_started.set()
            return _hits([("a", 1.0)])

    pipeline = RecordSearchPipeline(
        keyword_store=Keyword(),
        vector_store=Vector(),
        embedding_provider=Embedder(),
        hydrator=_hydrator({"a": _record("a")}),
        policy=RecordSearchPolicy(
            vector_candidate_ids=lambda ranking, filters: [
                hit.storage_key for hit in ranking
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
        keyword_store=FakeKeywordStore([("a", 40.0)]),
        vector_store=FakeVectorStore([("a", 0.9)]),
        embedding_provider=embedder,
        hydrator=_hydrator({"a": _record("a")}),
    )

    outcome = await pipeline.async_search("src/search_kernel.py", limit=1)

    assert [result.record_id for result in outcome.results] == ["a"]
    assert embedder.calls == 0
    assert "vector:artifact_keyword_confident" in outcome.diagnostics


async def test_artifact_eligibility_policy_approves_with_query_context() -> None:
    contexts: list[RecordSearchQueryContext] = []

    def eligible(
        ranking: Sequence[RecordHit], context: RecordSearchQueryContext
    ) -> bool:
        assert [hit.identity.source_id for hit in ranking] == ["a"]
        contexts.append(context)
        return context.query == "src/search_kernel.py" and context.limit == 1

    pipeline = RecordSearchPipeline(
        keyword_store=FakeKeywordStore([("a", 40.0)]),
        vector_store=FakeVectorStore([("a", 0.9)]),
        embedding_provider=FakeEmbedder(),
        hydrator=_hydrator({"a": _record("a")}),
        policy=RecordSearchPolicy(query_candidate_set_eligible=eligible),
    )

    await pipeline.async_search("src/search_kernel.py", limit=1)
    await pipeline.async_search("query", limit=1)

    assert len(contexts) == 1
    assert contexts[0]["statuses"] == ["active"]


async def test_artifact_eligibility_policy_veto_keeps_vector_lane() -> None:
    class CountingEmbedder:
        model_name = "fake-model"
        dim = 2

        def __init__(self) -> None:
            self.calls = 0

        def embed_query(self, query: str) -> list[float]:
            self.calls += 1
            return [1.0, 0.0]

    embedder = CountingEmbedder()
    vector_store = FakeVectorStore([("vector", 0.9)])
    pipeline = RecordSearchPipeline(
        keyword_store=FakeKeywordStore([("keyword", 40.0)]),
        vector_store=vector_store,
        embedding_provider=embedder,
        hydrator=_hydrator(
            {"keyword": _record("keyword"), "vector": _record("vector")}
        ),
        policy=RecordSearchPolicy(
            query_candidate_set_eligible=lambda ranking, context: False
        ),
    )

    outcome = await pipeline.async_search("src/search_kernel.py", limit=1)

    assert embedder.calls == 1
    assert vector_store.filters == [
        {
            "statuses": ["active"],
            "candidate_storage_keys": [
                RecordIdentity(None, "fake", "keyword").storage_key
            ],
        }
    ]
    assert "vector:artifact_keyword_ineligible" in outcome.diagnostics


async def test_artifact_keyword_results_bound_vector_acquisition() -> None:
    class AnyEmbedder(FakeEmbedder):
        def embed_query(self, query: str) -> list[float]:
            return [1.0, 0.0]

    keyword_store = FakeKeywordStore([("keyword", 0.1)])
    vector_store = FakeVectorStore([("vector", 0.9)])
    pipeline = RecordSearchPipeline(
        keyword_store=keyword_store,
        vector_store=vector_store,
        embedding_provider=AnyEmbedder(),
        hydrator=_hydrator(
            {"keyword": _record("keyword"), "vector": _record("vector")}
        ),
    )

    await pipeline.async_search("src/search_kernel.py", limit=1)

    assert vector_store.filters[0] == {
        "statuses": ["active"],
        "candidate_storage_keys": [
            RecordIdentity(None, "fake", "keyword").storage_key
        ],
    }


async def test_exact_artifact_query_keeps_empty_keyword_candidates_bounded() -> None:
    class OrderedVectorStore(FakeVectorStore):
        def search(
            self,
            query_vector: list[float],
            k: int,
            *,
            model_name: str,
            dim: int,
            filters: SearchFilters | None = None,
        ) -> Sequence[RecordHit]:
            assert keyword_store.queries
            return super().search(
                query_vector,
                k,
                model_name=model_name,
                dim=dim,
                filters=filters,
            )

    keyword_store = FakeKeywordStore([])
    vector_store = OrderedVectorStore([("vector", 0.9)])

    class AnyEmbedder(FakeEmbedder):
        def embed_query(self, query: str) -> list[float]:
            return [1.0, 0.0]

    pipeline = RecordSearchPipeline(
        keyword_store=keyword_store,
        vector_store=vector_store,
        embedding_provider=AnyEmbedder(),
        hydrator=_hydrator({"vector": _record("vector")}),
    )

    await pipeline.async_search('"src/search_kernel.py"', limit=1)

    assert vector_store.filters[0] == {
        "statuses": ["active"],
        "candidate_storage_keys": [],
    }


async def test_artifact_vector_only_search_falls_back_to_unbounded_acquisition() -> None:
    class AnyEmbedder(FakeEmbedder):
        def embed_query(self, query: str) -> list[float]:
            return [1.0, 0.0]

    vector_store = FakeVectorStore([("vector", 0.9)])
    pipeline = RecordSearchPipeline(
        vector_store=vector_store,
        embedding_provider=AnyEmbedder(),
        hydrator=_hydrator({"vector": _record("vector")}),
    )

    outcome = await pipeline.async_search("src/search_kernel.py", limit=1)

    assert [result.record_id for result in outcome.results] == ["vector"]
    assert vector_store.filters == [{"statuses": ["active"]}]
    assert "query_plan:skip:vector:keyword_unavailable_unbounded" in (
        outcome.diagnostics
    )


async def test_artifact_search_without_keyword_or_vector_returns_no_results() -> None:
    pipeline = RecordSearchPipeline(hydrator=_hydrator({}))

    outcome = await pipeline.async_search("src/search_kernel.py", limit=1)

    assert outcome.results == ()
    assert "query_plan:skip:keyword:unavailable" in outcome.diagnostics
    assert "query_plan:skip:vector:unavailable" in outcome.diagnostics


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
            max_neighbors: int | None = None,
        ) -> Sequence[GraphNeighbor]:
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


async def test_ordinary_query_does_not_touch_available_graph_store() -> None:
    class FailingGraph(FakeGraphStore):
        def neighbors(
            self,
            record_id: RecordIdentity | str,
            edge_types: list[str] | None = None,
            depth: int = 1,
            max_neighbors: int | None = None,
        ) -> Sequence[GraphNeighbor]:
            raise AssertionError("graph should be skipped for ordinary queries")

    pipeline = RecordSearchPipeline(
        keyword_store=FakeKeywordStore([("a", 1.0)]),
        graph_store=FailingGraph({}),
        hydrator=_hydrator({"a": _record("a")}),
        config=RecordSearchConfig(adaptive_graph_enabled=False),
    )

    outcome = await pipeline.async_search("what is caching?", limit=1)

    assert outcome.results[0].record_id == "a"
    assert "query_plan:skip:graph:query_not_relationship" in outcome.diagnostics


async def test_adaptive_graph_expands_strong_ordinary_seed_without_displacing_it() -> None:
    records = {"seed": _record("seed"), "neighbor": _record("neighbor")}
    pipeline = RecordSearchPipeline(
        keyword_store=FakeKeywordStore([("seed", 40.0)]),
        graph_store=FakeGraphStore(
            {"seed": [("neighbor", "related", 1.0)]}
        ),
        hydrator=_hydrator(records),
        config=RecordSearchConfig(),
    )

    outcome = await pipeline.async_search("what is caching?", limit=2)

    assert [result.record_id for result in outcome.results] == [
        "seed",
        "neighbor",
    ]
    assert outcome.results[1].provenance.strategies == ("graph",)
    assert "query_plan:graph:adaptive" in outcome.diagnostics


async def test_adaptive_graph_skips_weak_ordinary_seed() -> None:
    class FailingGraph(FakeGraphStore):
        def neighbors(
            self,
            record_id: RecordIdentity | str,
            edge_types: list[str] | None = None,
            depth: int = 1,
            max_neighbors: int | None = None,
        ) -> Sequence[GraphNeighbor]:
            raise AssertionError("weak seeds should not route to graph")

    pipeline = RecordSearchPipeline(
        keyword_store=FakeKeywordStore([("seed", 0.5)]),
        graph_store=FailingGraph({}),
        hydrator=_hydrator({"seed": _record("seed")}),
        config=RecordSearchConfig(adaptive_graph_enabled=True),
    )

    outcome = await pipeline.async_search("what is caching?", limit=1)

    assert [result.record_id for result in outcome.results] == ["seed"]
    assert "query_plan:skip:graph:awaiting_seed_confidence" in outcome.diagnostics


async def test_conditional_expansion_is_called_once_after_weak_first_pass() -> None:
    class AnyEmbedder(FakeEmbedder):
        def embed_query(self, query: str) -> list[float]:
            return [1.0, 0.0]

    class ExpandingKeyword(FakeKeywordStore):
        def search(
            self,
            query: str,
            k: int,
            filters: SearchFilters | None = None,
        ) -> Sequence[RecordHit]:
            self.queries.append((query, k, filters))
            return (
                _hits([("a", 1.0)])
                if query == "what is thing?"
                else _hits([("b", 0.9)])
            )

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


async def test_opt_in_query_expander_adds_bounded_synonyms() -> None:
    keyword_store = FakeKeywordStore([("a", 1.0)])

    def expand(query: str) -> list[str]:
        assert query == "deploy issue"
        return ["incident", "outage", "failure", "unbounded"]

    pipeline = RecordSearchPipeline(
        keyword_store=keyword_store,
        hydrator=_hydrator({"a": _record("a")}),
        policy=RecordSearchPolicy(query_expander=expand),
        config=RecordSearchConfig(
            synonym_expansion_enabled=True,
            synonym_expansion_max_terms=2,
        ),
    )

    outcome = await pipeline.async_search("deploy issue", limit=2)

    assert any(
        query == "deploy issue incident outage"
        for query, _, _ in keyword_store.queries
    )
    assert "synonym_expansion:applied" in outcome.diagnostics


async def test_graph_and_query_expansion_compose_through_one_fusion_boundary() -> None:
    class ExpandingKeyword(FakeKeywordStore):
        def search(
            self,
            query: str,
            k: int,
            filters: SearchFilters | None = None,
        ) -> Sequence[RecordHit]:
            self.queries.append((query, k, filters))
            return _hits(
                [("seed", 40.0)]
                if query == "deploy issue"
                else [("expanded", 1.0)]
            )

    def expand(query: str) -> str:
        assert query == "deploy issue"
        return "deploy issue outage"

    records = {
        record_id: _record(record_id)
        for record_id in ("seed", "neighbor", "expanded")
    }
    pipeline = RecordSearchPipeline(
        keyword_store=ExpandingKeyword([]),
        graph_store=FakeGraphStore({"seed": [("neighbor", "related", 1.0)]}),
        hydrator=_hydrator(records),
        policy=RecordSearchPolicy(query_expander=expand),
        config=RecordSearchConfig(synonym_expansion_enabled=True),
    )

    outcome = await pipeline.async_search("deploy issue", limit=3)

    results_by_id = {result.record_id: result for result in outcome.results}
    assert set(results_by_id) == {"seed", "neighbor", "expanded"}
    assert results_by_id["seed"].provenance.strategies == ("keyword",)
    assert results_by_id["neighbor"].provenance.strategies == ("graph",)
    assert results_by_id["expanded"].provenance.strategies == ("expansion",)


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
    assert "diagnostics" in _mapping(trace["provenance"])
    assert outcome.candidate_count == 1
    assert outcome.candidate_counts == {"keyword": 1}
    assert outcome.stage_timings_ms["search"] >= 0
    assert outcome.diagnostic_evidence is not None
    assert outcome.diagnostic_evidence.stage_timings_ms == outcome.stage_timings_ms


async def test_rerank_runs_once_with_a_bounded_candidate_set() -> None:
    class Reranker:
        model_name = "fake-reranker"

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
        model_name = "failing-reranker"

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
    assert outcome.diagnostic_evidence is not None
    assert outcome.diagnostic_evidence.degraded
    assert outcome.diagnostic_evidence.failures == outcome.failures


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
            filters: SearchFilters | None = None,
        ) -> list[RecordHit]:
            return [RecordHit(seed, 1.0)]

    class Graph(FakeGraphMutations):
        def upsert_edges(
            self,
            edges: Sequence[GraphEdge | tuple[str, str, str, float]],
        ) -> None:
            pass

        def delete_edges(
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
            max_neighbors: int | None = None,
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
            max_neighbors: int | None = None,
        ) -> Sequence[GraphNeighbor]:
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

    outcome = await pipeline.async_search("what relates to the seed?", limit=2)

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


async def test_batch_graph_normalizes_legacy_source_id_seed_keys() -> None:
    records = {record_id: _record(record_id) for record_id in ("seed", "target")}
    seed = records["seed"].identity
    target = records["target"].identity

    class Graph(FakeGraphMutations):
        def neighbors_many(
            self,
            identities: Sequence[RecordIdentity],
            *,
            depth: int,
            max_neighbors: int | None = None,
        ) -> dict[str, list[GraphNeighbor]]:
            assert identities == [seed]
            return {"seed": [GraphNeighbor(target, "related", 1.0)]}

        def neighbors(
            self,
            record_id: RecordIdentity | str,
            edge_types: list[str] | None = None,
            depth: int = 1,
            max_neighbors: int | None = None,
        ) -> Sequence[GraphNeighbor]:
            raise AssertionError("scalar graph lookup should not run")

    pipeline = RecordSearchPipeline(
        keyword_store=FakeKeywordStore([("seed", 1.0)]),
        graph_store=Graph(),
        hydrator=_hydrator(records),
    )

    outcome = await pipeline.async_search("what relates to the seed?", limit=2)

    assert [result.record_id for result in outcome.results] == ["seed", "target"]
    assert outcome.results[1].provenance.strategies == ("graph",)


@pytest.mark.parametrize(
    ("depth", "expected_ids"),
    [(1, ["seed", "one-hop"]), (2, ["seed", "one-hop", "two-hop"])],
)
async def test_graph_retrieval_recovers_neighbors_from_partial_batch(
    depth: int,
    expected_ids: list[str],
) -> None:
    records = {
        record_id: _record(record_id)
        for record_id in ("seed", "one-hop", "two-hop")
    }

    class Graph(FakeGraphMutations):
        def neighbors_many(
            self,
            identities: Sequence[RecordIdentity],
            *,
            depth: int,
        ) -> dict[str, list[GraphNeighbor]]:
            return {}

        def neighbors(
            self,
            record_id: RecordIdentity | str,
            edge_types: list[str] | None = None,
            depth: int = 1,
            max_neighbors: int | None = None,
        ) -> list[GraphNeighbor]:
            assert isinstance(record_id, RecordIdentity)
            assert max_neighbors == 10
            if record_id.source_id != "seed":
                return []
            neighbors = [
                GraphNeighbor(
                    records["one-hop"].identity,
                    "links_to",
                    1.0,
                )
            ]
            if depth > 1:
                neighbors.append(
                    GraphNeighbor(
                        records["two-hop"].identity,
                        "links_to",
                        0.5,
                    )
                )
            return neighbors

    pipeline = RecordSearchPipeline(
        keyword_store=FakeKeywordStore([("seed", 1.0)]),
        graph_store=Graph(),
        hydrator=_hydrator(records),
        config=RecordSearchConfig(graph_depth=depth),
    )

    outcome = await pipeline.async_search("what is linked to the seed?", limit=3)

    assert {result.record_id for result in outcome.results} == set(expected_ids)
    assert {
        result.record_id
        for result in outcome.results
        if result.provenance.strategies == ("graph",)
    } == set(expected_ids[1:])


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


async def test_hydration_merges_cache_hits_and_batch_loads_in_candidate_order() -> None:
    records = {record_id: _record(record_id) for record_id in ("a", "b")}

    class BatchHydrator:
        async def hydrate_records(
            self,
            identities: Sequence[RecordIdentity],
        ) -> dict[str, Record]:
            assert [identity.source_id for identity in identities] == ["a"]
            return {identities[0].storage_key: records["a"]}

    identity_b = RecordIdentity(None, "fake", "b")
    cache = HydrationCache()
    cache.set(
        HydrationCacheKey.build(
            identity_b,
            record_version=1,
            policy_version="policy/v1",
        ),
        records["b"],
    )
    pipeline = RecordSearchPipeline(
        keyword_store=FakeKeywordStore([("a", 1.0), ("b", 0.9)]),
        hydrator=BatchHydrator(),
        hydration_cache=cache,
        hydration_version=1,
        policy_version="policy/v1",
    )

    outcome = await pipeline.async_search("query", limit=2)

    assert [result.record_id for result in outcome.results] == ["a", "b"]


async def test_bulk_hydration_batches_misses_and_preserves_candidate_order() -> None:
    """Hydration batches stay bounded while all ranked records are returned."""
    record_ids = [f"record-{index}" for index in range(5)]
    records = {record_id: _record(record_id) for record_id in record_ids}

    class BatchHydrator:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        async def hydrate_records(
            self,
            identities: Sequence[RecordIdentity],
        ) -> dict[str, Record]:
            self.calls.append([identity.source_id for identity in identities])
            return {
                identity.storage_key: records[identity.source_id]
                for identity in identities
            }

    hydrator = BatchHydrator()
    pipeline = RecordSearchPipeline(
        keyword_store=FakeKeywordStore([(record_id, 1.0) for record_id in record_ids]),
        hydrator=hydrator,
        config=RecordSearchConfig(max_hydration_batch_size=2),
    )

    outcome = await pipeline.async_search("query", limit=5)

    assert hydrator.calls == [
        ["record-0", "record-1"],
        ["record-2", "record-3"],
        ["record-4"],
    ]
    assert [result.record_id for result in outcome.results] == record_ids


async def test_missing_top_candidate_backfills_from_lower_ranked_candidates() -> None:
    records = {record_id: _record(record_id) for record_id in ("b", "c")}

    class BatchHydrator:
        def __init__(self) -> None:
            self.identities: list[list[str]] = []

        async def hydrate_records(
            self,
            identities: Sequence[RecordIdentity],
        ) -> dict[str, Record | None]:
            self.identities.append([identity.source_id for identity in identities])
            return {
                identity.storage_key: records.get(identity.source_id)
                for identity in identities
            }

    hydrator = BatchHydrator()
    pipeline = RecordSearchPipeline(
        keyword_store=FakeKeywordStore(
            [("a", 1.0), ("b", 0.9), ("c", 0.8)]
        ),
        hydrator=hydrator,
    )

    outcome = await pipeline.async_search("query", limit=2)

    assert [result.record_id for result in outcome.results] == ["b", "c"]
    assert outcome.missing_record_ids == ("a",)
    assert hydrator.identities == [["a", "b"], ["c"]]
    assert outcome.diagnostic_evidence is not None
    assert outcome.diagnostic_evidence.missing_record_ids == ("a",)
    assert outcome.diagnostic_evidence.degraded


async def test_diagnostics_count_duplicates_after_final_post_processing() -> None:
    """Final duplicate evidence reflects the returned result sequence."""
    record = _record("a")
    pipeline = RecordSearchPipeline(
        keyword_store=FakeKeywordStore([("a", 1.0)]),
        vector_store=FakeVectorStore([("a", 0.9)]),
        embedding_provider=FakeEmbedder(),
        hydrator=_hydrator({"a": record}),
        policy=RecordSearchPolicy(
            post_process=lambda results: [*results, results[0]],
        ),
    )

    outcome = await pipeline.async_search("query", limit=2)

    assert [result.record_id for result in outcome.results] == ["a", "a"]
    assert outcome.diagnostic_evidence is not None
    assert outcome.diagnostic_evidence.raw_pre_fusion_overlap.count == 1
    assert outcome.diagnostic_evidence.final_duplicate_count == 1


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

        def delete_edges(
            self,
            edges: Sequence[GraphEdge | tuple[str, str, str, float]],
        ) -> None:
            pass

        def neighbors(
            self,
            record_id: RecordIdentity | str,
            edge_types: list[str] | None = None,
            depth: int = 1,
            max_neighbors: int | None = None,
        ) -> Sequence[GraphNeighbor]:
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

    await pipeline.async_search("what relates to these records?", limit=4)

    assert maximum_active == 2


async def test_batch_graph_failures_keep_strict_and_lenient_modes() -> None:
    class BrokenGraph(FakeGraphMutations):
        def upsert_edges(
            self,
            edges: Sequence[GraphEdge | tuple[str, str, str, float]],
        ) -> None:
            pass

        def delete_edges(
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
            max_neighbors: int | None = None,
        ) -> Sequence[GraphNeighbor]:
            raise AssertionError("scalar graph lookup should not run")

    strict = RecordSearchPipeline(
        keyword_store=FakeKeywordStore([("a", 1.0)]),
        graph_store=BrokenGraph(),
        hydrator=_hydrator({"a": _record("a")}),
    )
    with pytest.raises(RecordSearchError, match="graph retrieval failed"):
        await strict.async_search("what relates to record a?")

    lenient = RecordSearchPipeline(
        keyword_store=FakeKeywordStore([("a", 1.0)]),
        graph_store=BrokenGraph(),
        hydrator=_hydrator({"a": _record("a")}),
        continue_on_error=True,
    )
    outcome = await lenient.async_search("what relates to record a?")
    assert [result.record_id for result in outcome.results] == ["a"]
    assert outcome.failures[0].stage == "graph"
    assert outcome.diagnostic_evidence is not None
    assert outcome.diagnostic_evidence.raw_pre_fusion_overlap.available is False


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
            filters: SearchFilters | None = None,
        ) -> Sequence[RecordHit]:
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


async def test_exact_identifier_matches_keep_their_relative_order() -> None:
    """
    Two records can share a source_id across workspaces, so one query can
    match both exactly. Precedence over other results must not flatten the
    order between them.
    """
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    identities = {
        workspace: RecordIdentity(workspace, "fake", "doc")
        for workspace in ("w1", "w2")
    }
    records = {
        workspace: Record(
            workspace_id=workspace,
            source_kind="fake",
            source_id="doc",
            title=workspace,
            body=f"body for {workspace}",
            created_at=timestamp,
            updated_at=timestamp,
        )
        for workspace in identities
    }

    class WorkspaceKeywordStore:
        def index(self, records: list[Record]) -> None:
            pass

        def search(
            self,
            query: str,
            k: int,
            filters: SearchFilters | None = None,
        ) -> list[RecordHit]:
            # w2 is the better match but sorts second by storage key.
            return [
                RecordHit(identities["w2"], 9.0),
                RecordHit(identities["w1"], 1.0),
            ]

    def hydrate(record_id: RecordIdentity) -> Record | None:
        return records.get(record_id.workspace_id or "")

    pipeline = RecordSearchPipeline(
        keyword_store=WorkspaceKeywordStore(),
        hydrator=hydrate,
    )

    outcome = await pipeline.async_search("doc", limit=2)

    assert [result.record.workspace_id for result in outcome.results] == ["w2", "w1"]


async def test_exact_identifier_outranks_an_aggregated_chunk_parent() -> None:
    """
    Chunk aggregation re-sorts the final results, so it must not discard the
    precedence an exact identifier match already earned.
    """
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    parent = _record("parent")
    chunk = Record(
        source_kind="fake",
        source_id=f"{parent.identity.storage_key}#chunk:0",
        title="chunk",
        body="chunk body",
        created_at=timestamp,
        updated_at=timestamp,
        metadata={
            "_searchkernel_chunk": True,
            "_chunk_id": "0",
            "_chunk_parent_storage_key": parent.identity.storage_key,
            "_chunk_metadata": {},
        },
    )
    exact = _record("ENG-939")
    pipeline = RecordSearchPipeline(
        keyword_store=FakeKeywordStore(
            [RecordHit(chunk.identity, 9.0), RecordHit(exact.identity, 1.0)]
        ),
        hydrator=_hydrator(
            {
                chunk.source_id: chunk,
                parent.source_id: parent,
                exact.source_id: exact,
            }
        ),
    )

    outcome = await pipeline.async_search("eng-939", limit=2)

    assert [result.record_id for result in outcome.results] == ["ENG-939", "parent"]
