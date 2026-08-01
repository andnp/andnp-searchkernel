"""Golden-contract fixtures shared by the canonical and legacy search paths.

These tests compare result identity and ordering only.  The two paths are
allowed to use different stores and stage implementations while migration is
in progress.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

from searchkernel.domain import (
    GraphEdge,
    GraphNeighbor,
    Record,
    RecordHit,
    RecordIdentity,
    RecordStatus,
    SearchResultProvenance,
)
from searchkernel.pipeline.stage import SearchContext
from searchkernel.pipeline.stages.graph_expand import GraphExpandStage
from searchkernel.pipeline.stages.parent_expansion import ParentExpansionStage
from searchkernel.search.pipeline import SearchPipeline, SearchPipelineConfig
from searchkernel.search.record_pipeline import (
    RecordSearchConfig,
    RecordSearchPipeline,
    RecordSearchPolicy,
)
from searchkernel.search.score_pipeline import ScorePipeline


@dataclass(frozen=True)
class _ParityCase:
    records: dict[str, Record]
    keyword: list[tuple[str, float]]
    vector: list[tuple[str, float]]
    graph: dict[str, list[tuple[str, str, float]]]


def _record(
    source_id: str,
    body: str,
    *,
    source_kind: str = "note",
    workspace_id: str | None = "workspace",
    status: RecordStatus = RecordStatus.ACTIVE,
    parent_chunk_id: str | None = None,
) -> Record:
    metadata = {}
    if parent_chunk_id is not None:
        metadata["parent_chunk_id"] = parent_chunk_id
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    return Record(
        workspace_id=workspace_id,
        source_kind=source_kind,
        source_id=source_id,
        title=source_id,
        body=body,
        created_at=timestamp,
        updated_at=timestamp,
        metadata=metadata,
        status=status,
    )


def _hybrid_case() -> _ParityCase:
    records = {
        source_id: _record(source_id, f"body for {source_id}")
        for source_id in ("a", "b", "c")
    }
    return _ParityCase(
        records=records,
        keyword=[("b", 10.0), ("a", 1.0)],
        vector=[("a", 0.9), ("c", 0.8)],
        graph={},
    )


class _KeywordStore:
    def __init__(self, case: _ParityCase) -> None:
        self._case = case

    def search(
        self,
        _query: str,
        k: int,
        filters: dict[str, object] | None = None,
    ) -> list[RecordHit]:
        return [
            RecordHit(self._identity(source_id), score)
            for source_id, score in self._case.keyword
            if self._eligible(source_id, filters)
        ][:k]

    def index(self, _records: list[Record]) -> None:
        pass

    def _identity(self, source_id: str) -> RecordIdentity:
        record = self._case.records[source_id]
        return RecordIdentity(record.workspace_id, record.source_kind, record.source_id)

    def _eligible(self, source_id: str, filters: dict[str, object] | None) -> bool:
        return _eligible(self._case.records[source_id], filters)


class _VectorStore:
    def __init__(self, case: _ParityCase) -> None:
        self._case = case

    def search(
        self,
        _query_vector: list[float],
        k: int,
        *,
        model_name: str,
        dim: int,
        filters: dict[str, object] | None = None,
    ) -> list[RecordHit]:
        assert (model_name, dim) == ("fixture", 2)
        return [
            RecordHit(self._identity(source_id), score)
            for source_id, score in self._case.vector
            if _eligible(self._case.records[source_id], filters)
        ][:k]

    def upsert(self, _records: list[Record], _model_name: str, _dim: int) -> None:
        pass

    def delete(self, _record_ids: list[str]) -> None:
        pass

    def epoch(self) -> int:
        return 0

    def _identity(self, source_id: str) -> RecordIdentity:
        record = self._case.records[source_id]
        return RecordIdentity(record.workspace_id, record.source_kind, record.source_id)


class _GraphStore:
    def __init__(self, case: _ParityCase) -> None:
        self._case = case

    def neighbors(
        self,
        record_id: RecordIdentity | str,
        edge_types: list[str] | None = None,
        depth: int = 1,
    ) -> list[GraphNeighbor]:
        del edge_types, depth
        source_id = (
            record_id.source_id
            if isinstance(record_id, RecordIdentity)
            else record_id
        )
        return [
            GraphNeighbor(
                _identity(self._case.records[target]),
                edge_type,
                weight,
            )
            for target, edge_type, weight in self._case.graph.get(source_id, [])
        ]

    def upsert_edges(
        self,
        _edges: Sequence[GraphEdge | tuple[str, str, str, float]],
    ) -> None:
        pass


class _Embedder:
    model_name = "fixture"
    dim = 2

    def embed_query(self, query: str) -> list[float]:
        assert query == "hybrid query"
        return [1.0, 0.0]


def _eligible(record: Record, filters: dict[str, object] | None) -> bool:
    filters = filters or {}
    statuses = filters.get("statuses", ["active"])
    if isinstance(statuses, list) and record.status.value not in statuses:
        return False
    source_kinds = filters.get("source_kinds")
    if isinstance(source_kinds, list) and record.source_kind not in source_kinds:
        return False
    workspace_id = filters.get("workspace_id")
    return workspace_id is None or record.workspace_id == workspace_id


def _identity(record: Record) -> RecordIdentity:
    return RecordIdentity(record.workspace_id, record.source_kind, record.source_id)


def _canonical_pipeline(
    case: _ParityCase,
    *,
    policy: RecordSearchPolicy | None = None,
    config: RecordSearchConfig | None = None,
) -> RecordSearchPipeline:
    return RecordSearchPipeline(
        keyword_store=_KeywordStore(case),
        vector_store=_VectorStore(case) if case.vector else None,
        graph_store=_GraphStore(case) if case.graph else None,
        embedding_provider=_Embedder() if case.vector else None,
        hydrator=lambda identity: case.records.get(
            identity.source_id if isinstance(identity, RecordIdentity) else identity
        ),
        policy=policy,
        config=config,
    )


@pytest.mark.asyncio
async def test_hybrid_fixture_preserves_legacy_order_identity_and_provenance() -> None:
    case = _hybrid_case()

    legacy = ScorePipeline().fuse(
        {"semantic": case.vector, "keyword": case.keyword}
    )
    canonical = await _canonical_pipeline(case).async_search(
        "hybrid query",
        limit=3,
    )

    assert [source_id for source_id, _score in legacy] == [
        result.record_id for result in canonical.results
    ]
    assert [result.record_id for result in canonical.results] == ["a", "b", "c"]
    first = canonical.results[0]
    assert first.provenance.record_identity == RecordIdentity(
        "workspace", "note", "a"
    )
    assert set(first.provenance.strategy_details) == {"keyword", "vector"}
    assert first.provenance.strategy_details["vector"].rank == 1
    assert first.provenance.strategy_details["keyword"].rank == 2


@pytest.mark.asyncio
async def test_tie_fixture_uses_stable_identity_order_in_both_paths() -> None:
    records = {
        "a": _record("a", "body a"),
        "b": _record("b", "body b"),
    }
    case = _ParityCase(
        records=records,
        keyword=[("b", 1.0)],
        vector=[("a", 1.0)],
        graph={},
    )

    legacy = ScorePipeline().fuse(
        {"semantic": case.vector, "keyword": case.keyword}
    )
    canonical = await _canonical_pipeline(case).async_search(
        "hybrid query",
        limit=2,
    )

    assert [source_id for source_id, _score in legacy] == ["a", "b"]
    assert [result.record_id for result in canonical.results] == ["a", "b"]


@pytest.mark.asyncio
async def test_filter_fixture_preserves_eligible_identity_across_paths() -> None:
    records = {
        "allowed": _record("allowed", "alpha", source_kind="note"),
        "wrong-source": _record("wrong-source", "alpha", source_kind="commit"),
        "archived": _record(
            "archived",
            "alpha",
            source_kind="note",
            status=RecordStatus.ARCHIVED,
        ),
    }
    case = _ParityCase(
        records=records,
        keyword=[("wrong-source", 1.0), ("archived", 0.9), ("allowed", 0.8)],
        vector=[],
        graph={},
    )

    legacy_candidates = [
        (source_id, score)
        for source_id, score in case.keyword
        if case.records[source_id].source_kind == "note"
        and case.records[source_id].status is RecordStatus.ACTIVE
    ]
    legacy, _stats = SearchPipeline(
        SearchPipelineConfig(reranking_enabled=False)
    ).process(
        legacy_candidates,
        lambda _source_id: None,
        lambda source_id: case.records[source_id].body,
        "alpha",
        top_n=3,
    )
    canonical = await _canonical_pipeline(case).async_search(
        "alpha",
        limit=3,
        filters={"source_kinds": ["note"]},
    )

    assert [source_id for source_id, _score in legacy] == [
        result.record_id for result in canonical.results
    ]
    assert [result.record_id for result in canonical.results] == ["allowed"]


@pytest.mark.asyncio
async def test_dedup_fixture_keeps_the_first_ranked_identity_in_both_paths() -> None:
    records = {
        "duplicate-first": _record("duplicate-first", "same content"),
        "duplicate-second": _record("duplicate-second", "same content"),
        "unique": _record("unique", "different content"),
    }
    case = _ParityCase(
        records=records,
        keyword=[
            ("duplicate-first", 1.0),
            ("duplicate-second", 0.9),
            ("unique", 0.8),
        ],
        vector=[],
        graph={},
    )
    legacy, _stats = SearchPipeline(
        SearchPipelineConfig(reranking_enabled=False)
    ).process(
        case.keyword,
        lambda _source_id: None,
        lambda source_id: case.records[source_id].body,
        "duplicate",
        top_n=3,
    )

    seen_bodies: set[str] = set()

    def deduplicate(results):
        kept = []
        for result in results:
            if result.record.body not in seen_bodies:
                seen_bodies.add(result.record.body)
                kept.append(result)
        return kept

    canonical = await _canonical_pipeline(
        case,
        policy=RecordSearchPolicy(post_process=deduplicate),
    ).async_search("duplicate", limit=3)

    assert [source_id for source_id, _score in legacy] == [
        result.record_id for result in canonical.results
    ]
    assert [result.record_id for result in canonical.results] == [
        "duplicate-first",
        "unique",
    ]


@pytest.mark.asyncio
async def test_graph_fixture_preserves_seed_neighbor_identity_and_provenance() -> None:
    records = {
        "a-seed": _record("a-seed", "seed body"),
        "z-neighbor": _record("z-neighbor", "neighbor body"),
    }
    case = _ParityCase(
        records=records,
        keyword=[("a-seed", 1.0)],
        vector=[],
        graph={"a-seed": [("z-neighbor", "related", 0.5)]},
    )

    legacy_context = SearchContext(
        "what relates to a-seed?",
        metadata={
            "seed_scores": {"a-seed": 1.0},
            "top_k": 2,
            "excluded_chunk_ids": None,
        },
    )
    legacy_graph = GraphExpandStage(
        rank_neighbors=lambda _scores: [("z-neighbor", 0.5)],
        build_chunk_candidates=lambda ids, _top_k, _excluded: ids,
    ).run(legacy_context)

    canonical = await _canonical_pipeline(case).async_search(
        "what relates to a-seed?",
        limit=2,
    )

    assert legacy_graph.state.graph_chunk_ids == ["z-neighbor"]
    assert [result.record_id for result in canonical.results] == [
        "a-seed",
        "z-neighbor",
    ]
    neighbor = canonical.results[1]
    assert neighbor.provenance.record_identity == RecordIdentity(
        "workspace", "note", "z-neighbor"
    )
    assert "graph" in neighbor.provenance.strategies


def test_parent_fixture_locks_legacy_identity_and_provenance_contract() -> None:
    child = {
        "chunk_id": "child",
        "doc_id": "doc",
        "metadata": {"parent_chunk_id": "parent"},
    }
    parent = {"chunk_id": "parent", "doc_id": "doc", "metadata": {}}
    context = SearchContext(
        "query",
        candidates=[("child", 0.8)],
        metadata={
            "result_provenance": {
                "child": SearchResultProvenance()
            }
        },
    )
    context.state.result_provenance["child"].add_strategy("keyword", 1, 0.8)

    expanded = ParentExpansionStage(
        get_chunk=lambda chunk_id: {"child": child, "parent": parent}.get(chunk_id),
        get_parent_chunk=lambda chunk_id: parent if chunk_id == "parent" else None,
    ).run(context)

    assert expanded.candidates == [("parent", 0.8)]
    assert expanded.state.result_provenance["parent"].parent_expanded_from == "child"
