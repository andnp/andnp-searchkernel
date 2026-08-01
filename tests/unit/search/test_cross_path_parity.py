"""Golden-contract fixtures for direct and public record search composition."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pytest

from searchkernel import SearchKernel
from searchkernel.domain import (
    GraphEdge,
    GraphNeighbor,
    Record,
    RecordHit,
    RecordIdentity,
    RecordStatus,
)
from searchkernel.search.record_pipeline import (
    RecordSearchConfig,
    RecordSearchPipeline,
    RecordSearchPolicy,
)


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
        query: str,
        k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RecordHit]:
        del query
        return [
            RecordHit(self._identity(source_id), score)
            for source_id, score in self._case.keyword
            if self._eligible(source_id, filters)
        ][:k]

    def index(self, records: list[Record]) -> None:
        del records

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
        query_vector: list[float],
        k: int,
        *,
        model_name: str,
        dim: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RecordHit]:
        del query_vector
        assert (model_name, dim) == ("fixture", 2)
        return [
            RecordHit(self._identity(source_id), score)
            for source_id, score in self._case.vector
            if _eligible(self._case.records[source_id], filters)
        ][:k]

    def upsert(self, records: list[Record], model_name: str, dim: int) -> None:
        del records, model_name, dim

    def delete(self, record_ids: list[str]) -> None:
        del record_ids

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
        edges: Sequence[GraphEdge | tuple[str, str, str, float]],
    ) -> None:
        del edges

    def delete_edges(
        self,
        edges: Sequence[GraphEdge | tuple[str, str, str, float]],
    ) -> None:
        del edges


class _Embedder:
    model_name = "fixture"
    dim = 2

    def embed_query(self, query: str) -> list[float]:
        assert query in {"hybrid query", "what relates to a-seed?"}
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


def _hydrator(case: _ParityCase):
    return lambda identity: case.records.get(
        identity.source_id if isinstance(identity, RecordIdentity) else identity
    )


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
        hydrator=_hydrator(case),
        policy=policy,
        config=config,
    )


def _public_kernel(
    case: _ParityCase,
    *,
    policy: RecordSearchPolicy | None = None,
    config: RecordSearchConfig | None = None,
) -> SearchKernel:
    return SearchKernel.build(
        record_hydrator=_hydrator(case),
        keyword_store=_KeywordStore(case),
        vector_store=_VectorStore(case) if case.vector else None,
        graph_store=_GraphStore(case) if case.graph else None,
        embedding_provider=_Embedder() if case.vector else None,
        search_policy=policy,
        search_config=config,
    )


@pytest.mark.asyncio
async def test_hybrid_fixture_preserves_identity_and_provenance() -> None:
    case = _hybrid_case()

    direct = await _canonical_pipeline(case).async_search(
        "hybrid query",
        limit=3,
    )
    public = await _public_kernel(case).search_anything(
        "hybrid query",
        k=3,
    )

    assert [result.record_id for result in direct.results] == ["a", "b", "c"]
    assert [result.record_id for result in public] == [
        result.record_id for result in direct.results
    ]
    assert public[0].metadata["provenance"] == direct.results[0].provenance.to_dict()


@pytest.mark.asyncio
async def test_tie_fixture_uses_stable_identity_order() -> None:
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

    direct = await _canonical_pipeline(case).async_search(
        "hybrid query",
        limit=2,
    )
    public = await _public_kernel(case).search_anything(
        "hybrid query",
        k=2,
    )

    assert [result.record_id for result in direct.results] == ["a", "b"]
    assert [result.record_id for result in public] == ["a", "b"]


@pytest.mark.asyncio
async def test_filter_fixture_preserves_eligible_identity() -> None:
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

    direct = await _canonical_pipeline(case).async_search(
        "alpha",
        limit=3,
        filters={"source_kinds": ["note"]},
    )
    public = await _public_kernel(case).search_anything(
        "alpha",
        k=3,
        filters={"source_kinds": ["note"]},
    )

    assert [result.record_id for result in direct.results] == ["allowed"]
    assert [result.record_id for result in public] == ["allowed"]


@pytest.mark.asyncio
async def test_dedup_fixture_keeps_the_first_ranked_identity() -> None:
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

    seen_bodies: set[str] = set()

    def deduplicate(results):
        kept = []
        for result in results:
            if result.record.body not in seen_bodies:
                seen_bodies.add(result.record.body)
                kept.append(result)
        return kept

    policy = RecordSearchPolicy(post_process=deduplicate)
    direct = await _canonical_pipeline(case, policy=policy).async_search(
        "duplicate",
        limit=3,
    )
    seen_bodies.clear()
    public = await _public_kernel(case, policy=policy).search_anything(
        "duplicate",
        k=3,
    )

    assert [result.record_id for result in direct.results] == [
        "duplicate-first",
        "unique",
    ]
    assert [result.record_id for result in public] == [
        result.record_id for result in direct.results
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

    direct = await _canonical_pipeline(case).async_search(
        "what relates to a-seed?",
        limit=2,
    )
    public = await _public_kernel(case).search_anything(
        "what relates to a-seed?",
        k=2,
    )

    assert [result.record_id for result in direct.results] == [
        "a-seed",
        "z-neighbor",
    ]
    assert [result.record_id for result in public] == [
        result.record_id for result in direct.results
    ]
    assert direct.results[1].provenance.record_identity == _identity(
        records["z-neighbor"]
    )
    assert "graph" in direct.results[1].provenance.strategies


@pytest.mark.asyncio
async def test_parent_fixture_preserves_identity_and_provenance() -> None:
    records = {
        "child": _record("child", "child body", parent_chunk_id="parent"),
        "parent": _record("parent", "parent body"),
    }
    case = _ParityCase(
        records=records,
        keyword=[("child", 0.8)],
        vector=[],
        graph={},
    )

    class ParentExpander:
        def parent_identity(
            self,
            identity: RecordIdentity,
        ) -> RecordIdentity | None:
            record = records.get(identity.source_id)
            parent_id = record.metadata.get("parent_chunk_id") if record else None
            if parent_id is None or parent_id not in records:
                return None
            return _identity(records[str(parent_id)])

    policy = RecordSearchPolicy(parent_expander=ParentExpander())
    direct = await _canonical_pipeline(case, policy=policy).async_search(
        "query",
        limit=1,
    )
    public = await _public_kernel(case, policy=policy).search_anything(
        "query",
        k=1,
    )

    assert [result.record_id for result in direct.results] == ["parent"]
    assert [result.record_id for result in public] == ["parent"]
    provenance = direct.results[0].provenance
    assert provenance.record_identity == _identity(records["parent"])
    assert provenance.parent_expanded_from == "child"
    assert provenance.parent_expanded_from_identity == _identity(records["child"])
