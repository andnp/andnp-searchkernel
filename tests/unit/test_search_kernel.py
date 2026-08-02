from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any, cast

import pytest

from searchkernel import SearchAPI, SearchKernel
from searchkernel.domain import (
    ChunkResult,
    Record,
    RecordHit,
    RecordIdentity,
    ScoredRef,
    SearchResult,
    SearchResultProvenance,
)
from searchkernel.runtime.local import (
    LegacyLocalOrchestratorAdapter,
    LegacyQueryOrchestrator,
    LocalSearchSource,
)
from searchkernel.search.record_pipeline import (
    RecordSearchConfig,
    RecordSearchOutcome,
    RecordSearchResult,
)


class _Source:
    def __init__(self, source_kind: str, result: ScoredRef):
        self.source_kind = source_kind
        self.result = result
        self.calls: list[tuple[str, int, dict[str, Any] | None]] = []

    async def search(
        self, query: str, k: int, filters: dict[str, Any] | None = None
    ) -> Iterable[ScoredRef]:
        self.calls.append((query, k, filters))
        return [self.result]


class _Reranker:
    model_name = "test-reranker"

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        return [0.75 for _ in documents]


@pytest.mark.asyncio
async def test_search_kernel_delegates_fusion_and_returns_search_results():
    source = _Source(
        "memory",
        ScoredRef(
            source_id="memory-1",
            score=0.2,
            source_kind="memory",
            metadata={"text": "a memory"},
        ),
    )
    kernel = SearchKernel.build(
        {"reranker": _Reranker()},
        sources=[source],
    )

    assert isinstance(kernel, SearchAPI)
    results = await kernel.search_anything(
        "query",
        sources=["memory"],
        filters={"workspace": "personal"},
        k=1,
    )

    assert results == [
        SearchResult(
            record_id="memory-1",
            score=0.75,
            source_kind="memory",
            metadata={"text": "a memory", "source_score": 0.2},
        )
    ]
    assert source.calls == [("query", 1, {"workspace": "personal"})]


@pytest.mark.asyncio
async def test_search_kernel_builds_canonical_local_source_from_record_ports():
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    record = Record(
        source_kind="note",
        source_id="record-1",
        title="Record title",
        body="record body",
        created_at=timestamp,
        updated_at=timestamp,
    )

    class _KeywordStore:
        def search(
            self,
            query: str,
            k: int,
            filters: dict[str, Any] | None = None,
        ) -> list[RecordHit]:
            assert query == "query"
            assert filters == {"statuses": ["active"]}
            return [
                RecordHit(
                    RecordIdentity(
                        record.workspace_id,
                        record.source_kind,
                        record.source_id,
                    ),
                    1.0,
                )
            ][:k]

        def index(self, records: list[Record]) -> None:
            pass

    kernel = SearchKernel.build(
        record_hydrator=lambda identity: record,
        keyword_store=_KeywordStore(),
    )

    results = await kernel.search_anything("query", k=1)

    assert results[0].record_id == "record-1"
    assert results[0].source_kind == "note"
    assert results[0].metadata["text"] == "record body"


@pytest.mark.asyncio
async def test_search_kernel_wires_reranker_into_local_record_pipeline():
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    record = Record(
        source_kind="note",
        source_id="record-1",
        title="Low priority title",
        body="record body",
        created_at=timestamp,
        updated_at=timestamp,
    )

    class _KeywordStore:
        def search(self, query, k, filters=None):
            return [
                RecordHit(
                    RecordIdentity(None, "note", "record-1"),
                    1.0,
                )
            ]

    class _LocalReranker:
        model_name = "local-test-reranker"

        def __init__(self):
            self.documents = []

        def rerank(self, query, documents):
            assert query == "query"
            self.documents.append(documents)
            return [0.25]

    reranker = _LocalReranker()
    kernel = SearchKernel.build(
        record_hydrator=lambda identity: record,
        keyword_store=_KeywordStore(),
        reranker=reranker,
        search_config=RecordSearchConfig(rerank_budget=1),
    )

    results = await kernel.search_anything("query", k=1)

    assert results[0].score == pytest.approx(0.25)
    assert reranker.documents[0] == ["Low priority title\nrecord body"]


@pytest.mark.asyncio
async def test_search_kernel_builds_without_reranker():
    source = _Source(
        "memory",
        ScoredRef(
            source_id="memory-1",
            score=0.2,
            source_kind="memory",
            metadata={"text": "a memory"},
        ),
    )
    kernel = SearchKernel.build(sources=[source])

    results = await kernel.search_anything("query", k=1)

    assert results[0].record_id == "memory-1"
    assert results[0].score == pytest.approx(1 / 61)
    assert results[0].metadata == {"text": "a memory", "source_score": 0.2}


@pytest.mark.asyncio
async def test_local_search_source_exposes_canonical_record_identity_and_metadata():
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    record = Record(
        source_kind="note",
        source_id="record-1",
        title="Local result",
        body="local result",
        created_at=timestamp,
        updated_at=timestamp,
        metadata={"file_path": "notes.md"},
    )

    class _Orchestrator:
        def __init__(self):
            self.calls = []

        async def search(
            self,
            query: str,
            *,
            limit: int,
            filters: dict[str, Any] | None,
        ):
            self.calls.append((query, limit, filters))
            return RecordSearchOutcome(
                results=(
                    RecordSearchResult(
                        record=record,
                        score=0.9,
                        provenance=SearchResultProvenance(),
                    ),
                )
            )

    orchestrator = _Orchestrator()
    source = LocalSearchSource(orchestrator)  # type: ignore[arg-type]

    results = await source.search(
        "query",
        3,
        filters={"source_filter": ["note"]},
    )

    assert list(results) == [
        ScoredRef(
            source_id="record-1",
            score=0.9,
            source_kind="note",
            metadata={
                "file_path": "notes.md",
                "text": "local result",
                "title": "Local result",
                "uri": None,
                "storage_key": record.storage_key,
                "provenance": {"strategies": []},
            },
        )
    ]
    assert orchestrator.calls == [
        ("query", 3, {"source_kinds": ["note"]})
    ]


@pytest.mark.asyncio
async def test_legacy_local_adapter_is_explicit_and_preserves_chunk_metadata():
    class _LegacyOrchestrator:
        def __init__(self):
            self.calls = []

        async def query(
            self,
            query: str,
            *,
            top_k: int,
            top_n: int,
            source_filter: list[str] | None,
        ):
            self.calls.append((query, top_k, top_n, source_filter))
            return (
                [
                    ChunkResult(
                        chunk_id="chunk-1",
                        record_id="record-1",
                        score=0.9,
                        content="legacy result",
                        metadata={"file_path": "notes.md"},
                    )
                ],
                object(),
                object(),
            )

    legacy = _LegacyOrchestrator()
    source = LocalSearchSource(
        LegacyLocalOrchestratorAdapter(cast(LegacyQueryOrchestrator, legacy))
    )
    kernel = SearchKernel.build(sources=[source])
    assert isinstance(kernel, SearchKernel)

    results = list(
        await source.search("query", 3, filters={"source_filter": ["note"]})
    )

    assert results[0].source_id == "chunk-1"
    assert results[0].source_kind == "local"
    assert results[0].metadata["doc_id"] == "record-1"
    assert results[0].metadata["text"] == "legacy result"
    assert legacy.calls == [("query", 3, 3, ["note"])]
