from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

import pytest

from searchkernel import SearchAPI, SearchKernel
from searchkernel.domain import Record, ScoredRef, SearchResult, SearchResultProvenance
from searchkernel.runtime.local import LocalSearchSource
from searchkernel.search.record_pipeline import RecordSearchOutcome, RecordSearchResult


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
