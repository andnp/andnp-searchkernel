from collections.abc import Iterable
from typing import Any

import pytest

from searchkernel import SearchAPI, SearchKernel
from searchkernel.domain import ChunkResult, ScoredRef, SearchResult
from searchkernel.runtime.local import LocalSearchSource


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
async def test_local_search_source_preserves_chunk_identity_and_metadata():
    class _Orchestrator:
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
                        content="local result",
                        metadata={"file_path": "notes.md"},
                    )
                ],
                object(),
                object(),
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
            source_id="chunk-1",
            score=0.9,
            source_kind="local",
            metadata={
                "file_path": "notes.md",
                "text": "local result",
                "doc_id": "record-1",
            },
        )
    ]
    assert orchestrator.calls == [("query", 3, 3, ["note"])]
