from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import pytest

from searchkernel.search.orchestrator import SearchOrchestrator
from searchkernel.search.pipeline import SearchPipelineConfig


@dataclass
class _SearchTuning:
    semantic_weight: float = 1.0
    keyword_weight: float = 1.0
    min_confidence: float = 0.0
    max_chunks_per_doc: int = 0
    dedup_threshold: float = 0.85
    reranking_enabled: bool = False
    rerank_top_n: int = 10
    project_uplift_multiplier: float = 1.2


@dataclass
class _Indexing:
    documents_path: str = "/documents"


@dataclass
class _Config:
    search: _SearchTuning = field(default_factory=_SearchTuning)
    indexing: _Indexing = field(default_factory=_Indexing)
    detected_project: str | None = None


class _Vector:
    def __init__(self, results, chunks):
        self.results = results
        self.chunks = chunks
        self.calls = []

    def expand_query(self, query):
        return f"expanded:{query}"

    def search(self, query, top_k, excluded_files, docs_root):
        self.calls.append((query, top_k, excluded_files, docs_root))
        return self.results

    def get_chunk_by_id(self, chunk_id):
        return self.chunks.get(chunk_id)

    def get_chunk_ids_for_document(self, doc_id):
        return [chunk_id for chunk_id, data in self.chunks.items() if data["doc_id"] == doc_id]

    def get_parent_content(self, chunk_id):
        return None

    def get_embedding_for_chunk(self, chunk_id):
        return [1.0, 0.0]


class _Keyword:
    def __init__(self, results, chunks):
        self.results = results
        self.chunks = chunks
        self.calls = []

    def search(self, query, top_k, excluded_files, docs_root):
        self.calls.append((query, top_k, excluded_files, docs_root))
        return self.results

    def get_chunk_by_id(self, chunk_id):
        return self.chunks.get(chunk_id)


class _Graph:
    def rank_neighbors(self, seed_scores):
        return []

    def boost_by_community(self, doc_ids, seed_doc_ids, boost_factor):
        return {}

    def get_edges_from(self, source):
        return []


def _orchestrator(vector_results, keyword_results, chunks):
    vector = _Vector(vector_results, chunks)
    keyword = _Keyword(keyword_results, chunks)
    return (
        SearchOrchestrator(
            cast(Any, vector),
            cast(Any, keyword),
            cast(Any, _Graph()),
            cast(Any, _Config()),
            documents_path=Path("/documents"),
        ),
        vector,
        keyword,
    )


@pytest.mark.asyncio
async def test_query_retrieves_filters_hydrates_and_reports_strategy_counts():
    chunks = {
        "keep_chunk_0": {
            "chunk_id": "keep_chunk_0",
            "doc_id": "keep",
            "content": "kept",
            "file_path": "keep.md",
            "metadata": {"project_id": "alpha", "source_kind": "note"},
        },
        "drop_chunk_0": {
            "chunk_id": "drop_chunk_0",
            "doc_id": "drop",
            "content": "dropped",
            "file_path": "drop.md",
            "metadata": {"project_id": "beta", "source_kind": "note"},
        },
    }
    orchestrator, vector, keyword = _orchestrator(
        [
            {"chunk_id": "keep_chunk_0", "doc_id": "keep", "score": 0.9},
            {"chunk_id": "drop_chunk_0", "doc_id": "drop", "score": 0.8},
        ],
        [{"chunk_id": "keep_chunk_0", "doc_id": "keep", "score": 0.7}],
        chunks,
    )

    results, compression, strategy = await orchestrator.query(
        "query",
        top_k=4,
        top_n=2,
        excluded_files={"drop.md"},
        project_filter=["alpha"],
        source_filter=["note"],
        pipeline_config=SearchPipelineConfig(reranking_enabled=False),
    )

    assert [result.chunk_id for result in results] == ["keep_chunk_0"]
    assert results[0].content == "kept"
    assert compression.original_count >= 1
    assert strategy.vector_count == 2
    assert strategy.keyword_count == 1
    assert vector.calls[0][2] == {"drop.md"}
    assert keyword.calls[0][2] == {"drop.md"}


@pytest.mark.asyncio
async def test_query_returns_empty_for_blank_query_and_no_hits():
    orchestrator, vector, keyword = _orchestrator([], [], {})

    results, compression, strategy = await orchestrator.query("   ")
    assert results == []
    assert compression.original_count == 0
    assert strategy.vector_count is None
    assert strategy.keyword_count is None
    assert vector.calls == []
    assert keyword.calls == []

    results, _, strategy = await orchestrator.query(
        "nothing",
        pipeline_config=SearchPipelineConfig(reranking_enabled=False),
    )
    assert results == []
    assert strategy.vector_count == 0


@pytest.mark.asyncio
async def test_query_propagates_retrieval_timeout():
    orchestrator, vector, _keyword = _orchestrator([], [], {})

    def timeout(*args):
        raise TimeoutError("vector index timed out")

    vector.search = cast(Any, timeout)

    with pytest.raises(TimeoutError, match="vector index timed out"):
        await orchestrator.query(
            "query",
            pipeline_config=SearchPipelineConfig(reranking_enabled=False),
        )


@pytest.mark.asyncio
async def test_query_hydrates_missing_results_and_schedules_reindex():
    orchestrator, _vector, _keyword = _orchestrator(
        [{"chunk_id": "missing_chunk_0", "doc_id": "missing", "score": 0.9}],
        [],
        {},
    )

    results, _, _ = await orchestrator.query(
        "query",
        pipeline_config=SearchPipelineConfig(reranking_enabled=False),
    )

    assert len(results) == 1
    assert results[0].chunk_id == "missing_chunk_0"
    assert results[0].record_id == "missing"
    assert results[0].score == pytest.approx(0.5621765)
    assert results[0].content == ""
    assert results[0].metadata == {"header_path": "", "file_path": ""}
