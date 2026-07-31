from pathlib import Path
from typing import Any, cast

import pytest

from searchkernel.domain import ChunkResult
from searchkernel.search.chunk_hydrator import ChunkHydrator
from searchkernel.search.pipeline import SearchPipeline, SearchPipelineConfig
from searchkernel.search.query_execution import QueryExecutionContext
from searchkernel.search.tag_expansion import expand_query_with_tags


class _Vector:
    def __init__(self, chunks=None, embeddings=None, parents=None):
        self.chunks = chunks or {}
        self.embeddings = embeddings or {}
        self.parents = parents or {}
        self.calls: list[tuple[str, str]] = []

    def get_chunk_by_id(self, chunk_id):
        self.calls.append(("chunk", chunk_id))
        return self.chunks.get(chunk_id)

    def get_embedding_for_chunk(self, chunk_id):
        self.calls.append(("embedding", chunk_id))
        return self.embeddings.get(chunk_id)

    def get_parent_content(self, chunk_id):
        self.calls.append(("parent", chunk_id))
        return self.parents.get(chunk_id)

    def get_chunk_ids_for_document(self, doc_id):
        return [
            chunk_id
            for chunk_id, chunk in self.chunks.items()
            if chunk.get("doc_id") == doc_id
        ]


class _Keyword:
    def __init__(self, chunks=None):
        self.chunks = chunks or {}
        self.calls: list[str] = []

    def get_chunk_by_id(self, chunk_id):
        self.calls.append(chunk_id)
        return self.chunks.get(chunk_id)


def test_chunk_hydrator_enriches_incomplete_vector_data_from_keyword():
    vector = _Vector(
        {
            "doc_chunk_0": {
                "chunk_id": "doc_chunk_0",
                "doc_id": "doc",
                "content": "",
                "metadata": {"project_id": "p1"},
            }
        }
    )
    keyword = _Keyword(
        {
            "doc_chunk_0": {
                "chunk_id": "doc_chunk_0",
                "doc_id": "doc",
                "content": "keyword content",
                "headers": "Heading",
                "source_file": "doc.md",
                "title": "Title",
                "tags": "one,two",
            }
        }
    )
    queued: list[tuple[list[str], str]] = []
    hydrator = ChunkHydrator(
        cast(Any, vector),
        cast(Any, keyword),
        Path("/documents"),
        lambda ids, reason: queued.append((ids, reason)),
    )

    result = hydrator.hydrate_chunk_result("doc_chunk_0", 0.8)

    assert result == ChunkResult(
        chunk_id="doc_chunk_0",
        record_id="doc",
        score=0.8,
        content="keyword content",
        metadata={
            "project_id": "p1",
            "title": "Title",
            "tags": ["one", "two"],
            "header_path": "Heading",
            "file_path": "doc.md",
        },
    )
    assert queued and queued[0][0] == ["doc_chunk_0"]
    assert keyword.calls == ["doc_chunk_0"]


def test_chunk_hydrator_uses_parent_content_from_query_context():
    vector = _Vector(
        {
            "doc_chunk_1": {
                "chunk_id": "doc_chunk_1",
                "doc_id": "doc",
                "content": "child",
                "file_path": "doc.md",
                "metadata": {"parent_chunk_id": "doc_parent"},
            }
        },
        parents={"doc_parent": "parent body"},
    )
    keyword = _Keyword()
    context = QueryExecutionContext(
        cast(Any, vector), cast(Any, keyword), cast(Any, None)
    )
    hydrator = ChunkHydrator(cast(Any, vector), cast(Any, keyword), Path("/documents"))

    result = hydrator.hydrate_chunk_result("doc_chunk_1", 0.4, query_context=context)

    assert result is not None
    assert result.parent_chunk_id == "doc_parent"
    assert result.parent_content == "parent body"
    assert context.stats.parent_lookups == 1


def test_query_execution_context_caches_lookups_and_tracks_hits():
    vector = _Vector(
        {"c": {"chunk_id": "c", "doc_id": "d", "content": "body"}},
        embeddings={"c": [1.0]},
        parents={"p": "parent"},
    )
    keyword = _Keyword()
    hydrator = ChunkHydrator(cast(Any, vector), cast(Any, keyword), Path("/documents"))
    context = QueryExecutionContext(cast(Any, vector), cast(Any, keyword), hydrator)

    assert context.get_vector_chunk("c") == context.get_vector_chunk("c")
    assert context.get_chunk_embedding("c") == context.get_chunk_embedding("c")
    assert context.get_parent_content("p") == context.get_parent_content("p")
    assert context.get_chunk_content("c") == context.get_chunk_content("c")

    stats = context.stats
    assert stats.metadata_lookups == 2
    assert stats.metadata_cache_hits == 2
    assert stats.embedding_fetches == 1
    assert stats.embedding_cache_hits == 1
    assert stats.parent_lookups == 2
    assert stats.parent_cache_hits == 1
    assert stats.content_lookups == 1
    assert stats.content_cache_hits == 1


def test_search_pipeline_applies_filters_document_limit_and_clamps_scores():
    pipeline = SearchPipeline(
        SearchPipelineConfig(
            min_confidence=0.5,
            max_chunks_per_doc=1,
            reranking_enabled=False,
        )
    )
    results, stats = pipeline.process(
        [
            ("doc_chunk_0", 1.2),
            ("doc_chunk_1", 0.8),
            ("other_chunk_0", 0.4),
        ],
        lambda _chunk_id: [1.0],
        lambda chunk_id: chunk_id,
        "query",
        top_n=5,
    )

    assert results == [("doc_chunk_0", 1.0)]
    assert stats.original_count == 3
    assert stats.after_threshold == 2
    assert stats.after_doc_limit == 1


def test_search_pipeline_dispatches_reranking_lazily(monkeypatch):
    calls = []

    class _Reranker:
        def rerank(self, query, results, get_content, top_n):
            calls.append((query, results, get_content("a"), top_n))
            return [("a", 0.6)]

    monkeypatch.setattr("searchkernel.search.pipeline.ReRanker", lambda model_name: _Reranker())
    pipeline = SearchPipeline(SearchPipelineConfig(reranking_enabled=True))

    results, _ = pipeline.process(
        [("a", 0.2)],
        lambda _chunk_id: [1.0],
        lambda _chunk_id: "content",
        "find a",
        top_n=1,
    )

    assert results == [("a", 0.6)]
    assert calls == [("find a", [("a", 0.2)], "content", 10)]


def test_search_pipeline_empty_results_have_zeroed_compression_stats():
    results, stats = SearchPipeline(SearchPipelineConfig()).process(
        [], lambda _chunk_id: None, lambda _chunk_id: None, "query"
    )

    assert results == []
    assert stats.original_count == 0
    assert stats.after_dedup == 0


class _Graph:
    def __init__(self, edges):
        self.edges = edges

    def get_edges_from(self, node):
        return self.edges.get(node, [])


class _TagVector:
    def __init__(self, chunks, documents):
        self.chunks = chunks
        self.documents = documents

    def get_chunk_by_id(self, chunk_id):
        return self.chunks.get(chunk_id)

    def get_chunk_ids_for_document(self, doc_id):
        return self.documents.get(doc_id, [])


def test_tag_expansion_finds_related_tag_chunks_and_deduplicates():
    vector = _TagVector(
        {
            "seed_chunk_0": {"metadata": {"tags": ["python"]}},
            "new_chunk_0": {"metadata": {}},
        },
        {"new-doc": ["new_chunk_0"]},
    )
    graph = _Graph(
        {
            "tag:python": [
                {"target": "tag:programming", "edge_type": "RELATED_TO"}
            ],
            "tag:programming": [{"target": "new-doc"}],
        }
    )

    initial = [{"chunk_id": "seed_chunk_0", "doc_id": "seed", "score": 0.9}]
    expanded = expand_query_with_tags(
        cast(Any, initial), cast(Any, graph), cast(Any, vector), top_k=5
    )

    assert [result["chunk_id"] for result in expanded] == [
        "seed_chunk_0",
        "new_chunk_0",
    ]
    assert expanded[-1].get("score") == 0.5


def test_tag_expansion_skips_missing_tags_and_non_related_edges():
    vector = _TagVector({"seed_chunk_0": {"metadata": "invalid"}}, {})
    graph = _Graph({"tag:python": [{"target": "tag:other", "edge_type": "USES"}]})

    initial = [{"chunk_id": "seed_chunk_0", "doc_id": "seed", "score": 0.9}]

    assert expand_query_with_tags(
        cast(Any, initial), cast(Any, graph), cast(Any, vector)
    ) == initial


@pytest.mark.asyncio
async def test_query_execution_context_hydrates_missing_content_as_none():
    vector = _Vector()
    keyword = _Keyword()
    hydrator = ChunkHydrator(cast(Any, vector), cast(Any, keyword), Path("/documents"))
    context = QueryExecutionContext(cast(Any, vector), cast(Any, keyword), hydrator)

    assert context.get_chunk_content("missing") is None
    assert context.get_chunk_content("missing") is None
    assert context.stats.content_lookups == 1
    assert context.stats.content_cache_hits == 1
