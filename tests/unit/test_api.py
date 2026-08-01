import pytest

from searchkernel import api, ports
from searchkernel.domain import models
from searchkernel.indexing import async_ingestion, semantic
from searchkernel.kernel import SearchKernel
from searchkernel.ports import (
    content_source,
    embedding,
    live_indices,
    retrieval,
    search,
    stores,
)
from searchkernel.search import orchestrator, query_plan
from searchkernel.storage import db
from searchkernel.utils import similarity

API_EXPORTS = {
    "AsyncIndexIngestor": async_ingestion.AsyncIndexIngestor,
    "ContentSource": content_source.ContentSource,
    "DatabaseManager": db.DatabaseManager,
    "EmbeddingProvider": embedding.EmbeddingProvider,
    "QueryPlan": query_plan.QueryPlan,
    "Record": models.Record,
    "SearchKernel": SearchKernel,
    "SearchOrchestrator": orchestrator.SearchOrchestrator,
    "SearchResult": models.SearchResult,
    "SemanticInput": semantic.SemanticInput,
    "cosine_similarity": similarity.cosine_similarity,
}

PORT_EXPORTS = {
    "ContentSource": content_source.ContentSource,
    "EmbeddingProvider": embedding.EmbeddingProvider,
    "GraphIndexPort": live_indices.GraphIndexPort,
    "SearchAPI": search.SearchAPI,
    "VectorStore": stores.VectorStore,
    "extract_retrieval_fields": retrieval.extract_retrieval_fields,
}


@pytest.mark.parametrize(("name", "expected"), API_EXPORTS.items())
def test_api_exports_required_app_symbols(name: str, expected: object) -> None:
    assert name in api.__all__
    assert getattr(api, name) is expected


@pytest.mark.parametrize(("name", "expected"), PORT_EXPORTS.items())
def test_ports_exports_required_app_symbols(name: str, expected: object) -> None:
    assert name in ports.__all__
    assert getattr(ports, name) is expected
