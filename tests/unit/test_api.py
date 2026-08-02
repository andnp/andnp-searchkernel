import subprocess
import sys

import pytest

from searchkernel import api, ports
from searchkernel.adapters.embedding import HuggingFaceEmbeddingProvider
from searchkernel.domain import models
from searchkernel.indexing import semantic
from searchkernel.indexing.runtime_readiness import SearchAvailability
from searchkernel.kernel import SearchKernel
from searchkernel.ports import (
    content_source,
    embedding,
    retrieval,
    search,
    search_results,
    stores,
)
from searchkernel.ports.search_results import RecordSearchOutcome, RecordSearchResult
from searchkernel.runtime import reindex
from searchkernel.search import orchestrator, query_plan
from searchkernel.storage import db
from searchkernel.utils import similarity

API_EXPORTS = {
    "ContentSource": content_source.ContentSource,
    "DatabaseManager": db.DatabaseManager,
    "EmbeddingProvider": embedding.EmbeddingProvider,
    "HuggingFaceEmbeddingProvider": HuggingFaceEmbeddingProvider,
    "QueryPlan": query_plan.QueryPlan,
    "Record": models.Record,
    "RecordHit": models.RecordHit,
    "RecordIdentity": models.RecordIdentity,
    "ReindexError": reindex.ReindexError,
    "ReindexProgress": reindex.ReindexProgress,
    "ReindexRoutine": reindex.ReindexRoutine,
    "SearchKernel": SearchKernel,
    "SearchAvailability": SearchAvailability,
    "SearchOrchestrator": orchestrator.SearchOrchestrator,
    "SearchFilters": models.SearchFilters,
    "SearchResultProvenance": models.SearchResultProvenance,
    "RecordSearchOutcome": RecordSearchOutcome,
    "RecordSearchResult": RecordSearchResult,
    "SemanticInput": semantic.SemanticInput,
    "cosine_similarity": similarity.cosine_similarity,
}

PORT_EXPORTS = {
    "ContentSource": content_source.ContentSource,
    "EmbeddingProvider": embedding.EmbeddingProvider,
    "RecordHit": models.RecordHit,
    "SearchAPI": search.SearchAPI,
    "RecordSearchOutcome": search_results.RecordSearchOutcome,
    "RecordSearchResult": search_results.RecordSearchResult,
    "SearchFilters": models.SearchFilters,
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


def test_api_import_does_not_require_markdown_dependencies() -> None:
    script = """
import builtins

real_import = builtins.__import__

def reject_markdown_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "tree_sitter" or name.startswith("tree_sitter."):
        raise ModuleNotFoundError("tree-sitter intentionally blocked")
    if name == "tree_sitter_markdown" or name.startswith("tree_sitter_markdown."):
        raise ModuleNotFoundError("tree-sitter-markdown intentionally blocked")
    if name == "llama_index" or name.startswith("llama_index."):
        raise ModuleNotFoundError("llama-index intentionally blocked")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = reject_markdown_import

from searchkernel.api import SearchOrchestrator

assert SearchOrchestrator is not None
"""
    subprocess.run([sys.executable, "-c", script], check=True)
