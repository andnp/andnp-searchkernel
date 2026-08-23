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
from searchkernel.ports.federation import SearchDiagnostics
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
    "SearchDiagnostics": SearchDiagnostics,
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


def test_api_lazy_exports_resolve_all_supported_names() -> None:
    """Supported lazy names resolve to their canonical public objects.

    The facade should defer imports without changing object identity.
    """
    from searchkernel import chunking, embeddings

    assert api.TEST_FAKE_EMBEDDINGS_ENV_VAR == embeddings.TEST_FAKE_EMBEDDINGS_ENV_VAR
    assert api.TEST_FAKE_EMBEDDING_MODEL_NAME == (
        embeddings.TEST_FAKE_EMBEDDING_MODEL_NAME
    )
    assert api.should_use_test_fake_embeddings is embeddings.should_use_test_fake_embeddings
    assert api.ChunkingStrategy is chunking.ChunkingStrategy
    assert api.HeaderBasedChunker is chunking.HeaderBasedChunker
    assert api.get_chunker is chunking.get_chunker


def test_api_unknown_attribute_raises_descriptive_attribute_error() -> None:
    """Unknown facade names raise AttributeError with the requested name.

    Lazy lookup must not turn typos into unrelated import errors.
    """
    missing_name = "missing_export"
    with pytest.raises(AttributeError, match="missing_export"):
        getattr(api, missing_name)


def test_api_import_isolated_from_optional_chunk_dependencies() -> None:
    """The API can import and resolve core lazy names without Markdown extras.

    Optional chunking dependencies should remain isolated until chunking access.
    """
    script = """
import builtins

real_import = builtins.__import__

def reject_optional_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "tree_sitter" or name.startswith("tree_sitter."):
        raise ModuleNotFoundError("tree-sitter intentionally blocked")
    if name == "tree_sitter_markdown" or name.startswith("tree_sitter_markdown."):
        raise ModuleNotFoundError("tree-sitter-markdown intentionally blocked")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = reject_optional_import

from searchkernel import api
from searchkernel.embeddings import TEST_FAKE_EMBEDDINGS_ENV_VAR

assert api.TEST_FAKE_EMBEDDINGS_ENV_VAR == TEST_FAKE_EMBEDDINGS_ENV_VAR
"""
    subprocess.run([sys.executable, "-c", script], check=True)
