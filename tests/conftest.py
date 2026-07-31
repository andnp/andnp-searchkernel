"""
Shared pytest fixtures for library tests.

Provides both ephemeral (tmp_path) and persistent fixtures for different
testing scenarios:

- Ephemeral fixtures (tmp_path): Fast, isolated, used by default in unit tests
- Persistent fixtures: Realistic storage, shared across tests in a session/module

Use persistent fixtures when:
- Testing index persistence/loading behavior
- Testing manifest checking across test runs
- Simulating realistic production scenarios
- Testing index size/performance with larger datasets

Use ephemeral fixtures (tmp_path) when:
- Testing core logic in isolation
- Fast test iteration is priority
- Each test needs complete isolation
"""

# MUST be set before any HuggingFace/sentence-transformers imports to suppress
# progress bars that would pollute JSON output in tests.
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TQDM_DISABLE"] = "1"

# The test suite must never touch the HuggingFace Hub network: real-model tests
# rely entirely on a pre-populated local cache (see scripts/download_test_models.py).
# This avoids Hub rate limiting when multiple pytest-xdist workers start at once.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# llama_index's HuggingFaceEmbedding caches to its own directory (not HF_HOME),
# and HF_HOME itself governs where transformers/sentence-transformers/
# huggingface_hub look up cached models (embedding, reranker, and the
# query-pipeline cross-encoder reranker). Both are resolved from the real HOME
# at process startup - before isolate_xdg_data_home (below) starts giving each
# test an isolated fake HOME. Pin both once, globally, here, rather than
# per-test in isolate_xdg_data_home: relying on that fixture re-deriving the
# "original" value from the CURRENT os.environ["HOME"] each test is fragile to
# fixture-ordering races (a session-scoped fixture can build a real subprocess
# before or after HOME gets faked for a given test, depending on execution
# order) - pinning here is immune to that entirely, since it never changes.
os.environ.setdefault(
    "LLAMA_INDEX_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".cache", "llama_index")
)
os.environ.setdefault("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))

from collections.abc import Generator
from pathlib import Path

import pytest
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from searchkernel.embeddings import (
    TEST_FAKE_EMBEDDINGS_ENV_VAR,
    DeterministicFakeEmbeddingModel,
)
from searchkernel.indices.graph import GraphStore
from searchkernel.indices.keyword import KeywordIndex
from searchkernel.indices.vector import VectorIndex
from searchkernel.storage.db import DatabaseManager

# ============================================================================
# Test Fixture Factories
# ============================================================================


def create_test_document(docs_dir: Path | str, doc_id: str, content: str):
    """Create a test document file."""
    doc_path = Path(docs_dir) / f"{doc_id}.md"
    doc_path.write_text(content)
    return str(doc_path)


# ============================================================================
# Fake Embedding Model Fixture
# ============================================================================


@pytest.fixture(scope="session")
def deterministic_fake_embedding_model() -> DeterministicFakeEmbeddingModel:
    """Session-scoped fake embedding model for deterministic, offline tests."""
    return DeterministicFakeEmbeddingModel()


@pytest.fixture(autouse=True)
def configure_embedding_mode_for_test(request, monkeypatch):
    """Default to fake embeddings; real-model tests use the pre-cached HF models.

    HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE are set globally at import time (see top
    of this file), so real-model tests never touch the network - they require
    scripts/download_test_models.py to have been run beforehand.
    """
    if request.node.get_closest_marker("real_embeddings"):
        monkeypatch.delenv(TEST_FAKE_EMBEDDINGS_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(TEST_FAKE_EMBEDDINGS_ENV_VAR, "1")


# ============================================================================
# Shared Embedding Model Fixture
# ============================================================================


@pytest.fixture(scope="session")
def shared_embedding_model():
    """Session-scoped embedding model shared across all tests.

    Requires the model to already be cached locally (see
    scripts/download_test_models.py) - HF_HUB_OFFLINE is set globally, so this
    never touches the network. Uses a filelock because HuggingFace's cache
    loading is not safe against concurrent first-access from multiple
    pytest-xdist worker processes, even when the model is fully cached.
    Pre-warms with a dummy embedding call to avoid first-call overhead
    (~1-2s) during actual tests.
    """
    import filelock

    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    lock_path = os.path.join(hf_home, ".model_load.lock")
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)

    try:
        with filelock.FileLock(lock_path, timeout=300):
            model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
            _ = model.get_text_embedding("warmup")
    except OSError as exc:
        raise RuntimeError(
            "BAAI/bge-small-en-v1.5 is not in the local HuggingFace cache and "
            "the test suite runs offline. Run "
            "`uv run python scripts/download_test_models.py` first."
        ) from exc

    return model


@pytest.fixture(scope="module")
def module_vector_index(shared_embedding_model):
    """
    Module-scoped VectorIndex with shared embedding model.

    Use this instead of creating VectorIndex() in function-scoped fixtures
    to avoid redundant model loading (2-4s overhead per load).

    Note: Module scope means tests share index state. Only use when tests
    don't mutate the index or when using tmp_path for document isolation.
    """
    return VectorIndex(embedding_model=shared_embedding_model)


@pytest.fixture(scope="module")
def module_indices(shared_embedding_model, tmp_path_factory):
    """
    Module-scoped indices for integration tests.

    Returns (vector, keyword, graph) tuple with shared embedding model.
    Avoids redundant model loading across tests in the same module.

    Note: Module scope means tests share index state. Ensure tests either:
    1. Use separate tmp_path directories for document isolation, OR
    2. Don't mutate index state, OR
    3. Explicitly clear indices between tests
    """
    vector = VectorIndex(embedding_model=shared_embedding_model)
    db_path = tmp_path_factory.mktemp("keyword_module") / "index.db"
    keyword = KeywordIndex(DatabaseManager(db_path))
    graph = GraphStore()
    return vector, keyword, graph


# ============================================================================
# Persistent Storage Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def persistent_storage_root(tmp_path_factory) -> Path:
    """
    Create session-scoped persistent storage directory.

    This directory persists for the entire test session, allowing
    tests to share data and verify persistence behavior.

    Returns path to persistent storage root directory.
    """
    return tmp_path_factory.mktemp("persistent_test_storage")


@pytest.fixture(scope="session")
def persistent_docs_path(persistent_storage_root: Path) -> Path:
    """
    Create session-scoped documents directory.

    Documents stored here persist across tests in the session.

    Returns path to persistent documents directory.
    """
    docs_path = persistent_storage_root / "documents"
    docs_path.mkdir(parents=True, exist_ok=True)
    return docs_path


@pytest.fixture(scope="session")
def persistent_index_path(persistent_storage_root: Path) -> Path:
    """
    Create session-scoped index directory.

    Indices stored here persist across tests in the session.

    Returns path to persistent index directory.
    """
    index_path = persistent_storage_root / "indices"
    index_path.mkdir(parents=True, exist_ok=True)
    return index_path


# ============================================================================
# Function-Scoped Persistent Fixtures with Cleanup
# ============================================================================


@pytest.fixture
def persistent_indices_isolated(
    shared_embedding_model, tmp_path
) -> Generator[tuple[VectorIndex, KeywordIndex, GraphStore]]:
    """
    Create function-scoped indices that can use persistent storage.

    Fresh indices for each test but can persist to/load from disk.
    Provides isolation while allowing persistence testing.

    Yields tuple of (vector, keyword, graph) indices.
    """
    vector = VectorIndex(embedding_model=shared_embedding_model)
    keyword = KeywordIndex(DatabaseManager(tmp_path / "index.db"))
    graph = GraphStore()
    yield vector, keyword, graph


# ============================================================================
# Cleanup Utilities
# ============================================================================


@pytest.fixture
def cleanup_persistent_indices(
    persistent_index_path: Path,
) -> Generator[None]:
    """
    Clean up persistent indices after test execution.

    Use this fixture when you need guaranteed cleanup of persistent
    storage after a test, even if using session-scoped paths.

    Example:
        def test_with_cleanup(
            persistent_indices_isolated,
            cleanup_persistent_indices
        ):
            # Test code here
            # Indices will be cleaned up after test
            pass
    """
    yield
    # Cleanup after test
    if persistent_index_path.exists():
        import shutil

        for item in persistent_index_path.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()


@pytest.fixture
def cleanup_persistent_docs(persistent_docs_path: Path) -> Generator[None]:
    """
    Clean up persistent documents after test execution.

    Use this fixture when you need guaranteed cleanup of persistent
    documents after a test.

    Example:
        def test_with_doc_cleanup(
            persistent_docs_path,
            cleanup_persistent_docs
        ):
            # Test code here
            # Documents will be cleaned up after test
            pass
    """
    yield
    # Cleanup after test
    if persistent_docs_path.exists():
        for item in persistent_docs_path.iterdir():
            if item.is_dir():
                import shutil

                shutil.rmtree(item)
            else:
                item.unlink()


# ============================================================================
# pytest-xdist hook to handle serial tests
# ============================================================================


def pytest_xdist_auto_num_workers(config):
    """Hook to configure pytest-xdist behavior for serial tests."""
    # Let pytest-xdist determine worker count automatically
    return


def pytest_collection_modifyitems(config, items):
    """Mark serial tests to run in the main process and tag real embedding tests."""
    for item in items:
        # Mark tests that use shared_embedding_model to use real embeddings
        if "shared_embedding_model" in item.fixturenames:
            fixture_info = getattr(item, "_fixtureinfo", None)
            fixture_defs = (
                fixture_info.name2fixturedefs.get("shared_embedding_model", [])
                if fixture_info is not None
                else []
            )
            resolved_fixture = fixture_defs[-1] if fixture_defs else None
            fixture_func = getattr(resolved_fixture, "func", None)
            if (
                fixture_func is not None
                and fixture_func.__module__ == "tests.conftest"
                and fixture_func.__name__ == "shared_embedding_model"
            ):
                item.add_marker(pytest.mark.real_embeddings)

        if "serial" in item.keywords:
            # Force serial tests to run in dist group 'serial'
            # This ensures they don't run in parallel with other tests
            item.add_marker(pytest.mark.xdist_group(name="serial"))
