"""Unit tests for the HuggingFace EmbeddingProvider adapter.

These load the real Qwen3-Embedding-0.6B model (~1.2GB on first run), so
they are marked slow. A session-scoped fixture shares the loaded model.
"""

import math

import numpy as np
import pytest

from searchkernel.adapters.embedding import HuggingFaceEmbeddingProvider
from searchkernel.ports.embedding import EmbeddingProvider

pytestmark = pytest.mark.slow


def _l2_norm(vec: list[float]) -> float:
    return math.sqrt(sum(x * x for x in vec))


def _stub_encode(
    monkeypatch: pytest.MonkeyPatch,
    provider: HuggingFaceEmbeddingProvider,
    embeddings: np.ndarray,
) -> None:
    def encode(texts: list[str], **kwargs: object) -> np.ndarray:
        del kwargs
        return embeddings[: len(texts)]

    monkeypatch.setattr(provider._model, "encode", encode)


@pytest.fixture(scope="module")
def provider() -> HuggingFaceEmbeddingProvider:
    return HuggingFaceEmbeddingProvider()


def test_satisfies_port(provider: HuggingFaceEmbeddingProvider):
    assert isinstance(provider, EmbeddingProvider)


def test_native_dim_is_1024(provider: HuggingFaceEmbeddingProvider):
    assert provider.dim == 1024


def test_embed_returns_normalized_vectors_of_dim(
    provider: HuggingFaceEmbeddingProvider,
):
    texts = ["Cats are small carnivorous mammals.", "PostgreSQL is a database."]
    vecs = provider.embed(texts)

    assert len(vecs) == len(texts)
    for vec in vecs:
        assert len(vec) == provider.dim
        assert _l2_norm(vec) == pytest.approx(1.0, abs=1e-2)


def test_embed_query_differs_from_embed(provider: HuggingFaceEmbeddingProvider):
    text = "How do neural networks learn?"

    doc_vec = provider.embed([text])[0]
    query_vec = provider.embed_query(text)

    assert len(query_vec) == provider.dim
    assert _l2_norm(query_vec) == pytest.approx(1.0, abs=1e-2)
    # The query instruction prompt must actually change the embedding.
    assert query_vec != doc_vec
    max_delta = max(abs(a - b) for a, b in zip(query_vec, doc_vec))
    assert max_delta > 1e-4


def test_truncate_dim_yields_truncated_normalized_vectors():
    provider = HuggingFaceEmbeddingProvider(truncate_dim=512)

    assert provider.dim == 512
    vec = provider.embed(["Matryoshka representation learning."])[0]
    assert len(vec) == 512
    assert _l2_norm(vec) == pytest.approx(1.0, abs=1e-2)


def test_numpy_batch_returns_public_vector_type(
    provider: HuggingFaceEmbeddingProvider, monkeypatch: pytest.MonkeyPatch
):
    """A valid NumPy batch is validated once and returned as vector lists."""
    embeddings = np.array([[1.0, 2.0], [3.0, 4.0]])
    monkeypatch.setattr(provider, "dim", 2)
    _stub_encode(monkeypatch, provider, embeddings)

    result = provider.embed(["one", "two"])

    assert result == [[1.0, 2.0], [3.0, 4.0]]
    assert all(isinstance(vector, list) for vector in result)


def test_numpy_empty_batch_returns_empty_list(
    provider: HuggingFaceEmbeddingProvider, monkeypatch: pytest.MonkeyPatch
):
    """An empty input remains an empty public result without model rows."""
    _stub_encode(monkeypatch, provider, np.empty(0))

    assert provider.embed([]) == []


def test_numpy_malformed_batch_preserves_dimension_error(
    provider: HuggingFaceEmbeddingProvider, monkeypatch: pytest.MonkeyPatch
):
    """A NumPy vector with the wrong dimension keeps the public error."""
    _stub_encode(monkeypatch, provider, np.ones((1, provider.dim + 1)))

    with pytest.raises(
        RuntimeError,
        match=f"invalid vector 0: expected dimension {provider.dim}",
    ):
        provider.embed(["one"])


def test_numpy_nonfinite_batch_preserves_finite_error(
    provider: HuggingFaceEmbeddingProvider, monkeypatch: pytest.MonkeyPatch
):
    """A non-finite NumPy value identifies its containing vector."""
    embeddings = np.ones((2, provider.dim))
    embeddings[0, 0] = np.nan
    _stub_encode(monkeypatch, provider, embeddings)

    with pytest.raises(RuntimeError, match="non-finite vector 0"):
        provider.embed(["one", "two"])


def test_query_batch_matches_single_query_result(
    provider: HuggingFaceEmbeddingProvider, monkeypatch: pytest.MonkeyPatch
):
    """Single-query and batch-query calls return the same embedding."""
    _stub_encode(monkeypatch, provider, np.ones((1, provider.dim)))

    assert provider.embed_query("one") == provider.embed_queries(["one"])[0]
