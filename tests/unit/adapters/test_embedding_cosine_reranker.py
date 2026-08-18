from searchkernel.adapters.rerank.embedding_cosine import EmbeddingCosineReranker
from searchkernel.ports.rerank import Reranker


class _FakeEmbeddingProvider:
    model_name = "fake-embedder"

    def __init__(self, query_vector: list[float], document_vectors: list[list[float]]) -> None:
        self._query_vector = query_vector
        self._document_vectors = document_vectors

    def embed_query(self, text: str) -> list[float]:
        return self._query_vector

    def embed(self, texts: list[str]) -> list[list[float]]:
        assert len(texts) == len(self._document_vectors)
        return self._document_vectors


def test_implements_reranker_protocol() -> None:
    reranker = EmbeddingCosineReranker(_FakeEmbeddingProvider([1.0], [[1.0]]))
    assert isinstance(reranker, Reranker)


def test_model_name_identifies_the_wrapped_embedder() -> None:
    reranker = EmbeddingCosineReranker(_FakeEmbeddingProvider([1.0], [[1.0]]))
    assert reranker.model_name == "cosine(fake-embedder)"


def test_scores_are_rescaled_cosine_similarity() -> None:
    provider = _FakeEmbeddingProvider(
        query_vector=[1.0, 0.0],
        document_vectors=[[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]],
    )
    reranker = EmbeddingCosineReranker(provider)

    scores = reranker.rerank("query", ["aligned", "opposite", "orthogonal"])

    assert scores == [1.0, 0.0, 0.5]


def test_empty_documents_returns_empty_scores_without_embedding_calls() -> None:
    provider = _FakeEmbeddingProvider(query_vector=[1.0], document_vectors=[])
    reranker = EmbeddingCosineReranker(provider)

    assert reranker.rerank("query", []) == []
