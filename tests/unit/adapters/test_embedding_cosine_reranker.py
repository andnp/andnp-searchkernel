import random
from datetime import UTC, datetime

import pytest

from searchkernel.adapters.rerank.embedding_cosine import EmbeddingCosineReranker
from searchkernel.domain import Record
from searchkernel.ports.rerank import RecordReranker, Reranker
from searchkernel.utils.similarity import cosine_similarity_lists


def _record(source_id: str) -> Record:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    return Record(
        source_kind="test",
        source_id=source_id,
        title=source_id,
        body=f"body for {source_id}",
        created_at=timestamp,
        updated_at=timestamp,
    )


class _FakeEmbeddingProvider:
    model_name = "fake-embedder"
    dim = 2

    def __init__(self, query_vector: list[float], document_vectors: list[list[float]]) -> None:
        self._query_vector = query_vector
        self._document_vectors = document_vectors
        self.embed_calls: list[list[str]] = []

    def embed_query(self, text: str) -> list[float]:
        return self._query_vector

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls.append(list(texts))
        assert len(texts) == len(self._document_vectors)
        return self._document_vectors


class _FakeStoredVectorLookup:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors
        self.calls: list[tuple[list[str], str, int]] = []

    def get_many(
        self, records: list[Record], model_name: str, dim: int
    ) -> dict[str, list[float]]:
        keys = [record.storage_key for record in records]
        self.calls.append((keys, model_name, dim))
        return {key: self._vectors[key] for key in keys if key in self._vectors}


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


def test_implements_record_reranker_protocol() -> None:
    reranker = EmbeddingCosineReranker(_FakeEmbeddingProvider([1.0], [[1.0]]))
    assert isinstance(reranker, RecordReranker)


def test_rerank_records_without_lookup_embeds_every_record() -> None:
    provider = _FakeEmbeddingProvider(
        query_vector=[1.0, 0.0], document_vectors=[[1.0, 0.0], [-1.0, 0.0]]
    )
    reranker = EmbeddingCosineReranker(provider)
    records = [_record("a"), _record("b")]

    scores = reranker.rerank_records("query", records)

    assert scores == [1.0, 0.0]
    assert provider.embed_calls == [["a\nbody for a", "b\nbody for b"]]


def test_rerank_records_reuses_a_stored_vector_and_skips_embedding_it() -> None:
    records = [_record("a"), _record("b")]
    provider = _FakeEmbeddingProvider(query_vector=[1.0, 0.0], document_vectors=[[-1.0, 0.0]])
    lookup = _FakeStoredVectorLookup({records[0].storage_key: [1.0, 0.0]})
    reranker = EmbeddingCosineReranker(provider, stored_vectors=lookup)

    scores = reranker.rerank_records("query", records)

    assert scores == [1.0, 0.0]
    assert lookup.calls == [([records[0].storage_key, records[1].storage_key], "fake-embedder", 2)]
    # Only the record missing from the lookup was embedded.
    assert provider.embed_calls == [["b\nbody for b"]]


def test_rerank_records_embeds_all_when_lookup_has_nothing_valid() -> None:
    records = [_record("a"), _record("b")]
    provider = _FakeEmbeddingProvider(
        query_vector=[1.0, 0.0], document_vectors=[[1.0, 0.0], [-1.0, 0.0]]
    )
    lookup = _FakeStoredVectorLookup({})
    reranker = EmbeddingCosineReranker(provider, stored_vectors=lookup)

    scores = reranker.rerank_records("query", records)

    assert scores == [1.0, 0.0]
    assert provider.embed_calls == [["a\nbody for a", "b\nbody for b"]]


def test_rerank_records_empty_returns_empty_without_any_lookup_or_embed() -> None:
    provider = _FakeEmbeddingProvider(query_vector=[1.0], document_vectors=[])
    lookup = _FakeStoredVectorLookup({})
    reranker = EmbeddingCosineReranker(provider, stored_vectors=lookup)

    assert reranker.rerank_records("query", []) == []
    assert lookup.calls == []
    assert provider.embed_calls == []


def test_rerank_matches_scalar_cosine_similarity_lists_for_random_vectors() -> None:
    rng = random.Random(42)
    dim = 16
    query_vector = [rng.uniform(-1.0, 1.0) for _ in range(dim)]
    document_vectors = [[rng.uniform(-1.0, 1.0) for _ in range(dim)] for _ in range(20)]
    # Include a zero vector to exercise the zero-norm branch.
    document_vectors.append([0.0] * dim)

    provider = _FakeEmbeddingProvider(query_vector, document_vectors)
    reranker = EmbeddingCosineReranker(provider)

    scores = reranker.rerank("query", [f"doc-{i}" for i in range(len(document_vectors))])

    expected = [
        (cosine_similarity_lists(query_vector, vector) + 1.0) / 2.0 for vector in document_vectors
    ]
    for actual_score, expected_score in zip(scores, expected, strict=True):
        assert actual_score == pytest.approx(expected_score, abs=1e-9)
