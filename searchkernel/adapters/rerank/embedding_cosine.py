"""Embedding-cosine reranker: an ADDITIVE Reranker implementation requiring no
new dependency. Any caller that already has an EmbeddingProvider wired gets a
reranker for free by reusing its existing embeddings, making this a natural
fast tier for CascadingReranker.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from searchkernel.domain import Record, Vector
from searchkernel.ports.embedding import EmbeddingProvider


@runtime_checkable
class StoredVectorLookup(Protocol):
    """Fetches already-computed embeddings, keyed by record, when still valid."""

    def get_many(
        self, records: list[Record], model_name: str, dim: int
    ) -> dict[str, Vector]:
        """Return a storage_key -> vector mapping for records with a valid stored vector."""
        ...


def _record_text(record: Record) -> str:
    return f"{record.title}\n{record.indexed_text or record.body}".strip()


def _batched_cosine_scores(query_vector: Vector, vectors: list[Vector]) -> list[float]:
    query_array = np.asarray(query_vector, dtype=np.float64)
    vectors_array = np.asarray(vectors, dtype=np.float64)

    query_norm = np.linalg.norm(query_array)
    vector_norms = np.linalg.norm(vectors_array, axis=1)

    dot_products = vectors_array @ query_array
    denominators = vector_norms * query_norm
    valid = denominators != 0

    cosine = np.zeros_like(dot_products)
    cosine[valid] = dot_products[valid] / denominators[valid]

    return [float(score) for score in (cosine + 1.0) / 2.0]


class EmbeddingCosineReranker:
    """Scores documents by cosine similarity between query and document embeddings."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        *,
        stored_vectors: StoredVectorLookup | None = None,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._stored_vectors = stored_vectors
        self.model_name = f"cosine({embedding_provider.model_name})"

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        query_vector = self._embedding_provider.embed_query(query)
        document_vectors = self._embedding_provider.embed(documents)
        return _batched_cosine_scores(query_vector, document_vectors)

    def rerank_records(self, query: str, records: list[Record]) -> list[float]:
        if not records:
            return []
        query_vector = self._embedding_provider.embed_query(query)
        vectors = self._vectors_for(records)
        return _batched_cosine_scores(query_vector, vectors)

    def _vectors_for(self, records: list[Record]) -> list[Vector]:
        stored: dict[str, Vector] = {}
        if self._stored_vectors is not None:
            stored = self._stored_vectors.get_many(
                records, self._embedding_provider.model_name, self._embedding_provider.dim
            )
        missing = [record for record in records if record.storage_key not in stored]
        fresh: dict[str, Vector] = {}
        if missing:
            embedded = self._embedding_provider.embed([_record_text(record) for record in missing])
            fresh = dict(
                zip((record.storage_key for record in missing), embedded, strict=True)
            )
        return [
            stored[record.storage_key]
            if record.storage_key in stored
            else fresh[record.storage_key]
            for record in records
        ]
