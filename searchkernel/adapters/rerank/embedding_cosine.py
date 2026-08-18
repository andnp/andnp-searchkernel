"""Embedding-cosine reranker: an ADDITIVE Reranker implementation requiring no
new dependency. Any caller that already has an EmbeddingProvider wired gets a
reranker for free by reusing its existing embeddings, making this a natural
fast tier for CascadingReranker.
"""

from __future__ import annotations

from searchkernel.ports.embedding import EmbeddingProvider
from searchkernel.utils.similarity import cosine_similarity_lists


class EmbeddingCosineReranker:
    """Scores documents by cosine similarity between query and document embeddings."""

    def __init__(self, embedding_provider: EmbeddingProvider) -> None:
        self._embedding_provider = embedding_provider
        self.model_name = f"cosine({embedding_provider.model_name})"

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        query_vector = self._embedding_provider.embed_query(query)
        document_vectors = self._embedding_provider.embed(documents)
        return [
            (cosine_similarity_lists(query_vector, vector) + 1.0) / 2.0
            for vector in document_vectors
        ]
