"""HuggingFace / sentence-transformers EmbeddingProvider adapter.

In-process embedding via ``sentence_transformers.SentenceTransformer``.
Defaults to Qwen3-Embedding-0.6B. No Ollama, no external service.

This is an ADDITIVE port implementation. The live embedding path
(``searchkernel/indices/vector.py``, BAAI/bge-small-en-v1.5) is untouched.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from searchkernel.domain import Vector

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

# Documented Qwen3-Embedding query instruction, used only if the loaded model
# does not expose a "query" prompt via sentence-transformers.
_QWEN3_QUERY_INSTRUCTION = (
    "Instruct: Given a web search query, retrieve relevant passages\nQuery: "
)


class HuggingFaceEmbeddingProvider:
    """EmbeddingProvider backed by a sentence-transformers model.

    Qwen3-Embedding is asymmetric: queries take an instruction prompt,
    documents do not. ``embed`` embeds DOCUMENTS (no prompt); ``embed_query``
    applies the query instruction. The EmbeddingProvider port itself has no
    query variant yet -- that asymmetry lives here until W4a lifts it into
    the port contract.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        *,
        truncate_dim: int | None = None,
        device: str | None = None,
        batch_size: int = 32,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._truncate_dim = truncate_dim
        self._batch_size = batch_size
        # Matryoshka (MRL): passing truncate_dim makes the model emit
        # truncated + re-normalized vectors directly.
        self._model: SentenceTransformer = SentenceTransformer(
            model_name, truncate_dim=truncate_dim, device=device
        )
        native_dim = self._model.get_embedding_dimension()
        if native_dim is None or native_dim < 1:
            raise RuntimeError("embedding model did not report a positive dimension")
        self.dim: int = truncate_dim if truncate_dim is not None else int(native_dim)
        # Whether the model ships a named "query" prompt we can reference.
        self._has_query_prompt = "query" in getattr(self._model, "prompts", {})

    @property
    def encoder_namespace(self) -> str:
        prompt = "named-query" if self._has_query_prompt else _QWEN3_QUERY_INSTRUCTION
        return (
            f"{self.model_name}|dim={self.dim}|truncate_dim={self._truncate_dim}"
            f"|normalize=l2|query_prompt={prompt}"
        )

    def embed(self, texts: list[str]) -> list[Vector]:
        """Embed DOCUMENTS (no instruction prompt), L2-normalized."""
        embeddings = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return self._validate_embeddings(embeddings, len(texts))

    def embed_query(self, text: str) -> Vector:
        """Embed a single QUERY with the Qwen3 instruction prompt applied."""
        return self.embed_queries([text])[0]

    def embed_queries(self, texts: list[str]) -> list[Vector]:
        """Embed QUERIES with the Qwen3 instruction prompt applied."""
        if self._has_query_prompt:
            embeddings = self._model.encode(
                texts,
                prompt_name="query",
                batch_size=self._batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        else:
            embeddings = self._model.encode(
                texts,
                prompt=_QWEN3_QUERY_INSTRUCTION,
                batch_size=self._batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        return self._validate_embeddings(embeddings, len(texts))

    def _validate_embeddings(
        self, embeddings: object, expected_count: int
    ) -> list[Vector]:
        if isinstance(embeddings, np.ndarray):
            if expected_count == 0 and embeddings.size == 0:
                return []
            if embeddings.ndim != 2:
                count = len(embeddings) if embeddings.ndim > 0 else "an invalid number of"
                raise RuntimeError(
                    f"embedding model returned {count} vectors for {expected_count} inputs"
                )
            if len(embeddings) != expected_count:
                raise RuntimeError(
                    f"embedding model returned {len(embeddings)} vectors for {expected_count} inputs"
                )
            if embeddings.shape[1] != self.dim:
                raise RuntimeError(
                    f"embedding model returned invalid vector 0: expected dimension {self.dim}"
                )
            if embeddings.dtype.kind not in "biuf":
                raise RuntimeError(
                    "embedding model returned non-finite vector 0"
                )
            finite_rows = np.isfinite(embeddings).all(axis=1)
            if not finite_rows.all():
                invalid_index = int(np.flatnonzero(~finite_rows)[0])
                raise RuntimeError(
                    f"embedding model returned non-finite vector {invalid_index}"
                )
            return embeddings.tolist()
        if not isinstance(embeddings, list) or len(embeddings) != expected_count:
            raise RuntimeError(
                f"embedding model returned {len(embeddings) if isinstance(embeddings, list) else 'an invalid number of'} "
                f"vectors for {expected_count} inputs"
            )
        for index, vector in enumerate(embeddings):
            if not isinstance(vector, list) or len(vector) != self.dim:
                raise RuntimeError(
                    f"embedding model returned invalid vector {index}: expected dimension {self.dim}"
                )
            if not all(
                isinstance(value, (int, float)) and math.isfinite(float(value))
                for value in vector
            ):
                raise RuntimeError(f"embedding model returned non-finite vector {index}")
        return embeddings
