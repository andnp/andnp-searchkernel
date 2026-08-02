"""Ollama EmbeddingProvider adapter.

Calls a local Ollama daemon's HTTP API instead of loading model weights
in-process. Useful when multiple processes need the same embedding model
and should share one set of weights via the daemon rather than each
holding its own copy in RAM.

This is an ADDITIVE port implementation; no other adapters are modified.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from searchkernel.domain import Vector

if TYPE_CHECKING:
    import httpx


class OllamaEmbeddingProvider:
    """EmbeddingProvider backed by the Ollama HTTP API.

    Requires an Ollama daemon reachable at ``base_url`` with ``model_name``
    already pulled (``ollama pull <model_name>``).
    """

    def __init__(
        self,
        model_name: str,
        *,
        base_url: str = "http://localhost:11434",
        dim: int | None = None,
        timeout: float = 60.0,
    ):
        import httpx

        self.model_name = model_name
        self._base_url = base_url.rstrip("/")
        self._client: httpx.Client = httpx.Client(timeout=timeout)
        self.dim = dim if dim is not None else self._resolve_dim()

    def _resolve_dim(self) -> int:
        """Fetch the model's embedding dimension via /api/show.

        Ollama reports it as ``"<family>.embedding_length"`` inside
        ``model_info``; the family prefix varies per model architecture.
        """
        response = self._client.post(
            f"{self._base_url}/api/show", json={"model": self.model_name}
        )
        response.raise_for_status()
        model_info = response.json().get("model_info", {})
        for key, value in model_info.items():
            if key.endswith(".embedding_length"):
                return int(value)
        raise RuntimeError(
            f"could not determine embedding dimension for ollama model "
            f"'{self.model_name}' (no *.embedding_length in /api/show response)"
        )

    def embed(self, texts: list[str]) -> list[Vector]:
        """Return one embedding per input text, in input order."""
        response = self._client.post(
            f"{self._base_url}/api/embed",
            json={"model": self.model_name, "input": texts},
        )
        response.raise_for_status()
        body = response.json()
        embeddings = body.get("embeddings") if isinstance(body, dict) else None
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise RuntimeError(
                f"ollama returned {len(embeddings) if isinstance(embeddings, list) else 'an invalid number of'} "
                f"embeddings for {len(texts)} inputs"
            )
        for index, vector in enumerate(embeddings):
            if (
                not isinstance(vector, list)
                or len(vector) != self.dim
                or not all(
                    isinstance(value, (int, float)) and math.isfinite(float(value))
                    for value in vector
                )
            ):
                raise RuntimeError(
                    f"ollama returned invalid embedding {index}: expected finite vector dimension {self.dim}"
                )
        return embeddings
