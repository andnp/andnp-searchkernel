"""Ollama EmbeddingProvider adapter.

Calls a local Ollama daemon's HTTP API instead of loading model weights
in-process. Useful when multiple processes need the same embedding model
and should share one set of weights via the daemon rather than each
holding its own copy in RAM.

This is an ADDITIVE port implementation; no other adapters are modified.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Self

from searchkernel.domain import Vector

if TYPE_CHECKING:
    import httpx


class OllamaEmbeddingProvider:
    """EmbeddingProvider backed by the Ollama HTTP API.

    Requires an Ollama daemon reachable at ``base_url``. Missing models are
    pulled automatically unless ``auto_pull`` is disabled.

    ``embed_query`` defaults to symmetric (no prefix) because this adapter
    serves arbitrary Ollama models and cannot know which ones are
    asymmetric. Pass ``query_prefix`` for models that need one.
    """

    def __init__(
        self,
        model_name: str,
        *,
        base_url: str = "http://localhost:11434",
        dim: int | None = None,
        timeout: float = 60.0,
        auto_pull: bool = True,
        pull_timeout: float = 600.0,
        query_prefix: str = "",
    ):
        import httpx

        self.model_name = model_name
        self._base_url = base_url.rstrip("/")
        self._client: httpx.Client = httpx.Client(timeout=timeout)
        self._auto_pull = auto_pull
        self._pull_timeout = pull_timeout
        self._query_prefix = query_prefix
        try:
            self.dim = dim if dim is not None else self._resolve_dim()
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        """Close the persistent HTTP client."""
        self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def _resolve_dim(self) -> int:
        """Fetch the model's embedding dimension via /api/show.

        Ollama reports it as ``"<family>.embedding_length"`` inside
        ``model_info``; the family prefix varies per model architecture.
        """
        response = self._post("/api/show", json={"model": self.model_name})
        model_info = response.json().get("model_info", {})
        for key, value in model_info.items():
            if key.endswith(".embedding_length"):
                return int(value)
        raise RuntimeError(
            f"could not determine embedding dimension for ollama model "
            f"'{self.model_name}' (no *.embedding_length in /api/show response)"
        )

    def _post(self, path: str, *, json: dict[str, object]) -> httpx.Response:
        import httpx

        try:
            response = self._client.post(f"{self._base_url}{path}", json=json)
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            if not self._auto_pull or error.response.status_code != 404:
                raise
            self._pull_model()
            response = self._client.post(f"{self._base_url}{path}", json=json)
            response.raise_for_status()
        return response

    def _pull_model(self) -> None:
        response = self._client.post(
            f"{self._base_url}/api/pull",
            json={"name": self.model_name, "stream": False},
            timeout=self._pull_timeout,
        )
        response.raise_for_status()

    def embed_query(self, text: str) -> Vector:
        """Embed a single QUERY, applying ``query_prefix`` if configured.

        Ollama serves arbitrary models, so this adapter cannot know whether
        the configured model is asymmetric (e.g. ``nomic-embed-text`` wants
        ``search_query:`` / ``search_document:`` prefixes) or symmetric.
        Defaulting to no prefix preserves today's behavior for every
        existing caller; ``query_prefix`` gives callers who know their
        model's convention the seam to apply it correctly.
        """
        text_to_embed = f"{self._query_prefix}{text}" if self._query_prefix else text
        return self.embed([text_to_embed])[0]

    def embed(self, texts: list[str]) -> list[Vector]:
        """Return one embedding per input text, in input order."""
        response = self._post(
            "/api/embed",
            json={"model": self.model_name, "input": texts},
        )
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
