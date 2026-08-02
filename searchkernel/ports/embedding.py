"""Embedding ports for providers and source-owned embedding sinks."""

from collections.abc import Iterable, Iterator
from typing import Protocol, runtime_checkable

from searchkernel.domain import Vector


@runtime_checkable
class EmbeddingBatchProvider(Protocol):
    """Generates a batch of embeddings without imposing dimension policy."""

    @property
    def model_name(self) -> str:
        ...

    def embed(self, texts: list[str]) -> list[Vector]:
        """Return one embedding for each input text, in input order."""
        ...


@runtime_checkable
class StreamingEmbeddingProvider(EmbeddingBatchProvider, Protocol):
    """Optional provider seam for bounded, source-driven embedding."""

    def iter_embed_batches(
        self, texts: Iterable[str], batch_size: int
    ) -> Iterator[list[Vector]]:
        """Yield validated embedding batches without retaining prior vectors."""
        ...

@runtime_checkable
class EmbeddingSink(Protocol):
    """Persists one source embedding with source-owned write policy."""

    def upsert(
        self,
        *,
        source_kind: str,
        source_id: str,
        workspace_id: str | None,
        model_name: str,
        embedding: Vector,
        source_updated_at: str | None = None,
    ) -> bool:
        """Persist an embedding and report whether the write was accepted."""
        ...


@runtime_checkable
class EmbeddingProvider(EmbeddingBatchProvider, Protocol):
    """Embedding provider with an explicit, stable vector dimension."""

    dim: int


class AsyncEmbeddingProvider(Protocol):
    """Async query-embedding boundary used by record search."""

    model_name: str
    dim: int

    async def embed_query(self, text: str) -> Vector:
        ...
