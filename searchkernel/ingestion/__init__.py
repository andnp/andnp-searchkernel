"""Source-agnostic ingestion primitives."""

from searchkernel.ingestion.embedding import (
    EmbeddingBatchResult,
    EmbeddingInput,
    async_embed_and_upsert,
    async_embed_in_batches,
    embed_and_upsert,
    embed_in_batches,
)

__all__ = [
    "EmbeddingBatchResult",
    "EmbeddingInput",
    "async_embed_and_upsert",
    "async_embed_in_batches",
    "embed_and_upsert",
    "embed_in_batches",
]
