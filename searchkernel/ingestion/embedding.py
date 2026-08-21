"""Batch embedding mechanics shared by source-owned ingestion pipelines."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable, Iterator
from dataclasses import dataclass
from itertools import islice

from searchkernel.domain import Vector
from searchkernel.ports.embedding import (
    EmbeddingBatchProvider,
    EmbeddingBatchSink,
    EmbeddingSink,
    EmbeddingWrite,
)


@dataclass(frozen=True, slots=True)
class EmbeddingInput:
    """Source data needed to generate and persist one embedding."""

    source_kind: str
    source_id: str
    text: str
    workspace_id: str | None = None
    source_updated_at: str | None = None


@dataclass(frozen=True, slots=True)
class EmbeddingBatchResult:
    """Counts from one complete embedding/upsert operation."""

    attempted: int
    stored: int
    rejected: int
    batches: int


def embed_and_upsert(
    inputs: list[EmbeddingInput],
    *,
    provider: EmbeddingBatchProvider,
    sink: EmbeddingSink | EmbeddingBatchSink,
    batch_size: int,
) -> EmbeddingBatchResult:
    """Embed inputs in bounded batches and persist each result.

    Every provider response is validated before any writes occur, preventing
    silent truncation when an adapter returns the wrong count.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if not inputs:
        return EmbeddingBatchResult(attempted=0, stored=0, rejected=0, batches=0)

    embedded_batches = iter_embed_batches(
        (item.text for item in inputs),
        provider=provider,
        batch_size=batch_size,
    )
    stored = 0
    rejected = 0
    batches = 0
    for input_batch, embedding_batch in zip(
        _batches(inputs, batch_size), embedded_batches, strict=True
    ):
        writes = [
            EmbeddingWrite(
                source_kind=item.source_kind,
                source_id=item.source_id,
                workspace_id=item.workspace_id,
                model_name=provider.model_name,
                embedding=embedding,
                source_updated_at=item.source_updated_at,
            )
            for item, embedding in zip(input_batch, embedding_batch, strict=True)
        ]
        if isinstance(sink, EmbeddingBatchSink):
            accepted_batch = list(sink.upsert_batch(writes))
        else:
            accepted_batch = [
                sink.upsert(
                    source_kind=write.source_kind,
                    source_id=write.source_id,
                    workspace_id=write.workspace_id,
                    model_name=write.model_name,
                    embedding=write.embedding,
                    source_updated_at=write.source_updated_at,
                )
                for write in writes
            ]
        if len(accepted_batch) != len(writes):
            raise ValueError(
                f"Embedding sink returned {len(accepted_batch)} acceptance results "
                f"for {len(writes)} writes"
            )
        rejected += sum(accepted is False for accepted in accepted_batch)
        stored += sum(accepted is not False for accepted in accepted_batch)
        batches += 1

    return EmbeddingBatchResult(
        attempted=len(inputs),
        stored=stored,
        rejected=rejected,
        batches=batches,
    )


def _batches(
    inputs: list[EmbeddingInput], batch_size: int
) -> Iterator[list[EmbeddingInput]]:
    """Yield source inputs in the same batches sent to the provider."""
    iterator = iter(inputs)
    while batch := list(islice(iterator, batch_size)):
        yield batch


def embed_in_batches(
    texts: list[str],
    *,
    provider: EmbeddingBatchProvider,
    batch_size: int,
) -> list[Vector]:
    """Embed texts in bounded batches and validate positional completeness."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if not texts:
        return []

    vectors: list[Vector] = []
    iterator = iter(texts)
    while batch := list(islice(iterator, batch_size)):
        batch_vectors = provider.embed(batch)
        if len(batch_vectors) != len(batch):
            raise ValueError(
                f"Embedding provider {provider.model_name!r} returned {len(batch_vectors)} "
                f"vectors for {len(batch)} inputs"
            )
        vectors.extend(list(vector) for vector in batch_vectors)
    return vectors


def iter_embed_batches(
    texts: Iterable[str],
    *,
    provider: EmbeddingBatchProvider,
    batch_size: int,
) -> Iterator[list[Vector]]:
    """Yield validated embedding batches without retaining prior vectors."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    iterator = iter(texts)
    while batch := list(islice(iterator, batch_size)):
        batch_vectors = provider.embed(batch)
        if len(batch_vectors) != len(batch):
            raise ValueError(
                f"Embedding provider {provider.model_name!r} returned {len(batch_vectors)} "
                f"vectors for {len(batch)} inputs"
            )
        yield [list(vector) for vector in batch_vectors]


async def async_embed_and_upsert(
    inputs: list[EmbeddingInput],
    *,
    provider: EmbeddingBatchProvider,
    sink: EmbeddingSink,
    batch_size: int,
) -> EmbeddingBatchResult:
    """Run the blocking embedding/upsert path without blocking the event loop."""
    return await asyncio.to_thread(
        embed_and_upsert,
        inputs,
        provider=provider,
        sink=sink,
        batch_size=batch_size,
    )


async def async_embed_in_batches(
    texts: list[str],
    *,
    provider: EmbeddingBatchProvider,
    batch_size: int,
) -> list[Vector]:
    """Run blocking model inference in a worker thread."""
    return await asyncio.to_thread(
        embed_in_batches,
        texts,
        provider=provider,
        batch_size=batch_size,
    )


async def async_iter_embed_batches(
    texts: Iterable[str],
    *,
    provider: EmbeddingBatchProvider,
    batch_size: int,
) -> AsyncIterator[list[Vector]]:
    """Yield bounded embedding responses without blocking the event loop."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    iterator = iter(texts)
    while batch := list(islice(iterator, batch_size)):
        batch_vectors = await asyncio.to_thread(provider.embed, batch)
        if len(batch_vectors) != len(batch):
            raise ValueError(
                f"Embedding provider {provider.model_name!r} returned "
                f"{len(batch_vectors)} vectors for {len(batch)} inputs"
            )
        yield [list(vector) for vector in batch_vectors]
