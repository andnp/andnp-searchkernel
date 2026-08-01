"""Wires any ContentSource into the live IndexManager.

Ingests source records (git commits, notes, or any future ContentSource)
through the same chunking/indexing path as documents, so they land in the
shared vector/keyword/graph store and become discoverable via
SearchOrchestrator.query(source_filter=[...]).
"""

import logging
from collections.abc import AsyncIterator
from typing import Protocol

from searchkernel.domain import Cursor, Record
from searchkernel.ports.content_source import (
    IngestionFailureMode,
    IngestionReceipt,
    RecordIngestionResult,
)

logger = logging.getLogger(__name__)


class GitIndexManager(Protocol):
    """Minimum async index surface required for source ingestion."""

    async def index_records(
        self,
        records: list[Record],
        *,
        checkpoint: Cursor | None = None,
        failure_mode: IngestionFailureMode = "strict",
    ) -> IngestionReceipt:
        ...


class IngestibleSource(Protocol):
    """Minimum ContentSource surface required for ingestion."""

    repo_path: str

    def iter_records(self, since: Cursor | None) -> AsyncIterator[Record]: ...


async def ingest_git_source(
    index_manager: GitIndexManager,
    source: IngestibleSource,
    since: Cursor | None = None,
    *,
    batch_size: int = 100,
    failure_mode: IngestionFailureMode = "strict",
) -> IngestionReceipt:
    """Legacy direct ingestion helper for callers owning an IndexManager.

    ``SearchKernel.ingest_source`` is the canonical path for checkpointed
    ingestion, including sources that emit terminal batch cursors. This helper
    remains for callers that own the index manager and durable persistence.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    results: list[RecordIngestionResult] = []
    checkpoint = since
    batch: list[Record] = []
    async for record in source.iter_records(since):
        batch.append(record)
        if len(batch) < batch_size:
            continue
        receipt = await index_manager.index_records(
            batch,
            checkpoint=checkpoint,
            failure_mode=failure_mode,
        )
        results.extend(receipt.records)
        checkpoint = receipt.checkpoint
        batch = []

    if batch:
        receipt = await index_manager.index_records(
            batch,
            checkpoint=checkpoint,
            failure_mode=failure_mode,
        )
        results.extend(receipt.records)
        checkpoint = receipt.checkpoint

    if results:
        logger.info(
            "Ingested %d record(s) from %s into the live index",
            len(results),
            source.repo_path,
        )

    return IngestionReceipt(
        source_kind=results[0].source_kind if results else source.repo_path,
        workspace_id=None,
        checkpoint=checkpoint,
        records=tuple(results),
    )
