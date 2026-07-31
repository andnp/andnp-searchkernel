"""Async batch ingestion adapters for blocking index implementations."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Protocol

from searchkernel.domain import Record
from searchkernel.ports.content_source import (
    IngestionFailureMode,
    IngestionReceipt,
    RecordIngestionResult,
)


class BlockingRecordIndexer(Protocol):
    """Synchronous index surface that must be isolated from the event loop."""

    def index_record(self, record: Record) -> bool:
        ...


class AsyncIndexIngestor:
    """Adapt a blocking record indexer to the async batch ingestion port."""

    def __init__(self, indexer: BlockingRecordIndexer) -> None:
        self._indexer = indexer

    async def index_records(
        self,
        records: Sequence[Record],
        *,
        checkpoint: str | None = None,
        failure_mode: IngestionFailureMode = "strict",
    ) -> IngestionReceipt:
        if failure_mode not in {"strict", "lenient"}:
            raise ValueError("failure_mode must be 'strict' or 'lenient'")

        outcomes: list[RecordIngestionResult] = []
        for record in records:
            try:
                changed = await asyncio.to_thread(
                    self._indexer.index_record, record
                )
            except asyncio.CancelledError:
                raise
            except Exception as error:  # noqa: BLE001
                outcomes.append(
                    RecordIngestionResult(
                        source_kind=record.source_kind,
                        source_id=record.source_id,
                        workspace_id=record.workspace_id,
                        status="failed",
                        error=f"{type(error).__name__}: {error}",
                    )
                )
                if failure_mode == "strict":
                    break
            else:
                outcomes.append(
                    RecordIngestionResult(
                        source_kind=record.source_kind,
                        source_id=record.source_id,
                        workspace_id=record.workspace_id,
                        status="committed" if changed else "skipped",
                    )
                )

        return IngestionReceipt(
            source_kind=records[0].source_kind if records else "",
            workspace_id=_workspace_id(records),
            checkpoint=checkpoint,
            records=tuple(outcomes),
        )


def _workspace_id(records: Sequence[Record]) -> str | None:
    values = {record.workspace_id for record in records}
    if len(values) == 1:
        return next(iter(values))
    return None


__all__ = ["AsyncIndexIngestor", "BlockingRecordIndexer"]
