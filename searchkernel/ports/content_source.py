"""Ports for asynchronous, checkpointed content ingestion."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from searchkernel.domain import ChangeSignal, Cursor, Record, ScoredRef
from searchkernel.ports.retrieval import SourceCapabilities

IngestionFailureMode = Literal["strict", "lenient"]
RecordIngestionStatus = Literal["committed", "skipped", "failed", "cancelled"]


@dataclass(frozen=True, slots=True)
class RecordIngestionResult:
    """Outcome for one record in an ingestion batch."""

    source_kind: str
    source_id: str
    workspace_id: str | None
    status: RecordIngestionStatus
    cursor: Cursor = None
    error: str | None = None

    @property
    def successful(self) -> bool:
        return self.status in {"committed", "skipped"}


@dataclass(frozen=True, slots=True)
class IngestionReceipt:
    """Outcome of one committed or attempted ingestion batch."""

    source_kind: str
    workspace_id: str | None
    checkpoint: Cursor
    records: tuple[RecordIngestionResult, ...]
    cancelled: bool = False

    @property
    def attempted(self) -> int:
        return len(self.records)

    @property
    def committed(self) -> int:
        return sum(record.status == "committed" for record in self.records)

    @property
    def skipped(self) -> int:
        return sum(record.status == "skipped" for record in self.records)

    @property
    def failed(self) -> int:
        return sum(record.status == "failed" for record in self.records)

    @property
    def successful(self) -> int:
        return self.committed + self.skipped

    @property
    def failures(self) -> tuple[RecordIngestionResult, ...]:
        return tuple(record for record in self.records if record.status == "failed")


IngestionResult = IngestionReceipt
IngestionBatchResult = IngestionReceipt


@dataclass(frozen=True, slots=True)
class SourceBatch:
    """Records emitted together with the source cursor at batch termination."""

    records: Sequence[Record]
    terminal_cursor: Cursor = None


class IngestionError(RuntimeError):
    """Raised when strict ingestion cannot commit a complete batch."""

    def __init__(self, receipt: IngestionReceipt):
        self.receipt = receipt
        super().__init__(
            f"Failed to ingest {receipt.failed} record(s) from "
            f"{receipt.source_kind!r} in strict mode"
        )


@runtime_checkable
class ContentSource(Protocol):
    """Asynchronous source of source-agnostic records."""

    source_kind: str

    def iter_records(self, since: Cursor | None = None) -> AsyncIterator[Record]:
        """Yield records after the supplied source-owned cursor."""
        ...

    def change_signal(self) -> ChangeSignal:
        """Return source-specific watch or polling configuration."""
        ...

    def cursor_for(self, record: Record) -> Cursor:
        """Return the source-owned cursor represented by a record."""
        ...


@runtime_checkable
class BatchContentSource(Protocol):
    """Optional source contract for batches with terminal cursors."""

    source_kind: str

    def iter_batches(
        self, since: Cursor | None = None
    ) -> AsyncIterator[SourceBatch]:
        """Yield source batches after the supplied source-owned cursor."""
        ...


@runtime_checkable
class CheckpointStore(Protocol):
    """Durable cursor persistence used after an index commit."""

    async def load(
        self, source_kind: str, workspace_id: str | None = None
    ) -> Cursor:
        ...

    async def save(
        self,
        source_kind: str,
        workspace_id: str | None,
        checkpoint: Cursor,
    ) -> None:
        ...


@runtime_checkable
class RecordIngestor(Protocol):
    """Asynchronous batch indexing boundary."""

    async def index_records(
        self,
        records: Sequence[Record],
        *,
        checkpoint: Cursor | None = None,
        failure_mode: IngestionFailureMode = "strict",
    ) -> IngestionReceipt:
        """Index a batch and report per-record outcomes."""
        ...


AsyncRecordIngestor = RecordIngestor


@runtime_checkable
class SearchableSource(Protocol):
    """Federated source whose native search is merged by the kernel."""

    source_kind: str

    async def search(
        self, query: str, k: int, filters: dict[str, Any] | None = None
    ) -> Iterable[ScoredRef]:
        ...


@runtime_checkable
class HierarchicalSearchableSource(Protocol):
    """Optional parent-first search contract for structured source adapters."""

    source_kind: str
    capabilities: SourceCapabilities

    async def search_parents(
        self, query: str, k: int, filters: dict[str, Any] | None = None
    ) -> Iterable[ScoredRef]:
        ...

    async def search_children(
        self,
        query: str,
        parent_ids: Sequence[str],
        k: int,
        filters: dict[str, Any] | None = None,
    ) -> Iterable[ScoredRef]:
        """Search children for canonical parent storage keys.

        Legacy source-id requests remain adaptable when the selected parent
        identities have unique source IDs.
        """
        ...


__all__ = [
    "AsyncRecordIngestor",
    "BatchContentSource",
    "CheckpointStore",
    "ContentSource",
    "HierarchicalSearchableSource",
    "IngestionBatchResult",
    "IngestionError",
    "IngestionFailureMode",
    "IngestionReceipt",
    "IngestionResult",
    "RecordIngestionResult",
    "RecordIngestionStatus",
    "RecordIngestor",
    "SearchableSource",
    "SourceBatch",
]
