"""Capability protocols for embedding-model migration."""

from collections.abc import Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from searchkernel.domain import Cursor, Record
from searchkernel.domain.reindex import (
    ActiveModelMetadata,
    BackupMetadata,
    MigrationState,
    ModelNamespace,
    RollbackMetadata,
    ValidationResult,
)

type ReindexCursor = Cursor


@dataclass(frozen=True, slots=True)
class RecordBatch:
    """One bounded page returned by a cursor-based reindex source."""

    records: Sequence[Record]
    next_cursor: ReindexCursor = None
    total_records: int | None = None


@runtime_checkable
class RecordSource(Protocol):
    """Read records in bounded pages using an opaque source cursor."""

    @property
    def total_records(self) -> int | None:
        """Return the source count when it can be computed cheaply."""
        ...

    def fetch_batch(
        self,
        cursor: ReindexCursor,
        limit: int,
    ) -> RecordBatch:
        """Return at most ``limit`` records and the cursor for the next page."""
        ...


ReindexBatch = RecordBatch
ReindexRecordSource = RecordSource


class ModelNamespaceStore(Protocol):
    """Manage model-scoped vector namespaces."""

    def ensure_namespace(self, namespace: ModelNamespace) -> None:
        """Create a namespace if it does not already exist."""
        ...

    def delete_namespace(self, namespace: ModelNamespace) -> None:
        """Delete one namespace without affecting other model namespaces."""
        ...


class ModelValidationStore(Protocol):
    """Validate the completeness and dimensionality of a namespace."""

    def validate_namespace(
        self,
        namespace: ModelNamespace,
        expected_records: int,
    ) -> ValidationResult:
        """Return evidence that a namespace is safe to serve."""
        ...


class ActiveModelStore(Protocol):
    """Read and atomically update serving model metadata."""

    def get_active_model(self) -> ActiveModelMetadata | None:
        """Return the model selected for new queries."""
        ...

    def set_active_model(self, active_model: ActiveModelMetadata) -> None:
        """Atomically select the model for new queries."""
        ...

    def compare_and_set_active_model(
        self,
        expected: ActiveModelMetadata | None,
        active_model: ActiveModelMetadata,
    ) -> bool:
        """Select a model only when the current metadata matches expected."""
        ...


class ModelBackupStore(Protocol):
    """Create and restore model-scoped rollback backups."""

    def create_backup(self, namespace: ModelNamespace) -> BackupMetadata:
        """Create an opaque backup reference for a namespace."""
        ...

    def restore_backup(self, backup: BackupMetadata) -> RollbackMetadata:
        """Restore a namespace from a previously created backup."""
        ...


class ModelLifecycleStore(
    ModelNamespaceStore,
    ModelValidationStore,
    ActiveModelStore,
    ModelBackupStore,
    Protocol,
):
    """Full capability set required by the future migration state machine."""

    def acquire_transition_lock(self) -> AbstractContextManager[None]:
        """Acquire a durable lock for one full active-model transition."""
        ...

    def load_migration(self, migration_id: str) -> MigrationState | None:
        """Load durable migration state, if present."""
        ...

    def save_migration(self, migration: MigrationState) -> None:
        """Persist durable migration state atomically."""
        ...


__all__ = [
    "ActiveModelStore",
    "ModelBackupStore",
    "ModelLifecycleStore",
    "ModelNamespaceStore",
    "ModelValidationStore",
    "RecordBatch",
    "RecordSource",
    "ReindexBatch",
    "ReindexCursor",
    "ReindexRecordSource",
]
