"""Capability protocols for embedding-model migration."""

from typing import Protocol

from searchkernel.domain.reindex import (
    ActiveModelMetadata,
    BackupMetadata,
    MigrationState,
    ModelNamespace,
    RollbackMetadata,
    ValidationResult,
)


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

    def load_migration(self, migration_id: str) -> MigrationState | None:
        """Load durable migration state, if present."""
        ...

    def save_migration(self, migration: MigrationState) -> None:
        """Persist durable migration state atomically."""
        ...
