"""Pure contracts for embedding-model migrations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


@dataclass(frozen=True, slots=True)
class ModelNamespace:
    """Identity of one model-specific vector namespace."""

    model_name: str
    dim: int

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise ValueError("model_name must not be empty")
        if isinstance(self.dim, bool) or self.dim < 1:
            raise ValueError("dim must be a positive integer")

    @property
    def identity(self) -> tuple[str, int]:
        """Return the storage identity used by model-scoped backends."""
        return self.model_name, self.dim

    def to_dict(self) -> dict[str, Any]:
        return {"model_name": self.model_name, "dim": self.dim}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelNamespace:
        return cls(model_name=str(data["model_name"]), dim=int(data["dim"]))


class ModelDimensionMismatchError(ValueError):
    """Raised when one model name is used with more than one dimension."""


class MigrationPhase(str, Enum):
    """Durable phases understood by the model migration state machine."""

    INIT = "init"
    EXPAND = "expand"
    BACKFILL = "backfill"
    VALIDATE = "validate"
    FLIP = "flip"
    CONTRACT = "contract"
    ROLLBACK = "rollback"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ActiveModelMetadata:
    """Serving metadata for the model selected for new queries."""

    namespace: ModelNamespace
    generation: int = 0
    activated_at: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.generation, bool) or self.generation < 0:
            raise ValueError("generation must be a non-negative integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "namespace": self.namespace.to_dict(),
            "generation": self.generation,
            "activated_at": self.activated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ActiveModelMetadata:
        return cls(
            namespace=ModelNamespace.from_dict(data["namespace"]),
            generation=int(data.get("generation", 0)),
            activated_at=data.get("activated_at"),
        )


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Validation evidence for a model namespace before serving it."""

    namespace: ModelNamespace
    expected_records: int
    indexed_records: int
    errors: tuple[str, ...] = ()
    checked_at: str | None = None

    def __post_init__(self) -> None:
        if self.expected_records < 0 or self.indexed_records < 0:
            raise ValueError("record counts must be non-negative")

    @property
    def passed(self) -> bool:
        return (
            not self.errors
            and self.expected_records == self.indexed_records
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "namespace": self.namespace.to_dict(),
            "expected_records": self.expected_records,
            "indexed_records": self.indexed_records,
            "errors": list(self.errors),
            "checked_at": self.checked_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ValidationResult:
        return cls(
            namespace=ModelNamespace.from_dict(data["namespace"]),
            expected_records=int(data["expected_records"]),
            indexed_records=int(data["indexed_records"]),
            errors=tuple(str(error) for error in data.get("errors", [])),
            checked_at=data.get("checked_at"),
        )


@dataclass(frozen=True, slots=True)
class BackupMetadata:
    """Opaque storage reference for a model snapshot used by rollback."""

    backup_id: str
    namespace: ModelNamespace
    reference: str
    created_at: str | None = None

    def __post_init__(self) -> None:
        if not self.backup_id.strip():
            raise ValueError("backup_id must not be empty")
        if not self.reference.strip():
            raise ValueError("reference must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "backup_id": self.backup_id,
            "namespace": self.namespace.to_dict(),
            "reference": self.reference,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BackupMetadata:
        return cls(
            backup_id=str(data["backup_id"]),
            namespace=ModelNamespace.from_dict(data["namespace"]),
            reference=str(data["reference"]),
            created_at=data.get("created_at"),
        )


@dataclass(frozen=True, slots=True)
class RollbackMetadata:
    """Evidence describing a completed restoration."""

    from_namespace: ModelNamespace
    to_namespace: ModelNamespace
    backup_id: str
    reason: str
    rolled_back_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_namespace": self.from_namespace.to_dict(),
            "to_namespace": self.to_namespace.to_dict(),
            "backup_id": self.backup_id,
            "reason": self.reason,
            "rolled_back_at": self.rolled_back_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RollbackMetadata:
        return cls(
            from_namespace=ModelNamespace.from_dict(data["from_namespace"]),
            to_namespace=ModelNamespace.from_dict(data["to_namespace"]),
            backup_id=str(data["backup_id"]),
            reason=str(data["reason"]),
            rolled_back_at=data.get("rolled_back_at"),
        )


@dataclass(frozen=True, slots=True)
class MigrationState:
    """Durable state and evidence for one model migration."""

    migration_id: str
    source: ModelNamespace
    target: ModelNamespace
    phase: MigrationPhase = MigrationPhase.INIT
    validation: ValidationResult | None = None
    backup: BackupMetadata | None = None
    rollback: RollbackMetadata | None = None
    error: str | None = None
    resume_phase: MigrationPhase | None = None
    checkpoint: int = 0
    attempts: int = 0
    total_records: int | None = None

    def __post_init__(self) -> None:
        if not self.migration_id.strip():
            raise ValueError("migration_id must not be empty")
        if self.source == self.target:
            raise ValueError("source and target namespaces must differ")
        if (
            isinstance(self.checkpoint, bool)
            or self.checkpoint < 0
        ):
            raise ValueError("checkpoint must be a non-negative integer")
        if isinstance(self.attempts, bool) or self.attempts < 0:
            raise ValueError("attempts must be a non-negative integer")
        if (
            self.total_records is not None
            and (
                isinstance(self.total_records, bool)
                or self.total_records < 0
            )
        ):
            raise ValueError("total_records must be non-negative or None")

    def to_dict(self) -> dict[str, Any]:
        return {
            "migration_id": self.migration_id,
            "source": self.source.to_dict(),
            "target": self.target.to_dict(),
            "phase": self.phase.value,
            "checkpoint": self.checkpoint,
            "attempts": self.attempts,
            "total_records": self.total_records,
            "validation": (
                self.validation.to_dict() if self.validation is not None else None
            ),
            "backup": self.backup.to_dict() if self.backup is not None else None,
            "rollback": (
                self.rollback.to_dict() if self.rollback is not None else None
            ),
            "error": self.error,
            "resume_phase": (
                self.resume_phase.value
                if self.resume_phase is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MigrationState:
        validation = data.get("validation")
        backup = data.get("backup")
        rollback = data.get("rollback")
        return cls(
            migration_id=str(data["migration_id"]),
            source=ModelNamespace.from_dict(data["source"]),
            target=ModelNamespace.from_dict(data["target"]),
            phase=MigrationPhase(data.get("phase", MigrationPhase.INIT.value)),
            checkpoint=int(data.get("checkpoint", 0)),
            attempts=int(data.get("attempts", 0)),
            total_records=(
                int(data["total_records"])
                if data.get("total_records") is not None
                else None
            ),
            validation=(
                ValidationResult.from_dict(validation)
                if validation is not None
                else None
            ),
            backup=(
                BackupMetadata.from_dict(backup) if backup is not None else None
            ),
            rollback=(
                RollbackMetadata.from_dict(rollback)
                if rollback is not None
                else None
            ),
            error=data.get("error"),
            resume_phase=(
                MigrationPhase(data["resume_phase"])
                if data.get("resume_phase") is not None
                else None
            ),
        )
