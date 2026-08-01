"""Batch reindexing against the unified vector-store port.

The vector-store port deliberately selects a model per operation. It does not
own an active model, model-scoped deletion, or a migration manifest, so this
module cannot safely implement expand/flip/contract/rollback. Those operations
remain explicit errors rather than logging-only placeholders. ``backfill`` is
the supported operation: it writes the target model through the existing
``VectorStore.upsert`` contract and is safe to retry.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from searchkernel.domain import Record
from searchkernel.domain.reindex import (
    ActiveModelMetadata,
    BackupMetadata,
    MigrationPhase,
    MigrationState,
    ModelDimensionMismatchError,
    ModelNamespace,
    RollbackMetadata,
    ValidationResult,
)
from searchkernel.ports.embedding import EmbeddingProvider
from searchkernel.ports.reindex import (
    ActiveModelStore,
    ModelBackupStore,
    ModelLifecycleStore,
    ModelNamespaceStore,
    ModelValidationStore,
)
from searchkernel.ports.stores import VectorStore

__all__ = [
    "ActiveModelMetadata",
    "ActiveModelStore",
    "BackupMetadata",
    "MigrationPhase",
    "MigrationState",
    "ModelBackupStore",
    "ModelDimensionMismatchError",
    "ModelLifecycleStore",
    "ModelNamespace",
    "ModelNamespaceStore",
    "ModelValidationStore",
    "ReindexError",
    "ReindexProgress",
    "ReindexRoutine",
    "RollbackMetadata",
    "ValidationResult",
]


class ReindexError(Exception):
    """Raised when a supported reindex operation fails."""


@dataclass
class ReindexProgress:
    """Progress tracking for a backfill operation."""

    stage: str
    """Current stage. The supported stage is ``backfill``."""

    records_processed: int = 0
    """Number of records processed in the current stage."""

    total_records: int = 0
    """Total records to process."""

    errors: list[str] = field(default_factory=list)


class ReindexRoutine:
    """Backfill a new embedding model without pretending to control serving.

    ``VectorStore`` already provides per-model isolation, so backfill can
    coexist with an existing model. Selecting the active model and removing
    retired embeddings must be performed by the application that owns serving
    configuration and storage lifecycle.
    """

    def __init__(
        self,
        records: list[Record],
        target_provider: EmbeddingProvider,
        vector_store: VectorStore,
        batch_size: int = 64,
        truncate_dim: int | None = None,
    ):
        """Initialize a target-model backfill.

        Args:
            records: Full corpus of records to re-embed.
            target_provider: Provider for the target model.
            vector_store: Store receiving target embeddings.
            batch_size: Number of records embedded per provider call.
            truncate_dim: Optional target dimension no larger than the
                provider dimension.
        """
        if batch_size < 1:
            raise ReindexError("batch_size must be >= 1")
        if truncate_dim is not None and not 0 < truncate_dim <= target_provider.dim:
            raise ReindexError(
                f"truncate_dim {truncate_dim} must be > 0 and <= "
                f"provider dim {target_provider.dim}"
            )

        self.records = records
        self.target_provider = target_provider
        self.vector_store = vector_store
        self.batch_size = batch_size
        self.truncate_dim = truncate_dim or target_provider.dim
        self._current_stage = "init"

    @property
    def stage(self) -> str:
        """Return the current routine stage."""
        return self._current_stage

    @property
    def target_dim(self) -> int:
        """Effective target dimension after optional truncation."""
        return self.truncate_dim

    def expand(self) -> None:
        """Reject unsupported table creation through the narrow store port."""
        raise ReindexError(
            "expand is unavailable: VectorStore does not expose model-scoped "
            "table creation"
        )

    def backfill(self) -> ReindexProgress:
        """Embed and upsert the target model in retryable batches.

        Each write uses a copied ``Record`` so the caller's records continue
        to describe the currently served model until serving is switched by
        its owning application.
        """
        self._current_stage = "backfill"
        progress = ReindexProgress(
            stage="backfill",
            total_records=len(self.records),
        )

        for offset in range(0, len(self.records), self.batch_size):
            batch = self.records[offset : offset + self.batch_size]
            try:
                embeddings = self.target_provider.embed([record.body for record in batch])
                if len(embeddings) != len(batch):
                    raise ReindexError(
                        f"provider returned {len(embeddings)} embeddings for "
                        f"{len(batch)} records"
                    )

                target_records: list[Record] = []
                for record, embedding in zip(batch, embeddings, strict=True):
                    if len(embedding) != self.target_provider.dim:
                        raise ReindexError(
                            f"provider returned dimension {len(embedding)}; "
                            f"expected {self.target_provider.dim}"
                        )
                    target_embedding = embedding[: self.target_dim]
                    target_records.append(
                        replace(
                            record,
                            embedding=target_embedding,
                            embedding_model=self.target_provider.model_name,
                        )
                    )

                self.vector_store.upsert(
                    target_records,
                    model_name=self.target_provider.model_name,
                    dim=self.target_dim,
                )
                progress.records_processed += len(batch)
            except ReindexError as exc:
                message = f"Batch {offset // self.batch_size} failed: {exc}"
                progress.errors.append(message)
                raise ReindexError(message) from exc
            except Exception as exc:
                message = f"Batch {offset // self.batch_size} failed: {exc}"
                progress.errors.append(message)
                raise ReindexError(message) from exc

        return progress

    def flip(self) -> None:
        """Reject unsupported active-model switching through the store port."""
        raise ReindexError(
            "flip is unavailable: VectorStore selects models per operation and "
            "does not own active-model configuration"
        )

    def contract(self, old_model_name: str) -> None:
        """Reject unsupported model-scoped cleanup through the store port."""
        raise ReindexError(
            f"contract is unavailable for {old_model_name!r}: VectorStore does "
            "not expose model-scoped deletion"
        )

    def rollback(self, old_model_name: str) -> None:
        """Reject unsupported rollback through the store port."""
        raise ReindexError(
            f"rollback is unavailable for {old_model_name!r}: VectorStore does "
            "not expose model-scoped deletion or active-model switching"
        )
