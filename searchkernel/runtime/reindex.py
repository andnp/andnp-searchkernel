"""Durable embedding-model migration state machine."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime

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
from searchkernel.indexing.semantic import semantic_input_for_record
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
    """Raised when a migration cannot safely advance."""


@dataclass
class ReindexProgress:
    """Progress and durable checkpoint evidence for one backfill call."""

    stage: str
    records_processed: int = 0
    total_records: int = 0
    checkpoint: int = 0
    attempts: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        """Return whether the supplied corpus has reached its checkpoint."""
        return self.checkpoint >= self.total_records


class ReindexRoutine:
    """Coordinate a resumable expand/backfill/validate/flip/contract flow.

    A lifecycle store is required for destructive or serving-changing stages.
    The legacy vector-only constructor remains supported for backfill callers,
    but it deliberately rejects those lifecycle stages.
    """

    def __init__(
        self,
        records: list[Record],
        target_provider: EmbeddingProvider,
        vector_store: VectorStore,
        batch_size: int = 64,
        truncate_dim: int | None = None,
        *,
        lifecycle_store: ModelLifecycleStore | None = None,
        migration_id: str | None = None,
        source_namespace: ModelNamespace | None = None,
    ) -> None:
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
        self.lifecycle_store = lifecycle_store
        self.migration_id = migration_id or (
            f"reindex:{target_provider.model_name}:{self.truncate_dim}"
        )
        self.source_namespace = source_namespace
        self._current_stage = "init"

    @property
    def stage(self) -> str:
        """Return the current durable phase when lifecycle state is available."""
        if self.lifecycle_store is None:
            return self._current_stage
        state = self.lifecycle_store.load_migration(self.migration_id)
        return state.phase.value if state is not None else self._current_stage

    @property
    def target_namespace(self) -> ModelNamespace:
        """Return the target model namespace."""
        return ModelNamespace(
            model_name=self.target_provider.model_name,
            dim=self.target_dim,
        )

    @property
    def target_dim(self) -> int:
        """Return the effective target dimension after optional truncation."""
        return self.truncate_dim

    @property
    def state(self) -> MigrationState | None:
        """Return the persisted migration state, if lifecycle support is enabled."""
        if self.lifecycle_store is None:
            return None
        return self.lifecycle_store.load_migration(self.migration_id)

    @property
    def checkpoint(self) -> int:
        """Return the durable backfill checkpoint."""
        state = self.state
        return state.checkpoint if state is not None else 0

    def expand(self) -> MigrationState | None:
        """Create the target namespace without changing the active model."""
        self._require_lifecycle("expand")
        store = self._lifecycle()
        state = self._load_or_create_state()
        self._reject_failed(state)
        if state.phase is not MigrationPhase.INIT:
            self._current_stage = state.phase.value
            return state

        try:
            store.ensure_namespace(state.target)
            state = replace(
                state,
                phase=MigrationPhase.EXPAND,
                error=None,
                resume_phase=None,
            )
            self._save(state)
        except ModelDimensionMismatchError as exc:
            self._fail(state, MigrationPhase.EXPAND, str(exc))
            raise
        except Exception as exc:
            self._fail(state, MigrationPhase.EXPAND, str(exc))
            raise ReindexError(f"Failed to expand migration: {exc}") from exc

        self._current_stage = MigrationPhase.EXPAND.value
        return state

    def backfill(self) -> ReindexProgress:
        """Embed and persist target batches from the last durable checkpoint."""
        if self.lifecycle_store is None:
            return self._legacy_backfill()

        state = self._load_or_create_state()
        self._reject_failed(state)
        if state.phase is MigrationPhase.INIT:
            raise ReindexError("Cannot backfill before expand is complete")
        if state.phase in {
            MigrationPhase.VALIDATE,
            MigrationPhase.FLIP,
            MigrationPhase.CONTRACT,
            MigrationPhase.COMPLETE,
            MigrationPhase.ROLLBACK,
        }:
            return ReindexProgress(
                stage=state.phase.value,
                total_records=len(self.records),
                checkpoint=state.checkpoint,
                attempts=state.attempts,
            )
        if state.checkpoint > len(self.records):
            raise ReindexError(
                f"backfill checkpoint {state.checkpoint} exceeds "
                f"record count {len(self.records)}"
            )

        state = replace(
            state,
            phase=MigrationPhase.BACKFILL,
            error=None,
            resume_phase=None,
        )
        self._save(state)
        progress = ReindexProgress(
            stage=MigrationPhase.BACKFILL.value,
            total_records=len(self.records),
            checkpoint=state.checkpoint,
            attempts=state.attempts,
        )

        for offset in range(state.checkpoint, len(self.records), self.batch_size):
            batch = self.records[offset : offset + self.batch_size]
            attempt = state.attempts + 1
            state = replace(state, attempts=attempt, error=None)
            self._save(state)
            try:
                embeddings = self.target_provider.embed(
                    [semantic_input_for_record(record).text for record in batch]
                )
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
                    target_records.append(
                        replace(
                            record,
                            embedding=embedding[: self.target_dim],
                            embedding_model=self.target_provider.model_name,
                        )
                    )

                self.vector_store.upsert(
                    target_records,
                    model_name=self.target_provider.model_name,
                    dim=self.target_dim,
                )
            except ModelDimensionMismatchError as exc:
                message = f"Batch {offset // self.batch_size} failed: {exc}"
                self._fail(state, MigrationPhase.BACKFILL, message)
                raise
            except Exception as exc:
                message = f"Batch {offset // self.batch_size} failed: {exc}"
                self._fail(state, MigrationPhase.BACKFILL, message)
                raise ReindexError(message) from exc

            next_checkpoint = offset + len(batch)
            state = replace(
                state,
                checkpoint=next_checkpoint,
                error=None,
                resume_phase=None,
            )
            self._save(state)
            progress.records_processed += len(batch)
            progress.checkpoint = next_checkpoint
            progress.attempts = state.attempts

        self._current_stage = MigrationPhase.BACKFILL.value
        return progress

    def validate(self) -> ValidationResult:
        """Validate every target record before any serving-model change."""
        self._require_lifecycle("validate")
        store = self._lifecycle()
        state = self._load_or_create_state()
        self._reject_failed(state)
        if state.validation is not None and state.validation.passed:
            return state.validation
        if state.phase not in {
            MigrationPhase.BACKFILL,
            MigrationPhase.VALIDATE,
        }:
            raise ReindexError("Cannot validate before backfill is complete")
        if state.checkpoint != len(self.records):
            raise ReindexError(
                f"Cannot validate incomplete backfill: "
                f"{state.checkpoint}/{len(self.records)} records"
            )

        try:
            result = store.validate_namespace(
                state.target,
                expected_records=len(self.records),
            )
            if result.namespace != state.target:
                raise ReindexError(
                    "validation returned evidence for the wrong target namespace"
                )
            if not result.passed:
                message = "target namespace validation failed"
                if result.errors:
                    message = f"{message}: {'; '.join(result.errors)}"
                failed = replace(
                    state,
                    phase=MigrationPhase.FAILED,
                    validation=result,
                    error=message,
                    resume_phase=MigrationPhase.VALIDATE,
                )
                self._save(failed)
                raise ReindexError(message)
            state = replace(
                state,
                phase=MigrationPhase.VALIDATE,
                validation=result,
                error=None,
                resume_phase=None,
            )
            self._save(state)
        except ReindexError:
            raise
        except Exception as exc:
            self._fail(state, MigrationPhase.VALIDATE, str(exc))
            raise ReindexError(f"Failed to validate migration: {exc}") from exc

        self._current_stage = MigrationPhase.VALIDATE.value
        return result

    def flip(self) -> MigrationState:
        """Atomically select the validated target model for new queries."""
        self._require_lifecycle("flip")
        with self._transition_lock() as store:
            state = self._load_or_create_state()
            self._reject_failed(state)
            if state.validation is None or not state.validation.passed:
                raise ReindexError("Cannot flip before validation passes")
            if state.phase not in {
                MigrationPhase.VALIDATE,
                MigrationPhase.FLIP,
            }:
                raise ReindexError("Cannot flip before validation is complete")

            active = store.get_active_model()
            if active is not None and active.namespace == state.target:
                state = replace(
                    state,
                    phase=MigrationPhase.FLIP,
                    error=None,
                    resume_phase=None,
                )
                self._save(state)
                self._current_stage = MigrationPhase.FLIP.value
                return state
            if active is not None and active.namespace != state.source:
                raise ReindexError(
                    "active model changed during migration; refusing to overwrite policy"
                )

            pending = replace(
                state,
                phase=MigrationPhase.FLIP,
                error=None,
                resume_phase=None,
            )
            self._save(pending)
            generation = (active.generation if active is not None else 0) + 1
            target_active = ActiveModelMetadata(
                namespace=state.target,
                generation=generation,
                activated_at=datetime.now(UTC).isoformat(),
            )
            try:
                self._compare_and_set_active_model(
                    store,
                    expected=active,
                    active_model=target_active,
                    error_message=(
                        "active model changed during migration; refusing to "
                        "overwrite policy"
                    ),
                )
            except Exception as exc:
                self._fail(pending, MigrationPhase.FLIP, str(exc))
                raise ReindexError(f"Failed to flip active model: {exc}") from exc

            state = replace(pending, error=None, resume_phase=None)
            self._save(state)
            self._current_stage = MigrationPhase.FLIP.value
            return state

    def contract(self, old_model_name: str | None = None) -> MigrationState:
        """Backup and delete the old namespace after a validated flip."""
        self._require_lifecycle("contract")
        with self._transition_lock() as store:
            state = self._load_or_create_state()
            self._check_old_model_name(state, old_model_name)
            self._reject_failed(state)
            if state.phase is MigrationPhase.COMPLETE:
                return state
            if state.phase not in {
                MigrationPhase.FLIP,
                MigrationPhase.CONTRACT,
            }:
                raise ReindexError("Cannot contract before flip is complete")
            if state.validation is None or not state.validation.passed:
                raise ReindexError("Cannot contract before validation passes")

            active = store.get_active_model()
            if active is None or active.namespace != state.target:
                raise ReindexError(
                    "Cannot contract while the target model is not active"
                )

            pending = replace(
                state,
                phase=MigrationPhase.CONTRACT,
                error=None,
                resume_phase=None,
            )
            self._save(pending)
            if pending.backup is None:
                try:
                    backup = store.create_backup(state.source)
                except Exception as exc:
                    self._fail(pending, MigrationPhase.CONTRACT, str(exc))
                    raise ReindexError(f"Failed to back up old model: {exc}") from exc
                pending = replace(pending, backup=backup)
                self._save(pending)

            try:
                self._require_active_owner(
                    store,
                    active,
                    "active model changed during contract; refusing to delete "
                    "the old namespace",
                )
                store.delete_namespace(state.source)
            except Exception as exc:
                self._fail(pending, MigrationPhase.CONTRACT, str(exc))
                raise ReindexError(f"Failed to contract old model: {exc}") from exc

            complete = replace(
                pending,
                phase=MigrationPhase.COMPLETE,
                error=None,
                resume_phase=None,
            )
            self._save(complete)
            self._current_stage = MigrationPhase.COMPLETE.value
            return complete

    def rollback(self, old_model_name: str | None = None) -> MigrationState:
        """Restore the source model or remove an unflipped target namespace."""
        self._require_lifecycle("rollback")
        with self._transition_lock() as store:
            state = self._load_or_create_state()
            self._check_old_model_name(state, old_model_name)

            effective_phase = (
                state.resume_phase
                if state.phase is MigrationPhase.FAILED and state.resume_phase is not None
                else state.phase
            )
            if effective_phase is MigrationPhase.ROLLBACK:
                restore_required = False
            else:
                restore_required = (
                    state.backup is not None
                    and effective_phase in {
                        MigrationPhase.CONTRACT,
                        MigrationPhase.COMPLETE,
                    }
                )
            if effective_phase is MigrationPhase.INIT:
                return state

            pending = replace(
                state,
                phase=MigrationPhase.ROLLBACK,
                error=None,
                resume_phase=None,
            )
            self._save(pending)
            rollback_metadata = pending.rollback
            try:
                if restore_required and pending.backup is not None:
                    rollback_metadata = store.restore_backup(pending.backup)

                active = store.get_active_model()
                if active is not None and active.namespace not in {
                    state.source,
                    state.target,
                }:
                    raise ReindexError(
                        "active model changed during rollback; refusing to overwrite policy"
                    )
                if active is None or active.namespace == state.target:
                    source_generation = (active.generation if active else 0) + 1
                    source_active = ActiveModelMetadata(
                        namespace=state.source,
                        generation=source_generation,
                        activated_at=datetime.now(UTC).isoformat(),
                    )
                    self._compare_and_set_active_model(
                        store,
                        expected=active,
                        active_model=source_active,
                        error_message=(
                            "active model changed during rollback; refusing to "
                            "overwrite policy"
                        ),
                    )
                    active_after = source_active
                else:
                    active_after = active
                self._require_active_owner(
                    store,
                    active_after,
                    "active model changed during rollback; refusing to delete "
                    "the target namespace",
                )
                if effective_phase is not MigrationPhase.INIT:
                    store.delete_namespace(state.target)
            except Exception as exc:
                self._fail(pending, MigrationPhase.ROLLBACK, str(exc))
                raise ReindexError(f"Failed to roll back migration: {exc}") from exc

            complete = replace(
                pending,
                phase=MigrationPhase.ROLLBACK,
                rollback=rollback_metadata,
                error=None,
                resume_phase=None,
            )
            self._save(complete)
            self._current_stage = MigrationPhase.ROLLBACK.value
            return complete

    def retry(self) -> MigrationState:
        """Clear a durable failure and expose the exact stage to retry."""
        self._require_lifecycle("retry")
        state = self._load_or_create_state()
        if state.phase is not MigrationPhase.FAILED:
            raise ReindexError("migration is not failed")
        resume_phase = state.resume_phase
        if resume_phase is None:
            resume_phase = (
                MigrationPhase.BACKFILL
                if state.checkpoint < len(self.records)
                else MigrationPhase.VALIDATE
            )
        retried = replace(
            state,
            phase=resume_phase,
            error=None,
            resume_phase=None,
        )
        self._save(retried)
        self._current_stage = resume_phase.value
        return retried

    def run(self, *, contract: bool = False) -> MigrationState:
        """Run all safe stages, optionally performing post-flip cleanup."""
        self.expand()
        self.backfill()
        self.validate()
        self.flip()
        if contract:
            return self.contract()
        state = self.state
        if state is None:
            raise ReindexError("migration state was not persisted")
        return state

    def _legacy_backfill(self) -> ReindexProgress:
        self._current_stage = MigrationPhase.BACKFILL.value
        progress = ReindexProgress(
            stage=MigrationPhase.BACKFILL.value,
            total_records=len(self.records),
        )
        for offset in range(0, len(self.records), self.batch_size):
            batch = self.records[offset : offset + self.batch_size]
            try:
                embeddings = self.target_provider.embed(
                    [semantic_input_for_record(record).text for record in batch]
                )
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
                    target_records.append(
                        replace(
                            record,
                            embedding=embedding[: self.target_dim],
                            embedding_model=self.target_provider.model_name,
                        )
                    )
                self.vector_store.upsert(
                    target_records,
                    model_name=self.target_provider.model_name,
                    dim=self.target_dim,
                )
            except ReindexError as exc:
                message = f"Batch {offset // self.batch_size} failed: {exc}"
                progress.errors.append(message)
                raise ReindexError(message) from exc
            except Exception as exc:
                message = f"Batch {offset // self.batch_size} failed: {exc}"
                progress.errors.append(message)
                raise ReindexError(message) from exc
            progress.records_processed += len(batch)
            progress.checkpoint = offset + len(batch)
        return progress

    def _load_or_create_state(self) -> MigrationState:
        store = self._lifecycle()
        saved = store.load_migration(self.migration_id)
        target = self.target_namespace
        corpus_fingerprint = self._corpus_fingerprint()
        if saved is not None:
            if saved.target != target:
                raise ReindexError(
                    f"migration {self.migration_id!r} targets "
                    f"{saved.target.identity}, not {target.identity}"
                )
            self._check_namespace_dimensions(saved.source, saved.target)
            if self.source_namespace is not None and saved.source != self.source_namespace:
                raise ReindexError("migration source namespace does not match")
            if saved.corpus_fingerprint is None:
                return self._migrate_legacy_state(saved, corpus_fingerprint)
            if saved.corpus_fingerprint != corpus_fingerprint:
                raise ReindexError(
                    "migration checkpoint corpus identity does not match "
                    "the supplied records"
                )
            if (
                saved.total_records is not None
                and saved.total_records != len(self.records)
            ):
                raise ReindexError(
                    f"migration expects {saved.total_records} records, "
                    f"received {len(self.records)}"
                )
            return saved

        source = self.source_namespace
        if source is None:
            active = store.get_active_model()
            if active is None:
                raise ReindexError(
                    "source_namespace is required when no active model exists"
                )
            source = active.namespace
        self._check_namespace_dimensions(source, target)
        return MigrationState(
            migration_id=self.migration_id,
            source=source,
            target=target,
            total_records=len(self.records),
            corpus_fingerprint=corpus_fingerprint,
        )

    def _migrate_legacy_state(
        self,
        saved: MigrationState,
        corpus_fingerprint: str,
    ) -> MigrationState:
        if saved.checkpoint == 0:
            migrated = replace(
                saved,
                total_records=len(self.records),
                corpus_fingerprint=corpus_fingerprint,
            )
            self._save(migrated)
            return migrated
        if saved.phase in {
            MigrationPhase.FLIP,
            MigrationPhase.CONTRACT,
            MigrationPhase.COMPLETE,
            MigrationPhase.ROLLBACK,
        }:
            raise ReindexError(
                "legacy migration checkpoint lacks corpus identity; "
                "refusing to resume a completed serving transition"
            )

        migrated = replace(
            saved,
            checkpoint=0,
            total_records=len(self.records),
            corpus_fingerprint=corpus_fingerprint,
            validation=None,
            phase=(
                MigrationPhase.FAILED
                if saved.phase is MigrationPhase.FAILED
                else MigrationPhase.BACKFILL
            ),
            resume_phase=(
                MigrationPhase.BACKFILL
                if saved.phase is MigrationPhase.FAILED
                else None
            ),
            error=saved.error if saved.phase is MigrationPhase.FAILED else None,
        )
        self._save(migrated)
        return migrated

    def _corpus_fingerprint(self) -> str:
        payload = [
            {
                "identity": record.storage_key,
                "text": semantic_input_for_record(record).text,
            }
            for record in self.records
        ]
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _check_namespace_dimensions(
        source: ModelNamespace,
        target: ModelNamespace,
    ) -> None:
        if source.model_name == target.model_name and source.dim != target.dim:
            raise ModelDimensionMismatchError(
                f"model {source.model_name!r} cannot migrate from dimension "
                f"{source.dim} to {target.dim}"
            )

    def _check_old_model_name(
        self,
        state: MigrationState,
        old_model_name: str | None,
    ) -> None:
        if old_model_name is not None and old_model_name != state.source.model_name:
            raise ReindexError(
                f"expected old model {state.source.model_name!r}, "
                f"got {old_model_name!r}"
            )

    def _reject_failed(self, state: MigrationState) -> None:
        if state.phase is MigrationPhase.FAILED:
            raise ReindexError(
                "migration is failed; call retry() before advancing"
            )

    def _fail(
        self,
        state: MigrationState,
        resume_phase: MigrationPhase,
        message: str,
    ) -> None:
        self._save(
            replace(
                state,
                phase=MigrationPhase.FAILED,
                error=message,
                resume_phase=resume_phase,
            )
        )

    def _save(self, state: MigrationState) -> None:
        self._lifecycle().save_migration(state)
        self._current_stage = state.phase.value

    @contextmanager
    def _transition_lock(self) -> Iterator[ModelLifecycleStore]:
        store = self._lifecycle()
        acquire = getattr(store, "acquire_transition_lock", None)
        if acquire is None:
            raise ReindexError(
                "active model transitions require a durable lifecycle lock"
            )
        try:
            lock = acquire()
        except Exception as exc:
            raise ReindexError(
                f"Failed to acquire active model transition lock: {exc}"
            ) from exc
        with lock:
            yield store

    @staticmethod
    def _compare_and_set_active_model(
        store: ModelLifecycleStore,
        *,
        expected: ActiveModelMetadata | None,
        active_model: ActiveModelMetadata,
        error_message: str,
    ) -> None:
        compare_and_set = getattr(store, "compare_and_set_active_model", None)
        if compare_and_set is not None:
            if not compare_and_set(expected, active_model):
                raise ReindexError(error_message)
            return
        if store.get_active_model() != expected:
            raise ReindexError(error_message)
        store.set_active_model(active_model)

    @staticmethod
    def _require_active_owner(
        store: ModelLifecycleStore,
        expected: ActiveModelMetadata,
        error_message: str,
    ) -> None:
        if store.get_active_model() != expected:
            raise ReindexError(error_message)

    def _require_lifecycle(self, operation: str) -> None:
        if self.lifecycle_store is None:
            raise ReindexError(
                f"{operation} is unavailable: a ModelLifecycleStore is required"
            )

    def _lifecycle(self) -> ModelLifecycleStore:
        if self.lifecycle_store is None:
            raise ReindexError("a ModelLifecycleStore is required")
        return self.lifecycle_store
