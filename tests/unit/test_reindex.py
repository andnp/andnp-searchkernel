import base64
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from searchkernel.domain import (
    ActiveModelMetadata,
    BackupMetadata,
    MigrationPhase,
    MigrationState,
    ModelDimensionMismatchError,
    ModelNamespace,
    Record,
    RecordHit,
    RollbackMetadata,
    ValidationResult,
)
from searchkernel.indices import LocalRecordBackend, LocalVectorStore
from searchkernel.runtime.reindex import ReindexError, ReindexRoutine
from searchkernel.utils.atomic_io import atomic_write_json


class FakeProvider:
    model_name = "target-model"
    dim = 3

    def __init__(
        self,
        fail_on_call: int | None = None,
        *,
        model_name: str = "target-model",
        dim: int = 3,
    ):
        self.calls = 0
        self.fail_on_call = fail_on_call
        self.model_name = model_name
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("embedding failed")
        return [
            [float(len(text)), *[float(value) for value in range(2, self.dim + 1)]]
            for text in texts
        ]


class FakeStore:
    def __init__(self):
        self.calls: list[tuple[list[Record], str, int]] = []

    def upsert(self, records: list[Record], model_name: str, dim: int) -> None:
        self.calls.append((records, model_name, dim))

    def search(
        self,
        query_vector: list[float],
        k: int,
        *,
        model_name: str,
        dim: int,
        filters: dict[str, object] | None = None,
    ) -> list[RecordHit | tuple[str, float]]:
        return []

    def delete(self, record_ids: list[str]) -> None:
        return None

    def epoch(self) -> int:
        return 0


def make_records() -> list[Record]:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    return [
        Record(
            source_kind="note",
            source_id=f"note-{index}",
            title=f"Note {index}",
            body=f"body {index}",
            created_at=timestamp,
            updated_at=timestamp,
            embedding=[9.0],
            embedding_model="old-model",
        )
        for index in range(3)
    ]


class LocalLifecycleStore:
    """Durable lifecycle capabilities backed by the real local SQLite store."""

    def __init__(self, root: Path, backend: LocalRecordBackend):
        self.root = root
        self.backend = backend
        self.active_path = root / "active-model.json"
        self.migration_path = root / "migration.json"
        self.backup_dir = root / "backups"
        self.fail_active_update = False
        root.mkdir(parents=True, exist_ok=True)

    def ensure_namespace(self, namespace: ModelNamespace) -> None:
        conn = self.backend.db_manager.get_connection()
        dimensions = {
            int(row[0])
            for row in conn.execute(
                """
                SELECT DISTINCT dim
                FROM local_vectors_v2
                WHERE encoder_namespace = ?
                """,
                (namespace.model_name,),
            )
        }
        if dimensions and dimensions != {namespace.dim}:
            existing_dim = next(iter(dimensions))
            raise ModelDimensionMismatchError(
                f"Dimension mismatch for model {namespace.model_name!r}: "
                f"expected {existing_dim}, got {namespace.dim}"
            )

    def delete_namespace(self, namespace: ModelNamespace) -> None:
        conn = self.backend.db_manager.get_connection()
        conn.execute(
            """
            DELETE FROM local_vectors_v2
            WHERE encoder_namespace = ? AND dim = ?
            """,
            (namespace.model_name, namespace.dim),
        )
        conn.commit()
        self.backend._vector_snapshots.clear()

    def validate_namespace(
        self,
        namespace: ModelNamespace,
        expected_records: int,
    ) -> ValidationResult:
        indexed_records = self.backend.vector_count(
            namespace.model_name, namespace.dim
        )
        errors = (
            ()
            if indexed_records == expected_records
            else (
                f"expected {expected_records} records, found {indexed_records}",
            )
        )
        return ValidationResult(
            namespace=namespace,
            expected_records=expected_records,
            indexed_records=indexed_records,
            errors=errors,
            checked_at=datetime.now(UTC).isoformat(),
        )

    def get_active_model(self) -> ActiveModelMetadata | None:
        if not self.active_path.exists():
            return None
        return ActiveModelMetadata.from_dict(
            json.loads(self.active_path.read_text())
        )

    def set_active_model(self, active_model: ActiveModelMetadata) -> None:
        if self.fail_active_update:
            raise RuntimeError("active model update failed")
        atomic_write_json(self.active_path, active_model.to_dict())

    def create_backup(self, namespace: ModelNamespace) -> BackupMetadata:
        conn = self.backend.db_manager.get_connection()
        rows = conn.execute(
            """
            SELECT storage_key, embedding, format_version, normalization_policy
            FROM local_vectors_v2
            WHERE encoder_namespace = ? AND dim = ?
            ORDER BY storage_key
            """,
            (namespace.model_name, namespace.dim),
        ).fetchall()
        backup_id = f"{namespace.model_name}-{namespace.dim}"
        backup_path = self.backup_dir / f"{backup_id}.json"
        atomic_write_json(
            backup_path,
            {
                "namespace": namespace.to_dict(),
                "vectors": [
                    {
                        "storage_key": row["storage_key"],
                        "embedding": base64.b64encode(row["embedding"]).decode(
                            "ascii"
                        ),
                        "format_version": row["format_version"],
                        "normalization_policy": row["normalization_policy"],
                    }
                    for row in rows
                ],
            },
        )
        return BackupMetadata(
            backup_id=backup_id,
            namespace=namespace,
            reference=str(backup_path),
            created_at=datetime.now(UTC).isoformat(),
        )

    def restore_backup(self, backup: BackupMetadata) -> RollbackMetadata:
        payload = json.loads(Path(backup.reference).read_text())
        namespace = ModelNamespace.from_dict(payload["namespace"])
        conn = self.backend.db_manager.get_connection()
        conn.executemany(
            """
            INSERT INTO local_vectors_v2 (
                storage_key, encoder_namespace, dim, embedding,
                format_version, normalization_policy
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(storage_key, encoder_namespace, dim) DO UPDATE SET
                embedding = excluded.embedding,
                format_version = excluded.format_version,
                normalization_policy = excluded.normalization_policy
            """,
            [
                (
                    vector["storage_key"],
                    namespace.model_name,
                    namespace.dim,
                    base64.b64decode(vector["embedding"]),
                    vector["format_version"],
                    vector["normalization_policy"],
                )
                for vector in payload["vectors"]
            ],
        )
        conn.commit()
        self.backend._vector_snapshots.clear()
        active = self.get_active_model()
        return RollbackMetadata(
            from_namespace=active.namespace if active is not None else namespace,
            to_namespace=namespace,
            backup_id=backup.backup_id,
            reason="restore local namespace backup",
            rolled_back_at=datetime.now(UTC).isoformat(),
        )

    def load_migration(self, migration_id: str) -> MigrationState | None:
        if not self.migration_path.exists():
            return None
        state = MigrationState.from_dict(
            json.loads(self.migration_path.read_text())
        )
        return state if state.migration_id == migration_id else None

    def save_migration(self, migration: MigrationState) -> None:
        atomic_write_json(self.migration_path, migration.to_dict())


def _local_records(count: int = 4) -> list[Record]:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    return [
        Record(
            source_kind="note",
            source_id=f"local-{index}",
            title=f"Local {index}",
            body=f"local body {index}",
            created_at=timestamp,
            updated_at=timestamp,
            embedding=[1.0, 0.0],
            embedding_model="old-model",
        )
        for index in range(count)
    ]


def _local_migration(
    tmp_path: Path, *, count: int = 4
) -> tuple[
    list[Record],
    LocalRecordBackend,
    LocalLifecycleStore,
    ModelNamespace,
]:
    root = tmp_path / "lifecycle"
    backend = LocalRecordBackend(root / "records.db")
    store = LocalLifecycleStore(root, backend)
    records = _local_records(count)
    source = ModelNamespace("old-model", 2)
    backend.upsert(records, source.model_name, source.dim)
    store.set_active_model(ActiveModelMetadata(source, generation=1))
    return records, backend, store, source


def _routine(
    records: list[Record],
    backend: LocalRecordBackend,
    store: LocalLifecycleStore,
    source: ModelNamespace,
    *,
    provider: FakeProvider | None = None,
    batch_size: int = 2,
    migration_id: str = "local-migration",
) -> ReindexRoutine:
    return ReindexRoutine(
        records,
        provider or FakeProvider(model_name="new-model"),
        LocalVectorStore(backend),
        batch_size=batch_size,
        lifecycle_store=store,
        migration_id=migration_id,
        source_namespace=source,
    )


def _assert_records_are_preserved(
    backend: LocalRecordBackend, records: list[Record]
) -> None:
    assert all(backend.hydrate_record(record.storage_key) is not None for record in records)


def test_unsupported_lifecycle_operations_are_explicit():
    routine = ReindexRoutine(make_records(), FakeProvider(), FakeStore())

    with pytest.raises(ReindexError, match="expand is unavailable"):
        routine.expand()
    with pytest.raises(ReindexError, match="flip is unavailable"):
        routine.flip()
    with pytest.raises(ReindexError, match="contract is unavailable"):
        routine.contract("old-model")
    with pytest.raises(ReindexError, match="rollback is unavailable"):
        routine.rollback("old-model")

    assert routine.stage == "init"


def test_backfill_writes_target_records_without_mutating_source_records():
    records = make_records()
    store = FakeStore()
    routine = ReindexRoutine(
        records,
        FakeProvider(),
        store,
        batch_size=2,
        truncate_dim=2,
    )

    progress = routine.backfill()

    assert progress.records_processed == 3
    assert progress.total_records == 3
    assert routine.stage == "backfill"
    assert [record.embedding for record in records] == [[9.0]] * 3
    assert [record.embedding for record in store.calls[0][0]] == [[6.0, 2.0]] * 2
    assert [record.embedding for record in store.calls[1][0]] == [[6.0, 2.0]]
    assert all(record.embedding_model == "target-model" for batch, _, _ in store.calls for record in batch)


def test_backfill_failure_preserves_completed_batches_for_retry():
    store = FakeStore()
    routine = ReindexRoutine(
        make_records(),
        FakeProvider(fail_on_call=2),
        store,
        batch_size=2,
    )

    with pytest.raises(ReindexError, match="Batch 1 failed"):
        routine.backfill()

    assert routine.stage == "backfill"
    assert len(store.calls) == 1


def test_backfill_is_retryable_and_idempotent_for_upsert_store():
    store = FakeStore()
    routine = ReindexRoutine(make_records(), FakeProvider(), store, batch_size=3)

    first = routine.backfill()
    second = routine.backfill()

    assert first.records_processed == second.records_processed == 3
    assert len(store.calls) == 2
    assert store.calls[0][0][0].embedding == store.calls[1][0][0].embedding


def test_empty_backfill_does_not_call_provider_or_index():
    provider = FakeProvider()
    store = FakeStore()
    routine = ReindexRoutine([], provider, store)

    progress = routine.backfill()

    assert progress.records_processed == 0
    assert provider.calls == 0
    assert store.calls == []


def test_local_lifecycle_keeps_old_namespace_during_target_backfill(tmp_path: Path):
    records, backend, store, source = _local_migration(tmp_path)
    routine = _routine(records, backend, store, source)

    assert routine.expand() is not None
    progress = routine.backfill()

    assert progress.complete
    active = store.get_active_model()
    assert active is not None
    assert active.namespace == source
    assert backend.vector_count(source.model_name, source.dim) == len(records)
    assert backend.vector_count("new-model", 3) == len(records)
    _assert_records_are_preserved(backend, records)


def test_backfill_failure_is_resumable_and_retryable(tmp_path: Path):
    records, backend, store, source = _local_migration(tmp_path, count=5)
    routine = _routine(
        records,
        backend,
        store,
        source,
        provider=FakeProvider(model_name="new-model", fail_on_call=2),
    )
    routine.expand()

    with pytest.raises(ReindexError, match="Batch 1 failed"):
        routine.backfill()

    failed = routine.state
    assert failed is not None
    assert failed.phase is MigrationPhase.FAILED
    assert failed.checkpoint == 2
    assert failed.resume_phase is MigrationPhase.BACKFILL
    assert backend.vector_count("new-model", 3) == 2

    assert routine.retry().phase is MigrationPhase.BACKFILL
    progress = routine.backfill()

    assert progress.complete
    assert routine.checkpoint == len(records)
    assert backend.vector_count(source.model_name, source.dim) == len(records)
    assert backend.vector_count("new-model", 3) == len(records)
    _assert_records_are_preserved(backend, records)


def test_checkpoint_survives_routine_restart(tmp_path: Path):
    records, backend, store, source = _local_migration(tmp_path, count=5)
    first = _routine(
        records,
        backend,
        store,
        source,
        provider=FakeProvider(model_name="new-model", fail_on_call=2),
    )
    first.expand()
    with pytest.raises(ReindexError):
        first.backfill()

    restarted_provider = FakeProvider(model_name="new-model")
    restarted = _routine(
        records,
        backend,
        store,
        source,
        provider=restarted_provider,
    )
    assert restarted.checkpoint == 2
    assert restarted.state is not None
    assert restarted.state.phase is MigrationPhase.FAILED

    restarted.retry()
    restarted.backfill()

    assert restarted_provider.calls == 2
    assert backend.vector_count("new-model", 3) == len(records)
    _assert_records_are_preserved(backend, records)


def test_validation_failure_does_not_flip_or_lose_source_data(tmp_path: Path):
    records, backend, store, source = _local_migration(tmp_path)
    routine = _routine(records, backend, store, source)
    routine.expand()
    routine.backfill()

    target_key = records[-1].storage_key
    conn = backend.db_manager.get_connection()
    conn.execute(
        """
        DELETE FROM local_vectors_v2
        WHERE storage_key = ? AND encoder_namespace = ? AND dim = ?
        """,
        (target_key, "new-model", 3),
    )
    conn.commit()

    with pytest.raises(ReindexError, match="target namespace validation failed"):
        routine.validate()

    failed = routine.state
    assert failed is not None
    assert failed.phase is MigrationPhase.FAILED
    assert failed.resume_phase is MigrationPhase.VALIDATE
    assert failed.validation is not None
    assert failed.validation.indexed_records == len(records) - 1
    active = store.get_active_model()
    assert active is not None
    assert active.namespace == source
    assert backend.vector_count(source.model_name, source.dim) == len(records)
    _assert_records_are_preserved(backend, records)


def test_active_model_flip_is_atomic_on_failure(tmp_path: Path):
    records, backend, store, source = _local_migration(tmp_path)
    routine = _routine(records, backend, store, source)
    routine.expand()
    routine.backfill()
    routine.validate()
    store.fail_active_update = True

    with pytest.raises(ReindexError, match="Failed to flip active model"):
        routine.flip()

    failed = routine.state
    assert failed is not None
    assert failed.phase is MigrationPhase.FAILED
    assert failed.resume_phase is MigrationPhase.FLIP
    active = store.get_active_model()
    assert active is not None
    assert active.namespace == source
    assert backend.vector_count(source.model_name, source.dim) == len(records)
    assert backend.vector_count("new-model", 3) == len(records)

    store.fail_active_update = False
    routine.retry()
    flipped = routine.flip()
    assert flipped.phase is MigrationPhase.FLIP
    active = store.get_active_model()
    assert active is not None
    assert active.namespace == ModelNamespace("new-model", 3)
    assert active.generation == 2


def test_contract_protects_old_namespace_cleanup(tmp_path: Path):
    records, backend, store, source = _local_migration(tmp_path)
    routine = _routine(records, backend, store, source)
    routine.run()

    with pytest.raises(ReindexError, match="expected old model"):
        routine.contract("wrong-model")

    assert backend.vector_count(source.model_name, source.dim) == len(records)
    assert backend.vector_count("new-model", 3) == len(records)

    complete = routine.contract(source.model_name)
    assert complete.phase is MigrationPhase.COMPLETE
    assert backend.vector_count(source.model_name, source.dim) == 0
    assert backend.vector_count("new-model", 3) == len(records)
    active = store.get_active_model()
    assert active is not None
    assert active.namespace == ModelNamespace("new-model", 3)
    _assert_records_are_preserved(backend, records)


def test_mixed_dimension_namespace_is_rejected(tmp_path: Path):
    records, backend, store, source = _local_migration(tmp_path)

    with pytest.raises(ModelDimensionMismatchError, match="Dimension mismatch"):
        backend.upsert(records, source.model_name, 3)

    routine = _routine(
        records,
        backend,
        store,
        source,
        provider=FakeProvider(model_name=source.model_name, dim=3),
    )
    with pytest.raises(ModelDimensionMismatchError, match="cannot migrate"):
        routine.expand()

    assert backend.vector_count(source.model_name, source.dim) == len(records)
    _assert_records_are_preserved(backend, records)


def test_rollback_before_flip_removes_only_unvalidated_target_namespace(
    tmp_path: Path,
):
    records, backend, store, source = _local_migration(tmp_path)
    routine = _routine(records, backend, store, source)
    routine.expand()
    routine.backfill()
    routine.validate()

    rolled_back = routine.rollback(source.model_name)

    assert rolled_back.phase is MigrationPhase.ROLLBACK
    assert rolled_back.backup is None
    active = store.get_active_model()
    assert active is not None
    assert active.namespace == source
    assert backend.vector_count(source.model_name, source.dim) == len(records)
    assert backend.vector_count("new-model", 3) == 0
    _assert_records_are_preserved(backend, records)


def test_rollback_after_flip_restores_backup_without_data_loss(tmp_path: Path):
    records, backend, store, source = _local_migration(tmp_path)
    routine = _routine(records, backend, store, source)
    routine.run(contract=True)

    completed = routine.state
    assert completed is not None
    assert completed.phase is MigrationPhase.COMPLETE
    assert completed.backup is not None
    assert Path(completed.backup.reference).exists()
    assert backend.vector_count(source.model_name, source.dim) == 0
    assert backend.vector_count("new-model", 3) == len(records)

    rolled_back = routine.rollback(source.model_name)

    assert rolled_back.phase is MigrationPhase.ROLLBACK
    assert rolled_back.rollback is not None
    assert rolled_back.rollback.to_namespace == source
    active = store.get_active_model()
    assert active is not None
    assert active.namespace == source
    assert backend.vector_count(source.model_name, source.dim) == len(records)
    assert backend.vector_count("new-model", 3) == 0
    _assert_records_are_preserved(backend, records)
