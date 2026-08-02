import base64
import fcntl
import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
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
    RollbackMetadata,
    ValidationResult,
)
from searchkernel.indices import LocalRecordBackend, LocalVectorStore
from searchkernel.ports.reindex import RecordBatch
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
        self.embedded_texts: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        self.embedded_texts.append(texts)
        if self.calls == self.fail_on_call:
            raise RuntimeError("embedding failed")
        return [
            [float(len(text)), *[float(value) for value in range(2, self.dim + 1)]]
            for text in texts
        ]


class FakeRecordSource:
    total_records: int | None

    def __init__(self, records: list[Record], total_records: int | None = None):
        self.records = records
        self.total_records = total_records
        self.calls: list[str | None] = []

    def fetch_batch(self, cursor: str | None, limit: int) -> RecordBatch:
        self.calls.append(cursor)
        start = 0 if cursor is None else int(cursor.removeprefix("cursor-"))
        batch = self.records[start : start + limit]
        end = start + len(batch)
        next_cursor = None if end == len(self.records) else f"cursor-{end}"
        return RecordBatch(records=batch, next_cursor=next_cursor)


class LocalLifecycleStore:
    """Durable lifecycle capabilities backed by the real local SQLite store."""

    def __init__(self, root: Path, backend: LocalRecordBackend):
        self.root = root
        self.backend = backend
        self.active_path = root / "active-model.json"
        self.migration_dir = root / "migrations"
        self.backup_dir = root / "backups"
        self.fail_active_update = False
        self.transition_path = root / "active-model-transition.lock"
        self._transition_mutex = threading.RLock()
        self._transition_depth = 0
        self._transition_file = None
        root.mkdir(parents=True, exist_ok=True)
        self.migration_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)

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
        with self._active_model_lock():
            atomic_write_json(self.active_path, active_model.to_dict())

    def compare_and_set_active_model(
        self,
        expected: ActiveModelMetadata | None,
        active_model: ActiveModelMetadata,
    ) -> bool:
        with self._active_model_lock():
            if self.get_active_model() != expected:
                return False
            self.set_active_model(active_model)
            return True

    @contextmanager
    def acquire_transition_lock(self) -> Iterator[None]:
        with self._active_model_lock():
            yield

    @contextmanager
    def _active_model_lock(self) -> Iterator[None]:
        with self._transition_mutex:
            if self._transition_depth == 0:
                lock_file = self.transition_path.open("a+")
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                self._transition_file = lock_file
            self._transition_depth += 1
            try:
                yield
            finally:
                self._transition_depth -= 1
                if self._transition_depth == 0:
                    lock_file = self._transition_file
                    if lock_file is not None:
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                        lock_file.close()
                    self._transition_file = None

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
        migration_path = self.migration_dir / f"{migration_id}.json"
        if not migration_path.exists():
            return None
        state = MigrationState.from_dict(
            json.loads(migration_path.read_text())
        )
        return state if state.migration_id == migration_id else None

    def save_migration(self, migration: MigrationState) -> None:
        atomic_write_json(
            self.migration_dir / f"{migration.migration_id}.json",
            migration.to_dict(),
        )


class CoordinatedLifecycleStore(LocalLifecycleStore):
    def __init__(
        self,
        root: Path,
        backend: LocalRecordBackend,
        read_barrier: threading.Barrier,
    ):
        super().__init__(root, backend)
        self.read_barrier = read_barrier
        self.block_unlocked_reads = False

    def get_active_model(self) -> ActiveModelMetadata | None:
        active = super().get_active_model()
        if self.block_unlocked_reads and self._transition_depth == 0:
            self.read_barrier.wait(timeout=5)
        return active


class ContractOwnershipStore(LocalLifecycleStore):
    def __init__(self, root: Path, backend: LocalRecordBackend):
        super().__init__(root, backend)
        self.active_model_after_backup: ActiveModelMetadata | None = None
        self.active_model_after_source_write: ActiveModelMetadata | None = None

    def create_backup(self, namespace: ModelNamespace) -> BackupMetadata:
        backup = super().create_backup(namespace)
        if self.active_model_after_backup is not None:
            atomic_write_json(
                self.active_path,
                self.active_model_after_backup.to_dict(),
            )
        return backup

    def set_active_model(self, active_model: ActiveModelMetadata) -> None:
        super().set_active_model(active_model)
        if self.active_model_after_source_write is not None:
            atomic_write_json(
                self.active_path,
                self.active_model_after_source_write.to_dict(),
            )


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


def test_cursor_source_checkpoints_after_each_committed_batch(
    tmp_path: Path,
):
    records, backend, store, source_namespace = _local_migration(tmp_path, count=4)
    source = FakeRecordSource(records)
    routine = ReindexRoutine(
        source,
        FakeProvider(model_name="new-model"),
        LocalVectorStore(backend),
        batch_size=2,
        lifecycle_store=store,
        migration_id="cursor-migration",
        source_namespace=source_namespace,
    )

    routine.expand()
    progress = routine.backfill()

    assert source.calls == [None, "cursor-2"]
    assert progress.complete
    assert progress.total_records == 4
    assert routine.checkpoint == 4
    saved = routine.state
    assert saved is not None
    assert saved.total_records == 4
    assert backend.vector_count("new-model", 3) == 4


def test_cursor_source_resume_uses_durable_cursor_after_batch_failure(
    tmp_path: Path,
):
    records, backend, store, source_namespace = _local_migration(tmp_path, count=4)
    first_source = FakeRecordSource(records)
    first = ReindexRoutine(
        first_source,
        FakeProvider(model_name="new-model", fail_on_call=2),
        LocalVectorStore(backend),
        batch_size=2,
        lifecycle_store=store,
        migration_id="cursor-migration",
        source_namespace=source_namespace,
    )
    first.expand()

    with pytest.raises(ReindexError, match="Batch 1 failed"):
        first.backfill()

    failed = first.state
    assert failed is not None
    assert failed.checkpoint == 2
    assert backend.vector_count("new-model", 3) == 2

    restarted_source = FakeRecordSource(records)
    restarted = ReindexRoutine(
        restarted_source,
        FakeProvider(model_name="new-model"),
        LocalVectorStore(backend),
        batch_size=2,
        lifecycle_store=store,
        migration_id="cursor-migration",
        source_namespace=source_namespace,
    )
    restarted.retry()
    progress = restarted.backfill()

    assert restarted_source.calls == ["cursor-2"]
    assert progress.complete
    assert restarted.checkpoint == 4
    assert backend.vector_count("new-model", 3) == 4


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


def test_lifecycle_backfill_embeds_indexed_text_with_body_fallback(tmp_path: Path):
    records, backend, store, source = _local_migration(tmp_path, count=2)
    records[0].indexed_text = "indexed text"
    provider = FakeProvider(model_name="new-model")
    routine = _routine(records, backend, store, source, provider=provider)

    routine.expand()
    routine.backfill()

    assert provider.embedded_texts == [["indexed text", "local body 1"]]


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


def test_checkpoint_resumes_unchanged_corpus_after_restart(tmp_path: Path):
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
    assert restarted.state.corpus_fingerprint is not None

    restarted.retry()
    restarted.backfill()

    assert restarted_provider.calls == 2
    assert backend.vector_count("new-model", 3) == len(records)
    _assert_records_are_preserved(backend, records)


def test_checkpoint_rejects_changed_indexed_text(tmp_path: Path):
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

    records[2].indexed_text = "updated semantic text"
    restarted_provider = FakeProvider(model_name="new-model")
    restarted = _routine(
        records,
        backend,
        store,
        source,
        provider=restarted_provider,
    )

    with pytest.raises(ReindexError, match="corpus identity"):
        restarted.backfill()

    assert restarted_provider.calls == 0
    assert backend.vector_count("new-model", 3) == 2


def test_checkpoint_keeps_body_fallback_identity_stable(tmp_path: Path):
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

    records[2].indexed_text = ""
    restarted_provider = FakeProvider(model_name="new-model")
    restarted = _routine(
        records,
        backend,
        store,
        source,
        provider=restarted_provider,
    )

    assert restarted.checkpoint == 2
    restarted.retry()
    progress = restarted.backfill()

    assert progress.complete
    assert restarted_provider.embedded_texts == [
        ["local body 2", "local body 3"],
        ["local body 4"],
    ]


def test_checkpoint_rejects_reordered_corpus(tmp_path: Path):
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

    reordered = [records[1], records[0], *records[2:]]
    restarted_provider = FakeProvider(model_name="new-model")
    restarted = _routine(
        reordered,
        backend,
        store,
        source,
        provider=restarted_provider,
    )

    with pytest.raises(ReindexError, match="corpus identity"):
        restarted.backfill()

    assert restarted_provider.calls == 0
    assert backend.vector_count("new-model", 3) == 2


def test_checkpoint_rejects_replaced_record_with_same_corpus_length(
    tmp_path: Path,
):
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

    replaced = [
        *records[:2],
        replace(
            records[2],
            source_id="replacement-2",
            body="replacement body",
        ),
        *records[3:],
    ]
    restarted_provider = FakeProvider(model_name="new-model")
    restarted = _routine(
        replaced,
        backend,
        store,
        source,
        provider=restarted_provider,
    )

    with pytest.raises(ReindexError, match="corpus identity"):
        restarted.backfill()

    assert restarted_provider.calls == 0
    assert backend.vector_count("new-model", 3) == 2


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


def test_concurrent_flips_serialize_active_model_ownership(tmp_path: Path):
    records = _local_records()
    root = tmp_path / "lifecycle"
    backend = LocalRecordBackend(root / "records.db")
    source = ModelNamespace("old-model", 2)
    backend.upsert(records, source.model_name, source.dim)
    initial_store = LocalLifecycleStore(root, backend)
    initial_store.set_active_model(ActiveModelMetadata(source, generation=1))

    read_barrier = threading.Barrier(2)
    store_b = CoordinatedLifecycleStore(root, backend, read_barrier)
    store_c = CoordinatedLifecycleStore(root, backend, read_barrier)
    routine_b = _routine(
        records,
        backend,
        store_b,
        source,
        provider=FakeProvider(model_name="model-b"),
        migration_id="migration-b",
    )
    routine_c = _routine(
        records,
        backend,
        store_c,
        source,
        provider=FakeProvider(model_name="model-c"),
        migration_id="migration-c",
    )
    for routine in (routine_b, routine_c):
        routine.expand()
        routine.backfill()
        routine.validate()

    store_b.block_unlocked_reads = True
    store_c.block_unlocked_reads = True
    results: list[tuple[str, MigrationState | str]] = []

    def flip(routine: ReindexRoutine) -> None:
        try:
            results.append(("success", routine.flip()))
        except ReindexError as exc:
            results.append(("error", str(exc)))

    first = threading.Thread(target=flip, args=(routine_b,))
    second = threading.Thread(target=flip, args=(routine_c,))
    first.start()
    second.start()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not first.is_alive()
    assert not second.is_alive()
    assert [kind for kind, _ in results].count("success") == 1
    assert [kind for kind, _ in results].count("error") == 1
    active = initial_store.get_active_model()
    assert active is not None
    assert active.namespace in {
        ModelNamespace("model-b", 3),
        ModelNamespace("model-c", 3),
    }


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


def test_contract_revalidates_active_ownership_before_cleanup(tmp_path: Path):
    records = _local_records()
    root = tmp_path / "lifecycle"
    backend = LocalRecordBackend(root / "records.db")
    store = ContractOwnershipStore(root, backend)
    source = ModelNamespace("old-model", 2)
    backend.upsert(records, source.model_name, source.dim)
    store.set_active_model(ActiveModelMetadata(source, generation=1))
    routine = _routine(records, backend, store, source)
    routine.run()

    foreign = ActiveModelMetadata(
        ModelNamespace("other-model", 3),
        generation=9,
    )
    store.active_model_after_backup = foreign

    with pytest.raises(ReindexError, match="Failed to contract old model"):
        routine.contract(source.model_name)

    assert store.get_active_model() == foreign
    assert backend.vector_count(source.model_name, source.dim) == len(records)
    assert backend.vector_count("new-model", 3) == len(records)


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


def test_rollback_revalidates_active_ownership_before_cleanup(tmp_path: Path):
    records = _local_records()
    root = tmp_path / "lifecycle"
    backend = LocalRecordBackend(root / "records.db")
    store = ContractOwnershipStore(root, backend)
    source = ModelNamespace("old-model", 2)
    backend.upsert(records, source.model_name, source.dim)
    store.set_active_model(ActiveModelMetadata(source, generation=1))
    routine = _routine(records, backend, store, source)
    routine.run(contract=True)

    foreign = ActiveModelMetadata(
        ModelNamespace("other-model", 3),
        generation=9,
    )
    store.active_model_after_source_write = foreign

    with pytest.raises(ReindexError, match="Failed to roll back migration"):
        routine.rollback(source.model_name)

    assert store.get_active_model() == foreign
    assert backend.vector_count(source.model_name, source.dim) == len(records)
    assert backend.vector_count("new-model", 3) == len(records)
