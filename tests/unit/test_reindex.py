from datetime import UTC, datetime

import pytest

from searchkernel.domain import Record, RecordHit
from searchkernel.runtime.reindex import ReindexError, ReindexRoutine


class FakeProvider:
    model_name = "target-model"
    dim = 3

    def __init__(self, fail_on_call: int | None = None):
        self.calls = 0
        self.fail_on_call = fail_on_call

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("embedding failed")
        return [[float(len(text)), 2.0, 3.0] for text in texts]


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
