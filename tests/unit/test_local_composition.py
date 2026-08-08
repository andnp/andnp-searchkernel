import sqlite3
from datetime import UTC, datetime

import pytest

from searchkernel import Record, RecordSearchOutcome, build_local_record_kernel


class _FakeEmbeddingProvider:
    model_name = "fake"
    dim = 2

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


@pytest.fixture
def local_composition(tmp_path):
    with build_local_record_kernel(
        tmp_path / "records.db",
        embedding_provider=_FakeEmbeddingProvider(),
    ) as composition:
        yield composition


@pytest.mark.asyncio
async def test_public_local_composition_indexes_and_searches_records(
    local_composition, monkeypatch
) -> None:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    record = Record(
        workspace_id="workspace",
        source_kind="note",
        source_id="record-1",
        title="Local record",
        body="canonical local search content",
        created_at=timestamp,
        updated_at=timestamp,
    )
    composition = local_composition

    composition.keyword_store.index([record])
    batch_calls = []
    hydrate_records = composition.backend.hydrate_records

    def tracked_hydrate_records(identities):
        batch_calls.append(list(identities))
        return hydrate_records(identities)

    monkeypatch.setattr(composition.backend, "hydrate_records", tracked_hydrate_records)

    outcome = await composition.kernel.search("canonical", limit=1)

    assert isinstance(outcome, RecordSearchOutcome)
    assert [result.record.storage_key for result in outcome.results] == [
        record.storage_key
    ]
    assert outcome.results[0].record.body == record.body
    assert batch_calls == [[record.identity]]
    hydrated = composition.backend.hydrate_record(record.storage_key)
    assert hydrated is not None
    assert hydrated.storage_key == record.storage_key
    assert hydrated.body == record.body


def test_local_composition_closes_owned_database(tmp_path) -> None:
    composition = build_local_record_kernel(
        tmp_path / "records.db",
        embedding_provider=_FakeEmbeddingProvider(),
    )
    connection = composition.backend.db_manager.get_connection()

    composition.close()

    with pytest.raises(sqlite3.ProgrammingError):
        connection.execute("SELECT 1")
