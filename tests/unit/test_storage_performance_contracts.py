from datetime import UTC, datetime

import pytest

from searchkernel.domain import Record, RecordStatus
from searchkernel.indices import LocalRecordBackend
from searchkernel.storage.db import DatabaseManager


def _record(source_id: str, body: str, indexed_text: str) -> Record:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    return Record(
        source_kind="note",
        source_id=source_id,
        title=source_id,
        body=body,
        indexed_text=indexed_text,
        created_at=timestamp,
        updated_at=timestamp,
        status=RecordStatus.ACTIVE,
        metadata={"tags": [source_id]},
    )


def test_repeated_unchanged_upsert_is_idempotent(tmp_path) -> None:
    """Repeated identical upserts preserve data and record epochs.

    The second write must not change the externally visible record state.
    """
    backend = LocalRecordBackend(tmp_path / "records.db")
    record = _record("repeat", "raw body", "findable text")

    backend.index([record])
    before_epoch = backend.epoch()
    backend.index([record])

    assert backend.epoch() == before_epoch
    assert [hit.source_id for hit in backend.search_keyword("findable", 10)] == [
        "repeat"
    ]
    hydrated = backend.hydrate_record(record.storage_key)
    assert hydrated is not None
    assert hydrated.body == "raw body"


def test_changed_upsert_replaces_fts_and_hydrated_values(tmp_path) -> None:
    """Changed indexed fields replace old lexical and stored values.

    Searches for old content must stop matching after the replacement.
    """
    backend = LocalRecordBackend(tmp_path / "records.db")
    record = _record("changed", "old body", "old searchable text")

    backend.index([record])
    record.body = "new body"
    record.indexed_text = "new searchable text"
    backend.index([record])

    assert backend.search_keyword("old", 10) == []
    assert [hit.source_id for hit in backend.search_keyword("new", 10)] == [
        "changed"
    ]
    hydrated = backend.hydrate_record(record.storage_key)
    assert hydrated is not None
    assert hydrated.body == "new body"
    assert hydrated.indexed_text == "new searchable text"


def test_indexed_records_persist_with_fts_results_after_reopen(tmp_path) -> None:
    """Indexed records remain searchable after reopening the database.

    Persistence must include both canonical storage and the lexical index.
    """
    database_path = tmp_path / "records.db"
    backend = LocalRecordBackend(database_path)
    record = _record("persistent", "stored body", "persistent vocabulary")
    backend.index([record])
    backend.close()

    reopened = LocalRecordBackend(database_path)

    assert [hit.source_id for hit in reopened.search_keyword("vocabulary", 10)] == [
        "persistent"
    ]
    hydrated = reopened.hydrate_record(record.storage_key)
    assert hydrated is not None
    assert hydrated.body == "stored body"


def test_closed_database_manager_invalidates_thread_connection(tmp_path) -> None:
    """Closing a database manager invalidates its current connection.

    A later manager can reopen the same persistent database successfully.
    """
    database_path = tmp_path / "records.db"
    manager = DatabaseManager(database_path)
    assert manager.get_connection() is manager.get_connection()

    manager.close()

    with pytest.raises(RuntimeError, match="database manager is closed"):
        manager.get_connection()
    reopened = DatabaseManager(database_path)
    assert reopened.get_connection().execute("SELECT 1").fetchone()[0] == 1
