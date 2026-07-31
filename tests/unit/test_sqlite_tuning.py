import sqlite3

import pytest

from searchkernel.storage import DatabaseManager, SQLiteTuning


@pytest.mark.parametrize(
    "kwargs",
    [
        {"busy_timeout_ms": -1},
        {"busy_timeout_ms": 1.5},
        {"page_size": 1000},
        {"page_size": 4096.0},
        {"page_size": 131_072},
        {"cache_size": 0},
        {"cache_size": "100"},
        {"mmap_size": -1},
        {"mmap_size": 1.5},
        {"temp_store": "invalid"},
        {"temp_store": 2.0},
        {"checkpoint_policy": "invalid"},
        {"checkpoint_interval": -1},
        {"checkpoint_interval": 1.5},
    ],
)
def test_sqlite_tuning_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        SQLiteTuning(**kwargs)


def test_sqlite_tuning_applies_connection_pragmas(tmp_path):
    tuning = SQLiteTuning(
        busy_timeout_ms=1234,
        page_size=4096,
        cache_size=-100,
        mmap_size=1_048_576,
        temp_store="memory",
        checkpoint_policy="none",
    )
    manager = DatabaseManager(tmp_path / "records.db", tuning=tuning)
    conn = manager.get_connection()

    assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 1234
    assert conn.execute("PRAGMA page_size").fetchone()[0] == 4096
    assert conn.execute("PRAGMA cache_size").fetchone()[0] == -100
    assert conn.execute("PRAGMA temp_store").fetchone()[0] == 2
    assert conn.execute("PRAGMA wal_autocheckpoint").fetchone()[0] == 0
    assert manager.checkpoint() == (0, 0, 0)


def test_sqlite_tuning_accepts_sqlite_temp_store_values():
    assert SQLiteTuning(temp_store=0).temp_store_value == 0
    assert SQLiteTuning(temp_store=1).temp_store_value == 1
    assert SQLiteTuning(temp_store=2).temp_store_value == 2


def test_sqlite_tuning_checkpoint_policy_is_configurable(tmp_path):
    tuning = SQLiteTuning(
        checkpoint_policy="truncate",
        checkpoint_interval=17,
    )
    manager = DatabaseManager(tmp_path / "records.db", tuning=tuning)
    conn = manager.get_connection()

    assert conn.execute("PRAGMA wal_autocheckpoint").fetchone()[0] == 17
    result = manager.checkpoint()
    assert isinstance(result, tuple)
    assert len(result) == 3
    assert all(isinstance(value, int) for value in result)


def test_sqlite_tuning_preserves_sqlite_connection_usage(tmp_path):
    manager = DatabaseManager(tmp_path / "records.db")
    conn = manager.get_connection()

    assert isinstance(conn, sqlite3.Connection)
    conn.execute("CREATE TABLE tuning_probe (value TEXT)")
    conn.execute("INSERT INTO tuning_probe VALUES ('ok')")
    conn.commit()
    assert conn.execute("SELECT value FROM tuning_probe").fetchone()[0] == "ok"
