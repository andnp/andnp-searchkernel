from datetime import UTC, datetime
from typing import ClassVar

import pytest

from searchkernel.adapters.stores import create_schema as exported_create_schema
from searchkernel.adapters.stores import pgvector
from searchkernel.adapters.stores.pgvector import (
    PGVectorFeatureSupport,
    PGVectorStore,
    Psycopg3Connection,
    bounded_scan_limits,
)
from searchkernel.domain import Record


def test_schema_bootstrap_has_public_compatibility_entry_points() -> None:
    """The schema bootstrap is public while the old private name remains valid."""
    assert exported_create_schema is pgvector.create_schema
    assert pgvector._create_schema is pgvector.create_schema


class _BulkCursor:
    def __init__(self) -> None:
        self.connection = None
        self.executed: list[tuple[object, object]] = []
        self.executemany_calls: list[tuple[object, object]] = []

    def execute(self, statement: object, params: object = None) -> None:
        self.executed.append((statement, params))

    def executemany(self, statement: object, params: object) -> None:
        self.executemany_calls.append((statement, params))

    def fetchall(self) -> list[object]:
        return []

    def close(self) -> None:
        pass


class _BulkConnection:
    def __init__(self, cursor: _BulkCursor) -> None:
        self.cursor_value = cursor

    def cursor(self) -> _BulkCursor:
        return self.cursor_value

    def commit(self) -> None:
        pass

    def rollback(self) -> None:
        pass


class _BulkPool:
    def __init__(self, connection: _BulkConnection) -> None:
        self.connection = connection

    def getconn(self) -> _BulkConnection:
        return self.connection

    def putconn(self, conn: _BulkConnection) -> None:
        pass


class _Psycopg3Pool(Psycopg3Connection):
    def __init__(self, connection: _BulkConnection) -> None:
        self.pool = _BulkPool(connection)


class _Cursor:
    def __init__(self, version: str | None):
        self.version = version
        self.statements: list[str] = []
        self.params: list[object] = []

    def execute(self, statement: object, params: object = None) -> None:
        self.statements.append(str(statement))
        self.params.append(params)

    def fetchone(self):
        if "vector_tables" in self.statements[-1]:
            return (1,)
        return (self.version,) if self.version is not None else None

    def fetchall(self):
        return [("workspace", "note", "one", 0.1)]

    def close(self) -> None:
        pass


class _Connection:
    def __init__(self, cursor: _Cursor):
        self.cursor_value = cursor

    def cursor(self) -> _Cursor:
        return self.cursor_value

    def commit(self) -> None:
        pass

    def rollback(self) -> None:
        pass


class _Pool:
    def __init__(self, cursor: _Cursor):
        self.connection = _Connection(cursor)

    def get_connection(self) -> _Connection:
        return self.connection

    def put_connection(self, conn: object) -> None:
        pass


class _SearchCursor:
    def __init__(self) -> None:
        self.statements: list[str] = []

    def execute(self, statement: object, params: object = None) -> None:
        self.statements.append(str(statement))

    def fetchone(self):
        if "vector_tables" in self.statements[-1]:
            return (1,)
        return ("0.8.0",)

    def fetchall(self):
        return [("workspace", "note", "one", 0.1)]

    def close(self) -> None:
        pass


class _SearchConnection:
    def __init__(self) -> None:
        self.cursor_value = _SearchCursor()
        self.rollback_calls = 0

    def cursor(self) -> _SearchCursor:
        return self.cursor_value

    def commit(self) -> None:
        pass

    def rollback(self) -> None:
        self.rollback_calls += 1


class _SearchPool:
    def __init__(self) -> None:
        self.connections = [_SearchConnection(), _SearchConnection()]
        self.next_connection = 0

    def get_connection(self) -> _SearchConnection:
        connection = self.connections[self.next_connection]
        self.next_connection = (self.next_connection + 1) % len(self.connections)
        return connection

    def put_connection(self, conn: object) -> None:
        pass


class _ConnectionPoolFactory:
    last_kwargs: ClassVar[dict[str, object]] = {}

    def __init__(self, dsn: str, **kwargs: object) -> None:
        type(self).last_kwargs = kwargs


class _Psycopg3PoolModule:
    ConnectionPool = _ConnectionPoolFactory


def test_psycopg3_pool_opens_before_first_checkout(monkeypatch) -> None:
    monkeypatch.setattr(pgvector, "psycopg_pool", _Psycopg3PoolModule)

    Psycopg3Connection("postgresql://example", min_connections=1, max_connections=3)

    assert _ConnectionPoolFactory.last_kwargs["open"] is True


def test_iterative_scan_support_comes_from_server_extension_version() -> None:
    assert PGVectorFeatureSupport.from_extension_version("0.7.4").iterative_scan is False
    assert PGVectorFeatureSupport.from_extension_version("0.8.0").iterative_scan is True
    assert PGVectorFeatureSupport.from_extension_version(None).iterative_scan is False


def test_upsert_submits_records_with_one_psycopg3_bulk_call() -> None:
    cursor = _BulkCursor()
    pool = _Psycopg3Pool(_BulkConnection(cursor))
    store = PGVectorStore(pool)
    now = datetime.now(UTC)
    records = [
        Record(
            source_kind="test",
            source_id=f"test:{index}",
            title=f"Title {index}",
            body=f"Body {index}",
            created_at=now,
            updated_at=now,
            metadata={"index": index},
            embedding=[0.1, 0.2, 0.3],
        )
        for index in range(2)
    ]

    store.upsert(records, model_name="test-model", dim=3)

    assert len(cursor.executemany_calls) == 1
    record_calls = [
        call for call in cursor.executemany_calls if "INSERT INTO records" in str(call[0])
    ]
    assert len(record_calls) == 1
    vector_calls = [
        call for call in cursor.executed if "RETURNING record_id" in str(call[0])
    ]
    assert len(vector_calls) == 1
    statement, rows = record_calls[0]
    assert "INSERT INTO records" in str(statement)
    assert isinstance(rows, list)
    typed_rows = [row for row in rows if isinstance(row, tuple)]
    assert len(typed_rows) == len(rows)
    assert len(rows) == len(records)
    assert typed_rows[0][0] == records[0].storage_key
    assert typed_rows[0][10] == '{"index": 0}'
    assert (
        sum("INSERT INTO records" in str(statement) for statement, _ in cursor.executed)
        == 0
    )


def test_bounded_scan_limits_grow_without_exceeding_hard_bounds() -> None:
    assert bounded_scan_limits(
        3,
        max_scan_tuples=10,
        max_scan_rounds=4,
        overfetch_multiplier=2.0,
    ) == [3, 6, 10]
    assert bounded_scan_limits(
        20,
        max_scan_tuples=10,
        max_scan_rounds=4,
        overfetch_multiplier=2.0,
    ) == [10]


def test_hnsw_settings_are_applied_only_when_server_supports_them() -> None:
    cursor = _Cursor("0.8.0")
    store = PGVectorStore(
        _Pool(cursor),
        hnsw_ef_search=120,
        hnsw_iterative_scan="relaxed_order",
        hnsw_max_scan_tuples=5000,
        hnsw_scan_mem_multiplier=2.0,
    )

    store._configure_hnsw(cursor)

    assert "SET LOCAL hnsw.ef_search = 120;" in cursor.statements
    assert "SET LOCAL hnsw.iterative_scan = 'relaxed_order';" in cursor.statements
    assert "SET LOCAL hnsw.max_scan_tuples = 5000;" in cursor.statements
    assert "SET LOCAL hnsw.scan_mem_multiplier = 2.0;" in cursor.statements


def test_hnsw_iterative_settings_fall_back_on_older_servers() -> None:
    cursor = _Cursor("0.7.4")
    store = PGVectorStore(_Pool(cursor))

    store._configure_hnsw(cursor)

    assert cursor.statements == [
        "SELECT extversion FROM pg_extension WHERE extname = 'vector';",
        "SET LOCAL hnsw.ef_search = 100;",
    ]


def test_filtered_search_reports_bounded_under_returning() -> None:
    cursor = _Cursor("0.7.4")
    store = PGVectorStore(
        _Pool(cursor),
        hnsw_max_scan_tuples=5,
        max_scan_rounds=4,
    )

    hits = store.search([1.0, 0.0], 3, model_name="model", dim=2)

    assert len(hits) == 1
    assert store.last_search_diagnostics == {
        "requested_k": 3,
        "returned": 1,
        "scan_rounds": 2,
        "scan_limit": 5,
        "scan_bound_hit": True,
        "under_returned": True,
        "iterative_scan": False,
        "extension_version": "0.7.4",
    }


def test_search_reuses_table_metadata_across_connections_and_transactions() -> None:
    """Stable table metadata is reused while transaction-local setup repeats."""
    pool = _SearchPool()
    store = PGVectorStore(pool)

    store.search([1.0, 0.0], 1, model_name="model", dim=2)
    store.search([1.0, 0.0], 1, model_name="model", dim=2)

    statements = [
        statement
        for connection in pool.connections
        for statement in connection.cursor_value.statements
    ]
    assert sum("FROM vector_tables" in statement for statement in statements) == 1
    assert sum("SET LOCAL hnsw.ef_search" in statement for statement in statements) == 2


def test_cached_dimension_mismatch_rolls_back_before_returning() -> None:
    """A skipped cached-dimension search returns its connection cleanly."""
    pool = _SearchPool()
    store = PGVectorStore(pool)
    store._vector_tables["model"] = (3, "vectors__model__3")

    assert store.search([1.0, 0.0], 1, model_name="model", dim=2) == []
    assert pool.connections[0].rollback_calls == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hnsw_ef_search": 0},
        {"hnsw_iterative_scan": "invalid"},
        {"hnsw_max_scan_tuples": 0},
        {"hnsw_scan_mem_multiplier": 0.5},
    ],
)
def test_hnsw_settings_validate_bounds(kwargs) -> None:
    with pytest.raises(ValueError):
        PGVectorStore(_Pool(_Cursor(None)), **kwargs)
