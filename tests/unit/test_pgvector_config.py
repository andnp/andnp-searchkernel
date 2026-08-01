import pytest

from searchkernel.adapters.stores.pgvector import (
    PGVectorFeatureSupport,
    PGVectorStore,
    bounded_scan_limits,
)


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


def test_iterative_scan_support_comes_from_server_extension_version() -> None:
    assert PGVectorFeatureSupport.from_extension_version("0.7.4").iterative_scan is False
    assert PGVectorFeatureSupport.from_extension_version("0.8.0").iterative_scan is True
    assert PGVectorFeatureSupport.from_extension_version(None).iterative_scan is False


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
