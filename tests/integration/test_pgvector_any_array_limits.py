"""Evidence checks for oversized PostgreSQL ANY(array) candidate filters."""

from datetime import UTC, datetime

import pytest

from searchkernel.adapters.stores.pgvector import (
    PGKeywordStore,
    PGVectorStore,
    PostgresConnection,
    Psycopg3Connection,
    create_schema,
)
from searchkernel.domain import Record, RecordIdentity
from tests.integration.conftest import pg_dsn_for_schema, pg_worker_schema


@pytest.fixture(scope="function")
def pg_conn(pg_dsn, request, connection_type, pg_cleanup_executor):
    """Create an isolated connection pool using the selected supported driver."""
    schema = pg_worker_schema(request.config)
    scoped_dsn = pg_dsn_for_schema(pg_dsn, schema)
    try:
        bootstrap_pool = connection_type(pg_dsn, min_connections=1, max_connections=1)
        bootstrap_conn = bootstrap_pool.get_connection()
        bootstrap_cursor = bootstrap_conn.cursor()
        bootstrap_cursor.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}";')
        bootstrap_conn.commit()
        bootstrap_cursor.close()
        bootstrap_pool.put_connection(bootstrap_conn)
        bootstrap_pool.close()

        conn_pool = connection_type(scoped_dsn)
        create_schema(conn_pool)
    except Exception as exc:  # noqa: BLE001 - optional PostgreSQL is unavailable
        pytest.skip(f"Postgres unavailable for {connection_type.__name__}: {exc}")

    try:
        yield conn_pool
    finally:
        pg_cleanup_executor.submit(conn_pool.close)


@pytest.fixture(
    params=[PostgresConnection, Psycopg3Connection], ids=["psycopg2", "psycopg3"]
)
def connection_type(request):
    """Select each supported PostgreSQL driver for the evidence check."""
    return request.param


def test_oversized_any_array_candidate_filter(pg_conn, connection_type):
    """Report whether each driver accepts a large ANY(array) key list."""
    now = datetime(2026, 1, 1, tzinfo=UTC)
    record = Record(
        source_kind="note",
        source_id="kept",
        title="ANY array limit probe",
        body="A minimal searchable record.",
        created_at=now,
        updated_at=now,
        embedding=[1.0],
    )
    PGVectorStore(pg_conn).upsert([record], model_name="any-array-probe", dim=1)
    PGKeywordStore(pg_conn).index([record])

    candidate_keys = [
        RecordIdentity(None, "note", f"decoy-{index}").storage_key
        for index in range(99_999)
    ]
    candidate_keys.append(record.storage_key)

    try:
        results = PGKeywordStore(pg_conn).search(
            "minimal searchable", k=1, filters={"candidate_ids": candidate_keys}
        )
    except Exception as exc:  # noqa: BLE001 - report driver-specific limit failures
        pytest.fail(
            f"{connection_type.__name__} rejected {len(candidate_keys)} ANY(array) "
            f"keys with {type(exc).__name__}: {exc}"
        )

    assert [hit.storage_key for hit in results] == [record.storage_key], (
        f"{connection_type.__name__} returned an unexpected result for "
        f"{len(candidate_keys)} keys"
    )
