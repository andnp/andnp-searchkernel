"""Integration tests for pgvector store adapter with psycopg3.

Mirrors test_pgvector_store.py to prove Psycopg3Connection works end-to-end
with PGVectorStore, PGKeywordStore, and the connection pool protocol.
"""

import os
from datetime import UTC, datetime

import pytest

from searchkernel.adapters.stores.pgvector import (
    PGKeywordStore,
    PGVectorStore,
    Psycopg3Connection,
    _create_schema,
)
from searchkernel.domain import Record, RecordStatus
from tests.integration.conftest import pg_dsn_for_schema, pg_worker_schema


@pytest.fixture(scope="session")
def pg_dsn():
    """Use the Docker-backed DSN installed by the integration conftest."""
    dsn = os.environ.get("SEARCHKERNEL_PG_DSN")
    if not dsn:
        pytest.skip("SEARCHKERNEL_PG_DSN not set")
    return dsn


@pytest.fixture(scope="function")
def pg_conn(pg_dsn, request):
    """Create a test connection pool using Psycopg3Connection.

    Each xdist worker gets a private Postgres schema (pinned via search_path
    on the connection DSN), so this file's cleanup only touches this worker's
    own tables -- concurrent workers running the same file's tests can never
    collide, regardless of --dist mode.
    """
    schema = pg_worker_schema(request.config)
    scoped_dsn = pg_dsn_for_schema(pg_dsn, schema)

    bootstrap_pool = Psycopg3Connection(pg_dsn, min_connections=1, max_connections=1)
    bootstrap_conn = bootstrap_pool.get_connection()
    bootstrap_cursor = bootstrap_conn.cursor()
    bootstrap_cursor.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}";')
    bootstrap_conn.commit()
    bootstrap_cursor.close()
    bootstrap_pool.put_connection(bootstrap_conn)
    bootstrap_pool.close()

    conn_pool = Psycopg3Connection(scoped_dsn)
    _create_schema(conn_pool)

    # Clean slate for this worker's schema before every test.
    conn = conn_pool.get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT table_name FROM vector_tables;")
    for (table_name,) in cursor.fetchall():
        cursor.execute(f'DROP TABLE IF EXISTS "{table_name}";')

    cursor.execute("DELETE FROM vector_tables;")
    cursor.execute("DELETE FROM records;")
    cursor.execute("DELETE FROM graph_edges;")
    cursor.execute("DELETE FROM cache_store;")
    cursor.execute("UPDATE index_epoch SET epoch = 0;")
    conn.commit()
    cursor.close()
    conn_pool.put_connection(conn)

    yield conn_pool

    # Cleanup
    conn_pool.close()


@pytest.fixture
def fixture_records():
    """Create fixture records for testing."""
    now = datetime.now(UTC)
    return [
        Record(
            source_kind="test",
            source_id="test:1",
            title="Machine Learning Basics",
            body="Machine learning is a subset of AI. It enables systems to learn from data.",
            created_at=now,
            updated_at=now,
            metadata={"category": "ai"},
            uri="http://example.com/ml",
            status=RecordStatus.ACTIVE,
            embedding=[1.0, 0.0, 0.0, 0.0],
        ),
        Record(
            source_kind="test",
            source_id="test:2",
            title="Deep Learning Neural Networks",
            body="Neural networks are inspired by biological neurons. Deep learning uses many layers.",
            created_at=now,
            updated_at=now,
            metadata={"category": "ai"},
            uri="http://example.com/dl",
            status=RecordStatus.ACTIVE,
            embedding=[0.9, 0.1, 0.0, 0.0],
        ),
        Record(
            source_kind="test",
            source_id="test:3",
            title="Database Systems",
            body="Relational databases use SQL. PostgreSQL is a popular open-source database.",
            created_at=now,
            updated_at=now,
            metadata={"category": "database"},
            uri="http://example.com/db",
            status=RecordStatus.ACTIVE,
            embedding=[0.0, 0.0, 1.0, 0.0],
        ),
    ]


class TestPsycopg3VectorStore:
    """Tests for VectorStore with psycopg3 connection pool."""

    def test_upsert_and_search_with_psycopg3(self, pg_conn, fixture_records):
        """Test vector upsert and search work end-to-end with Psycopg3Connection."""
        store = PGVectorStore(pg_conn)

        # Upsert fixture records
        store.upsert(fixture_records, model_name="test-model", dim=4)

        # Query vector similar to record 1 and 2
        query_vec = [0.95, 0.05, 0.0, 0.0]

        # Search should return results
        results = store.search(query_vec, k=3, model_name="test-model", dim=4)

        assert len(results) > 0
        # First result should be close to the first records
        result_ids = [r[0] for r in results]
        assert "test:1" in result_ids or "test:2" in result_ids


class TestPsycopg3KeywordStore:
    """Tests for KeywordStore with psycopg3 connection pool."""

    def test_keyword_search_with_psycopg3(self, pg_conn, fixture_records):
        """Test full-text search works end-to-end with Psycopg3Connection."""
        vector_store = PGVectorStore(pg_conn)
        keyword_store = PGKeywordStore(pg_conn)

        # First upsert records (populates records table)
        vector_store.upsert(fixture_records, model_name="test-model", dim=4)

        # Index for keyword search
        keyword_store.index(fixture_records)

        # Search for "machine learning" should return top result
        results = keyword_store.search("machine learning", k=3)
        assert len(results) > 0

        # First result should be about machine learning
        top_id = results[0][0]
        top_record = next(r for r in fixture_records if r.source_id == top_id)
        assert "machine" in top_record.body.lower()


class TestPsycopg3ConnectionPool:
    """Tests for the Psycopg3Connection pool interface."""

    def test_get_put_connection(self, pg_conn):
        """Test get_connection and put_connection work correctly."""
        conn = pg_conn.get_connection()
        assert conn is not None

        cursor = conn.cursor()
        cursor.execute("SELECT 1;")
        result = cursor.fetchone()
        assert result == (1,)
        cursor.close()

        # Put the connection back
        pg_conn.put_connection(conn)

    def test_execute_method(self, pg_conn):
        """Test the execute method."""
        result = pg_conn.execute("SELECT 1;")
        assert len(result) == 1
        assert result[0] == (1,)

    def test_execute_one_method(self, pg_conn):
        """Test the execute_one method."""
        result = pg_conn.execute_one("SELECT 1;")
        assert result == (1,)

    def test_execute_one_returns_none_for_empty(self, pg_conn):
        """Test that execute_one returns None when no rows match."""
        pg_conn.execute("DELETE FROM records;")
        result = pg_conn.execute_one(
            "SELECT record_id FROM records LIMIT 1;"
        )
        assert result is None
