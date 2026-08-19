"""Integration tests for pgvector store adapter with psycopg3.

Mirrors test_pgvector_store.py to prove Psycopg3Connection works end-to-end
with PGVectorStore, PGKeywordStore, and the connection pool protocol.
"""

import os
from dataclasses import replace
from datetime import UTC, datetime
from typing import cast

import pytest
from psycopg.errors import NotNullViolation

from searchkernel.adapters.stores.pgvector import (
    PGGraphStore,
    PGKeywordStore,
    PGVectorStore,
    Psycopg3Connection,
    _vector_table_name,
    create_schema,
)
from searchkernel.domain import (
    GraphEdge,
    GraphNeighbor,
    Record,
    RecordIdentity,
    RecordStatus,
)
from searchkernel.indices import LocalRecordBackend
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
    create_schema(conn_pool)

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

    def test_vector_revision_persists_with_psycopg3(self, pg_conn, fixture_records):
        """Psycopg3 vector upserts persist the deterministic revision."""
        store = PGVectorStore(pg_conn)
        record = fixture_records[0]
        model_name = "revision-model"
        dim = 4

        store.upsert([record], model_name=model_name, dim=dim)

        table_name = _vector_table_name(model_name, dim)
        revision = pg_conn.execute_one(
            f'SELECT revision FROM "{table_name}" WHERE record_id = %s;',
            (record.storage_key,),
        )[0]

        assert revision

    def test_repeated_upsert_skips_unchanged_vector_with_psycopg3(
        self, pg_conn, fixture_records
    ):
        """Psycopg3 retries do not rewrite unchanged vectors or advance its epoch."""
        store = PGVectorStore(pg_conn)
        record = fixture_records[0]
        model_name = "psycopg3-idempotency"

        store.upsert([record], model_name=model_name, dim=4)
        before = store.epochs()
        changed = replace(record, embedding=[0.0, 1.0, 0.0, 0.0])
        store.upsert([changed], model_name=model_name, dim=4)

        assert store.epochs() == {
            "keyword": before["keyword"] + 1,
            "vector": before["vector"] + 1,
            "graph": before["graph"],
        }
        vector_epoch = store.vector_epoch()
        store.upsert([changed], model_name=model_name, dim=4)
        assert store.vector_epoch() == vector_epoch


class TestPsycopg3KeywordStore:
    """Tests for KeywordStore with psycopg3 connection pool."""

    def test_lexical_queries_match_local_backend(self, pg_conn, tmp_path):
        """Keep Psycopg3 lexical retrieval aligned with the local contract.

        Phrase, prefix, artifact, filter, empty-query, and tie ordering cases
        exercise the query shapes that must remain portable across backends.
        """
        now = datetime(2026, 1, 1, tzinfo=UTC)
        records = [
            Record(
                workspace_id=workspace,
                source_kind="note",
                source_id=source_id,
                title=title,
                body=body,
                uri=uri,
                status=status,
                created_at=now,
                updated_at=now,
                embedding=[1.0, 0.0, 0.0, 0.0],
            )
            for workspace, source_id, title, body, uri, status in (
                ("workspace-a", "phrase", "Alpha beta guide", "alpha beta phrase", "src/searchkernel/search.py", RecordStatus.ACTIVE),
                ("workspace-a", "prefix", "Alphabet", "alphabet soup", "", RecordStatus.ACTIVE),
                ("workspace-a", "symbol", "Parser", "parse_record implementation", "", RecordStatus.ACTIVE),
                ("workspace-a", "active", "Common active", "common token", "", RecordStatus.ACTIVE),
                ("workspace-b", "other-workspace", "Common other workspace", "common token", "", RecordStatus.ACTIVE),
                ("workspace-a", "archived", "Common archived", "common token", "", RecordStatus.ARCHIVED),
            )
        ]
        local = LocalRecordBackend(tmp_path / "local.db")
        local.index(records)
        keyword_store = PGKeywordStore(pg_conn)
        PGVectorStore(pg_conn).upsert(records, "lexical-parity", 4)
        keyword_store.index(records)

        cases = [
            ('"alpha beta"', None),
            ("alph*", None),
            ("src/searchkernel/search.py", None),
            ("parse_record", None),
            ("common", {"workspace_id": "workspace-a", "statuses": ["active"]}),
            ("", None),
        ]
        for query, filters in cases:
            local_keys = [hit.storage_key for hit in local.search_keyword(query, 10, filters)]
            pg_keys = [hit.storage_key for hit in keyword_store.search(query, 10, filters)]
            assert set(pg_keys) == set(local_keys)

        tie_records = [record for record in records if record.source_id in {"active", "other-workspace"}]
        assert [hit.storage_key for hit in keyword_store.search("common", 10)] == sorted(
            record.storage_key for record in tie_records
        )

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

    def test_repeated_keyword_index_skips_unchanged_projection_with_psycopg3(
        self, pg_conn, fixture_records
    ):
        """Psycopg3 retries preserve the projection and keyword epoch.

        The persisted record timestamp remains unchanged on the clean skip.
        """
        vector_store = PGVectorStore(pg_conn)
        keyword_store = PGKeywordStore(pg_conn)
        record = fixture_records[0]

        vector_store.upsert([record], model_name="psycopg3-keyword", dim=4)
        keyword_store.index([record])
        before = (
            keyword_store.keyword_epoch(),
            pg_conn.execute_one(
                "SELECT tsvector_body::text, updated_at FROM records "
                "WHERE record_id = %s;",
                (record.storage_key,),
            ),
        )

        keyword_store.index([record])

        assert keyword_store.keyword_epoch() == before[0]
        assert pg_conn.execute_one(
            "SELECT tsvector_body::text, updated_at FROM records "
            "WHERE record_id = %s;",
            (record.storage_key,),
        ) == before[1]

    def test_psycopg3_keyword_index_updates_changed_fields_and_duplicates(
        self, pg_conn, fixture_records
    ):
        """Psycopg3 updates changed lexical fields using the final duplicate."""
        vector_store = PGVectorStore(pg_conn)
        keyword_store = PGKeywordStore(pg_conn)
        original = fixture_records[0]
        vector_store.upsert([original], model_name="psycopg3-keyword", dim=4)
        keyword_store.index([original])
        changed = replace(
            original,
            title="psycopg3changedtitle",
            body="psycopg3changedbody",
            uri="docs/psycopg3changeduri",
            metadata={"marker": "psycopg3changedmetadata"},
        )
        vector_store.upsert([changed], model_name="psycopg3-keyword", dim=4)
        before = keyword_store.keyword_epoch()

        keyword_store.index([original, changed])

        assert keyword_store.keyword_epoch() == before + 1
        for term in (
            "psycopg3changedtitle",
            "psycopg3changedbody",
            "docs/psycopg3changeduri",
            "psycopg3changedmetadata",
        ):
            assert [hit.source_id for hit in keyword_store.search(term, 10)] == [
                original.source_id
            ]


class TestPsycopg3GraphStore:
    """Behavioral tests for graph writes through Psycopg3Connection."""

    def test_upsert_retrieves_edges_and_repeated_updates_replace_weight(self, pg_conn):
        """Repeated upserts keep one exact edge with its latest weight."""
        store = PGGraphStore(pg_conn)
        source = RecordIdentity("workspace-a", "note", "source")
        target = RecordIdentity("workspace-a", "note", "target")

        store.upsert_edges([GraphEdge(source, target, "related", 0.9)])
        store.upsert_edges([GraphEdge(source, target, "related", 0.2)])

        assert store.neighbors(source) == [
            GraphNeighbor(target, "related", pytest.approx(0.2))
        ]

    def test_empty_graph_batches_are_noops(self, pg_conn):
        """Empty graph batches leave rows and the graph epoch unchanged."""
        store = PGGraphStore(pg_conn)
        epoch = store.graph_epoch()

        store.upsert_edges([])
        store.delete_edges([])

        assert store.graph_epoch() == epoch
        assert pg_conn.execute_one("SELECT COUNT(*) FROM graph_edges;") == (0,)

    def test_delete_removes_only_the_exact_edge_in_its_workspace(self, pg_conn):
        """Exact deletion preserves other edge types and workspaces."""
        store = PGGraphStore(pg_conn)
        source_a = RecordIdentity("workspace-a", "note", "source")
        target_a = RecordIdentity("workspace-a", "note", "target")
        source_b = RecordIdentity("workspace-b", "note", "source")
        target_b = RecordIdentity("workspace-b", "note", "target")
        edges = [
            GraphEdge(source_a, target_a, "related", 0.9),
            GraphEdge(source_a, target_a, "links", 0.8),
            GraphEdge(source_b, target_b, "related", 0.7),
        ]
        store.upsert_edges(edges)

        store.delete_edges([edges[0]])

        assert store.neighbors(source_a) == [
            GraphNeighbor(target_a, "links", pytest.approx(0.8))
        ]
        assert store.neighbors(source_b) == [
            GraphNeighbor(target_b, "related", pytest.approx(0.7))
        ]

    def test_failed_batch_rolls_back_all_graph_changes(self, pg_conn):
        """A rejected batch leaves no partial edge or epoch mutation."""
        store = PGGraphStore(pg_conn)
        source = RecordIdentity("workspace-a", "note", "source")
        target = RecordIdentity("workspace-a", "note", "target")
        epoch = store.graph_epoch()

        with pytest.raises(NotNullViolation):
            store.upsert_edges(
                [
                    GraphEdge(source, target, "valid", 0.9),
                    GraphEdge(source, target, cast(str, None), 0.8),
                ]
            )

        assert store.graph_epoch() == epoch
        assert store.neighbors(source) == []


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
