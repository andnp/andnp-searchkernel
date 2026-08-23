"""Unit-level coverage for PGKeywordStore's optional artifact-scorer hook."""

import re
from datetime import UTC, datetime

import pytest

from searchkernel.adapters.stores.pgvector import (
    PGKeywordStore,
    PGVectorStore,
    PostgresConnection,
    create_schema,
)
from searchkernel.domain import Record
from tests.integration.conftest import pg_dsn_for_schema, pg_worker_schema


class _StubIdentifierScorer:
    """Recognizes Jira-key-shaped queries (e.g. PROJ-1234), not file paths."""

    _PATTERN = re.compile(r"^[A-Za-z]+-\d+$")

    def looks_like_identifier_query(self, query: str) -> bool:
        return bool(self._PATTERN.match(query.strip()))

    def identifier_tokens(self, query: str) -> list[str]:
        return [query.strip()] if self.looks_like_identifier_query(query) else []

    def score(
        self,
        query: str,
        *,
        title: str,
        body: str,
        indexed_text: str | None,
        headers: str,
        uri: str,
    ) -> float:
        return 100.0 if query.strip().lower() in headers.lower() else 0.0


@pytest.fixture
def pg_conn(pg_dsn, request, pg_cleanup_executor):
    schema = pg_worker_schema(request.config) + "_artifact_scorer"
    scoped_dsn = pg_dsn_for_schema(pg_dsn, schema)

    bootstrap_pool = PostgresConnection(pg_dsn, min_connections=1, max_connections=1)
    bootstrap_conn = bootstrap_pool.get_connection()
    bootstrap_cursor = bootstrap_conn.cursor()
    bootstrap_cursor.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}";')
    bootstrap_conn.commit()
    bootstrap_cursor.close()
    bootstrap_pool.put_connection(bootstrap_conn)
    bootstrap_pool.close()

    conn_pool = PostgresConnection(scoped_dsn)
    create_schema(conn_pool)

    conn = conn_pool.get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT table_name FROM vector_tables;")
    for (table_name,) in cursor.fetchall():
        cursor.execute(f'DROP TABLE IF EXISTS "{table_name}";')
    cursor.execute("DELETE FROM vector_tables;")
    cursor.execute("DELETE FROM records;")
    conn.commit()
    cursor.close()
    conn_pool.put_connection(conn)

    yield conn_pool

    pg_cleanup_executor.submit(conn_pool.close)


@pytest.fixture
def identifier_records() -> list[Record]:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    return [
        Record(
            source_kind="note",
            source_id="ranked-higher-by-relevance",
            title="Deployment notes",
            body="PROJ-1234 " * 20,
            created_at=timestamp,
            updated_at=timestamp,
            embedding=[1.0, 0.0],
        ),
        Record(
            source_kind="note",
            source_id="tagged-with-identifier",
            title="Unrelated note",
            body="proj-1234 mentioned once",
            metadata={"tags": ["PROJ-1234"]},
            created_at=timestamp,
            updated_at=timestamp,
            embedding=[0.9, 0.1],
        ),
    ]


def test_search_without_scorer_matches_base_relevance_order(
    pg_conn, identifier_records
) -> None:
    PGVectorStore(pg_conn).upsert(identifier_records, "artifact-scorer", 2)
    keyword_store = PGKeywordStore(pg_conn)
    keyword_store.index(identifier_records)

    hits = keyword_store.search("PROJ-1234", 1)

    assert len(hits) == 1
    assert hits[0].source_id == "ranked-higher-by-relevance"


def test_search_with_scorer_boosts_and_reorders_identifier_match(
    pg_conn, identifier_records
) -> None:
    PGVectorStore(pg_conn).upsert(identifier_records, "artifact-scorer", 2)
    keyword_store = PGKeywordStore(pg_conn, artifact_scorer=_StubIdentifierScorer())
    keyword_store.index(identifier_records)

    hits = keyword_store.search("PROJ-1234", 1)

    assert len(hits) == 1
    assert hits[0].source_id == "tagged-with-identifier"


def test_search_with_scorer_leaves_non_identifier_query_unaffected(
    pg_conn, identifier_records
) -> None:
    PGVectorStore(pg_conn).upsert(identifier_records, "artifact-scorer", 2)
    keyword_store = PGKeywordStore(pg_conn, artifact_scorer=_StubIdentifierScorer())
    keyword_store.index(identifier_records)

    hits = keyword_store.search("deployment notes", 1)

    assert len(hits) == 1
    assert hits[0].source_id == "ranked-higher-by-relevance"
