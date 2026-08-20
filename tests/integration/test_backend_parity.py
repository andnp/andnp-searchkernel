"""Local/Postgres parity checks for canonical record retrieval."""

import os
import re
from datetime import UTC, datetime

import pytest

from searchkernel.adapters.stores.pgvector import (
    PGKeywordStore,
    PGVectorStore,
    PostgresConnection,
    create_schema,
)
from searchkernel.domain import Record, RecordIdentity, RecordStatus
from searchkernel.indices import LocalRecordBackend
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
def parity_records() -> list[Record]:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    return [
        Record(
            workspace_id="workspace-a",
            source_kind="note",
            source_id="shared",
            title="Deployment incident note",
            body="deployment incident runbook",
            created_at=timestamp,
            updated_at=timestamp,
            embedding=[1.0, 0.0],
        ),
        Record(
            workspace_id="workspace-a",
            source_kind="commit",
            source_id="shared",
            title="Deployment incident commit",
            body="deployment incident change",
            created_at=timestamp,
            updated_at=timestamp,
            embedding=[0.9, 0.1],
        ),
        Record(
            workspace_id="workspace-b",
            source_kind="note",
            source_id="shared",
            title="Deployment incident other workspace",
            body="deployment incident unrelated workspace",
            created_at=timestamp,
            updated_at=timestamp,
            embedding=[1.0, 0.0],
        ),
        Record(
            workspace_id="workspace-a",
            source_kind="note",
            source_id="archived",
            title="Deployment incident archive",
            body="deployment incident archived",
            created_at=timestamp,
            updated_at=timestamp,
            status=RecordStatus.ARCHIVED,
            embedding=[1.0, 0.0],
        ),
    ]


@pytest.fixture
def parity_backends(tmp_path, request):
    dsn = os.environ.get("SEARCHKERNEL_PG_DSN")
    if not dsn:
        pytest.skip("SEARCHKERNEL_PG_DSN not set")

    schema = pg_worker_schema(request.config) + "_parity"
    bootstrap_pool = PostgresConnection(dsn, min_connections=1, max_connections=1)
    bootstrap_connection = bootstrap_pool.get_connection()
    bootstrap_cursor = bootstrap_connection.cursor()
    bootstrap_cursor.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}";')
    bootstrap_connection.commit()
    bootstrap_cursor.close()
    bootstrap_pool.put_connection(bootstrap_connection)
    bootstrap_pool.close()

    scoped_dsn = pg_dsn_for_schema(dsn, schema)
    pg_pool = PostgresConnection(scoped_dsn)
    create_schema(pg_pool)
    connection = pg_pool.get_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT table_name FROM vector_tables;")
    for (table_name,) in cursor.fetchall():
        cursor.execute(f'DROP TABLE IF EXISTS "{table_name}";')
    cursor.execute("DELETE FROM vector_tables;")
    cursor.execute("DELETE FROM records;")
    cursor.execute("DELETE FROM graph_edges;")
    cursor.execute("DELETE FROM cache_store;")
    cursor.execute("UPDATE index_epoch SET epoch = 0;")
    connection.commit()
    cursor.close()
    pg_pool.put_connection(connection)

    try:
        yield LocalRecordBackend(tmp_path / "records.db"), PGKeywordStore(pg_pool), PGVectorStore(pg_pool)
    finally:
        pg_pool.close()


def _keys(hits) -> set[str]:
    return {hit.storage_key for hit in hits}


def test_keyword_retrieval_preserves_workspace_and_source_kind_parity(
    parity_records, parity_backends
) -> None:
    local, pg_keyword, pg_vector = parity_backends
    local.upsert(parity_records, "parity", 2)
    pg_vector.upsert(parity_records, "parity", 2)

    filters = {"workspace_id": "workspace-a", "source_kinds": ["note"]}
    local_hits = local.search_keyword("deployment incident", 10, filters)
    pg_hits = pg_keyword.search("deployment incident", 10, filters)
    expected = {
        parity_records[0].storage_key,
    }

    assert _keys(local_hits) == expected
    assert _keys(pg_hits) == expected
    assert _keys(local.search_keyword("deployment incident", 10)) == (
        {
            parity_records[0].storage_key,
            parity_records[1].storage_key,
            parity_records[2].storage_key,
        }
    )


def test_keyword_compound_filter_preserves_eligible_storage_key_parity(
    parity_records, parity_backends
) -> None:
    local, pg_keyword, pg_vector = parity_backends
    local.upsert(parity_records, "parity", 2)
    pg_vector.upsert(parity_records, "parity", 2)
    filters = {
        "statuses": [RecordStatus.ACTIVE.value],
        "workspace_id": "workspace-a",
        "source_kinds": ["note"],
        "candidate_ids": [record.identity for record in parity_records],
    }
    local_hits = local.search_keyword("deployment incident", 10, filters)
    pg_hits = pg_keyword.search("deployment incident", 10, filters)

    assert _keys(local_hits) == _keys(pg_hits) == {parity_records[0].storage_key}


def test_vector_retrieval_preserves_candidate_identity_parity(
    parity_records, parity_backends
) -> None:
    local, _pg_keyword, pg_vector = parity_backends
    local.upsert(parity_records, "parity", 2)
    pg_vector.upsert(parity_records, "parity", 2)
    candidate = parity_records[1].identity

    filters = {"candidate_ids": [candidate]}
    local_hits = local.search_vector(
        [1.0, 0.0], 10, model_name="parity", dim=2, filters=filters
    )
    pg_hits = pg_vector.search(
        [1.0, 0.0], 10, model_name="parity", dim=2, filters=filters
    )

    assert _keys(local_hits) == {candidate.storage_key}
    assert _keys(pg_hits) == {candidate.storage_key}
    assert local_hits[0].identity == RecordIdentity("workspace-a", "commit", "shared")
    assert pg_hits[0].identity == local_hits[0].identity


def test_keyword_artifact_scorer_boosts_identifier_query_in_parity(
    parity_backends,
) -> None:
    """A shared identifier scorer reorders both backends the same way.

    Base relevance alone would rank the record with more raw term
    occurrences first; the scorer instead promotes the record whose
    metadata carries the identifier as a tag, and both backends must agree.
    """
    _local, pg_keyword, _pg_vector = parity_backends
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    records = [
        Record(
            workspace_id="workspace-a",
            source_kind="note",
            source_id="ranked-higher-by-relevance",
            title="Deployment notes",
            body="PROJ-1234 " * 20,
            created_at=timestamp,
            updated_at=timestamp,
            embedding=[1.0, 0.0],
        ),
        Record(
            workspace_id="workspace-a",
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

    scorer = _StubIdentifierScorer()
    local = LocalRecordBackend(keyword_artifact_scorer=scorer)
    pg_scored = PGKeywordStore(pg_keyword.conn_pool, artifact_scorer=scorer)

    local.index(records)
    PGVectorStore(pg_keyword.conn_pool).upsert(records, "parity-artifact-scorer", 2)
    pg_scored.index(records)

    local_hits = local.search_keyword("PROJ-1234", 1)
    pg_hits = pg_scored.search("PROJ-1234", 1)

    assert len(local_hits) == 1
    assert len(pg_hits) == 1
    assert local_hits[0].source_id == "tagged-with-identifier"
    assert pg_hits[0].source_id == local_hits[0].source_id
