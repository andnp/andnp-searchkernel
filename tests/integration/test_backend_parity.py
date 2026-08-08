"""Local/Postgres parity checks for canonical record retrieval."""

import os
from datetime import UTC, datetime

import pytest

from searchkernel.adapters.stores.pgvector import (
    PGKeywordStore,
    PGVectorStore,
    PostgresConnection,
    _create_schema,
)
from searchkernel.domain import Record, RecordIdentity, RecordStatus
from searchkernel.indices import LocalRecordBackend
from tests.integration.conftest import pg_dsn_for_schema, pg_worker_schema


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
    _create_schema(pg_pool)
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

    filters = {"workspace_id": "workspace-a"}
    local_hits = local.search_keyword("deployment incident", 10, filters)
    pg_hits = pg_keyword.search("deployment incident", 10, filters)
    expected = {
        parity_records[0].storage_key,
        parity_records[1].storage_key,
    }

    assert _keys(local_hits) == expected
    assert _keys(pg_hits) == expected
    assert _keys(local.search_keyword("deployment incident", 10)) == (
        expected | {parity_records[2].storage_key}
    )


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
