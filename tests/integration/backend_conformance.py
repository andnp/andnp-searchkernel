"""Shared backend adapters for representative conformance tests."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from searchkernel.adapters.stores.pgvector import (
    PGKeywordStore,
    PGVectorStore,
    PostgresConnection,
    create_schema,
)
from searchkernel.domain import Record, RecordHit, RecordIdentity, SearchFilters, Vector
from searchkernel.indices import FAISSLocalVectorStore, LocalRecordBackend
from tests.integration.conftest import pg_dsn_for_schema, pg_worker_schema

MODEL_NAME = "fixture-v1"
VECTOR_DIMENSION = 4


class BackendConformanceTarget(Protocol):
    """Behavior boundary shared by local and optional backend adapters."""

    name: str

    def keyword(
        self, query: str, k: int, filters: SearchFilters | None = None
    ) -> Sequence[RecordHit]:
        ...

    def vector(
        self,
        query: Vector,
        k: int,
        filters: SearchFilters | None = None,
    ) -> Sequence[RecordHit]:
        ...

    def delete(self, identities: Sequence[RecordIdentity]) -> None:
        ...

    def upsert(self, records: list[Record]) -> None:
        ...

    def close(self) -> None:
        ...


class LocalBackendTarget:
    """Conformance adapter for the local record backend."""

    name = "local"

    def __init__(self, backend: LocalRecordBackend) -> None:
        self._backend = backend

    def keyword(
        self, query: str, k: int, filters: SearchFilters | None = None
    ) -> Sequence[RecordHit]:
        return self._backend.search_keyword(query, k, filters)

    def vector(
        self,
        query: Vector,
        k: int,
        filters: SearchFilters | None = None,
    ) -> Sequence[RecordHit]:
        return self._backend.search_vector(
            query,
            k,
            model_name=MODEL_NAME,
            dim=VECTOR_DIMENSION,
            filters=filters,
        )

    def delete(self, identities: Sequence[RecordIdentity]) -> None:
        self._backend.delete([identity.storage_key for identity in identities])

    def upsert(self, records: list[Record]) -> None:
        self._backend.upsert(records, MODEL_NAME, VECTOR_DIMENSION)

    def close(self) -> None:
        self._backend.db_manager.close()


class FAISSBackendTarget(LocalBackendTarget):
    """Conformance adapter using FAISS for vector retrieval."""

    name = "faiss"

    def __init__(self, backend: LocalRecordBackend, index_path: Path) -> None:
        super().__init__(backend)
        self._backend = backend
        self._vectors = FAISSLocalVectorStore(backend, index_path=index_path)

    def vector(
        self,
        query: Vector,
        k: int,
        filters: SearchFilters | None = None,
    ) -> Sequence[RecordHit]:
        return self._vectors.search(
            query,
            k,
            model_name=MODEL_NAME,
            dim=VECTOR_DIMENSION,
            filters=filters,
        )


class PostgresBackendTarget:
    """Conformance adapter for PostgreSQL keyword and vector stores."""

    name = "postgres"

    def __init__(self, pool: PostgresConnection) -> None:
        self._pool = pool
        self._keyword_store = PGKeywordStore(pool)
        self._vector_store = PGVectorStore(pool)

    def keyword(
        self, query: str, k: int, filters: SearchFilters | None = None
    ) -> Sequence[RecordHit]:
        return self._keyword_store.search(
            query,
            k,
            dict(filters) if filters is not None else None,
        )

    def vector(
        self,
        query: Vector,
        k: int,
        filters: SearchFilters | None = None,
    ) -> Sequence[RecordHit]:
        return self._vector_store.search(
            query,
            k,
            model_name=MODEL_NAME,
            dim=VECTOR_DIMENSION,
            filters=dict(filters) if filters is not None else None,
        )

    def delete(self, identities: Sequence[RecordIdentity]) -> None:
        self._vector_store.delete(list(identities))

    def upsert(self, records: list[Record]) -> None:
        self._vector_store.upsert(records, MODEL_NAME, VECTOR_DIMENSION)

    def close(self) -> None:
        self._pool.close()


def seed_local_target(
    records: list[Record], database_path: Path
) -> LocalBackendTarget:
    """Build a local target with both keyword and vector data indexed."""
    backend = LocalRecordBackend(database_path)
    backend.index(records)
    backend.upsert(records, MODEL_NAME, VECTOR_DIMENSION)
    return LocalBackendTarget(backend)


def seed_faiss_target(
    records: list[Record], database_path: Path, index_path: Path
) -> FAISSBackendTarget:
    """Build a FAISS target over the shared local record data."""
    backend = LocalRecordBackend(database_path)
    backend.index(records)
    backend.upsert(records, MODEL_NAME, VECTOR_DIMENSION)
    return FAISSBackendTarget(backend, index_path)


def seed_postgres_target(
    records: list[Record], dsn: str, schema: str
) -> PostgresBackendTarget:
    """Build an isolated PostgreSQL target for optional parity runs."""
    bootstrap_pool = PostgresConnection(dsn, min_connections=1, max_connections=1)
    bootstrap_connection = bootstrap_pool.get_connection()
    bootstrap_cursor = bootstrap_connection.cursor()
    bootstrap_cursor.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}";')
    bootstrap_connection.commit()
    bootstrap_cursor.close()
    bootstrap_pool.put_connection(bootstrap_connection)
    bootstrap_pool.close()

    pool = PostgresConnection(pg_dsn_for_schema(dsn, schema))
    create_schema(pool)
    connection = pool.get_connection()
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
    pool.put_connection(connection)

    target = PostgresBackendTarget(pool)
    target.upsert(records)
    return target


def conformance_schema(config: object) -> str:
    """Return a worker-isolated schema name for the optional PostgreSQL target."""
    return f"{pg_worker_schema(config)}_conformance"


def storage_keys(hits: Sequence[RecordHit]) -> tuple[str, ...]:
    """Normalize backend hits to their canonical ordered storage keys."""
    return tuple(hit.storage_key for hit in hits)
