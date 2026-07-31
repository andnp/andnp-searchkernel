"""Postgres + pgvector store adapter implementing all four store ports.

Provides VectorStore (with HNSW ANN), KeywordStore (full-text search),
GraphStore (edge relationships), and CacheStore (epoch-based invalidation).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import UTC, datetime
from typing import Any

import psycopg2
import psycopg2.extras
import psycopg2.pool
from psycopg2 import sql

from searchkernel.domain import (
    GraphEdge,
    GraphNeighbor,
    Record,
    RecordHit,
    RecordIdentity,
    Vector,
    canonical_storage_key,
)

logger = logging.getLogger(__name__)

_IDENT_RE = re.compile(r"[^a-z0-9_]+")

# Default HNSW query-time recall knob. Higher = better recall, more latency.
DEFAULT_HNSW_EF_SEARCH = 100
_SCHEMA_ADVISORY_LOCK_KEY = 907341005


def _sanitize_model_name(model_name: str) -> str:
    """Turn an arbitrary model name into a safe SQL-identifier fragment.

    Appends a short content hash so distinct names that sanitize to the
    same fragment (e.g. differing only in punctuation) never collide.
    """
    lowered = model_name.lower()
    sanitized = _IDENT_RE.sub("_", lowered).strip("_") or "model"
    digest = hashlib.sha256(model_name.encode("utf-8")).hexdigest()[:8]
    return f"{sanitized}_{digest}"


def _vector_table_name(model_name: str, dim: int) -> str:
    """Deterministic per-(model_name, dim) table name.

    Each embedding model/dimension pair gets its own typed `vector(dim)`
    column and its own HNSW index, so ANN search stays index-compatible
    even as multiple models coexist (e.g. during a model migration).
    """
    return f"vectors__{_sanitize_model_name(model_name)}__{dim}"


def _vector_literal(vec: Vector) -> str:
    """Serialize a Python vector to pgvector's `[v1,v2,...]` text format."""
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"


def _utc_timestamp(value: datetime) -> datetime:
    """Normalize timestamps before passing them to PostgreSQL."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _canonical_ids(values: list[str | RecordIdentity]) -> list[str]:
    result: list[str] = []
    for value in values:
        if isinstance(value, RecordIdentity):
            result.append(value.storage_key)
        elif value.startswith("record:"):
            result.append(value)
        else:
            raise ValueError(
                "record deletion requires a canonical storage key or RecordIdentity"
            )
    return result


def _migrate_records_schema(cursor) -> None:
    """Upgrade legacy records in the same transaction as schema creation."""
    cursor.execute(
        "ALTER TABLE records ADD COLUMN IF NOT EXISTS workspace_id TEXT;"
    )
    cursor.execute(
        "SELECT record_id, workspace_id, source_kind, source_id FROM records;"
    )
    rows = cursor.fetchall()
    cursor.execute("SELECT table_name FROM vector_tables;")
    vector_tables = [row[0] for row in cursor.fetchall()]

    for old_key, workspace_id, source_kind, source_id in rows:
        new_key = canonical_storage_key(workspace_id, source_kind, source_id)
        if old_key == new_key:
            continue
        cursor.execute(
            "SELECT 1 FROM records WHERE record_id = %s;",
            (new_key,),
        )
        if cursor.fetchone() is not None:
            raise ValueError(
                f"cannot migrate duplicate record identity {new_key!r}"
            )
        for table_name in vector_tables:
            cursor.execute(
                sql.SQL("UPDATE {table} SET record_id = %s WHERE record_id = %s;").format(
                    table=sql.Identifier(table_name)
                ),
                (new_key, old_key),
            )
        cursor.execute(
            "UPDATE records SET record_id = %s WHERE record_id = %s;",
            (new_key, old_key),
        )

    cursor.execute(
        """
        SELECT 1
        FROM pg_constraint c
        JOIN pg_attribute a
          ON a.attrelid = c.conrelid
         AND a.attnum = ANY(c.conkey)
        WHERE c.conrelid = 'records'::regclass
          AND c.contype = 'u'
        GROUP BY c.oid
        HAVING array_agg(
            a.attname::text ORDER BY array_position(c.conkey, a.attnum)
        )
            = ARRAY['workspace_id', 'source_kind', 'source_id'];
        """
    )
    if cursor.fetchone() is None:
        cursor.execute(
            "ALTER TABLE records ADD CONSTRAINT records_identity_unique "
            "UNIQUE (workspace_id, source_kind, source_id);"
        )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_records_workspace "
        "ON records (workspace_id);"
    )


def _migrate_graph_schema(cursor) -> None:
    """Upgrade legacy graph IDs to endpoint composite identities."""
    for column in (
        "source_workspace_id",
        "source_kind",
        "target_workspace_id",
        "target_kind",
    ):
        cursor.execute(
            sql.SQL(
                "ALTER TABLE graph_edges ADD COLUMN IF NOT EXISTS {} TEXT "
                "DEFAULT '';"
            ).format(
                sql.Identifier(column)
            )
        )
    cursor.execute(
        """
        UPDATE graph_edges
        SET source_kind = COALESCE(NULLIF(source_kind, ''), 'legacy'),
            target_kind = COALESCE(NULLIF(target_kind, ''), 'legacy')
        WHERE source_kind IS NULL OR source_kind = ''
           OR target_kind IS NULL OR target_kind = '';
        """
    )
    cursor.execute(
        """
        UPDATE graph_edges
        SET source_workspace_id = COALESCE(source_workspace_id, ''),
            target_workspace_id = COALESCE(target_workspace_id, '');
        """
    )
    cursor.execute(
        "ALTER TABLE graph_edges ALTER COLUMN source_workspace_id SET NOT NULL;"
    )
    cursor.execute(
        "ALTER TABLE graph_edges ALTER COLUMN target_workspace_id SET NOT NULL;"
    )
    cursor.execute(
        """
        SELECT conname
        FROM pg_constraint
        WHERE conrelid = 'graph_edges'::regclass
          AND contype = 'p';
        """
    )
    primary = cursor.fetchone()
    if primary is not None and primary[0] != "graph_edges_identity_pkey":
        cursor.execute(
            sql.SQL("ALTER TABLE graph_edges DROP CONSTRAINT {};").format(
                sql.Identifier(primary[0])
            )
        )
    cursor.execute(
        """
        SELECT 1
        FROM pg_constraint
        WHERE conrelid = 'graph_edges'::regclass
          AND conname = 'graph_edges_identity_unique';
        """
    )
    if cursor.fetchone() is None:
        cursor.execute(
            """
            ALTER TABLE graph_edges
            ADD CONSTRAINT graph_edges_identity_unique UNIQUE (
                source_workspace_id, source_kind, source_id,
                target_workspace_id, target_kind, target_id, edge_type
            );
            """
        )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_graph_edges_source_identity "
        "ON graph_edges (source_workspace_id, source_kind, source_id);"
    )


class PostgresConnection:
    """Thread-safe Postgres connection pool."""

    def __init__(self, dsn: str, min_connections: int = 2, max_connections: int = 10):
        """Initialize connection pool.

        Args:
            dsn: PostgreSQL connection string
            min_connections: Minimum idle connections in pool
            max_connections: Maximum connections in pool
        """
        self.dsn = dsn
        self.pool = psycopg2.pool.SimpleConnectionPool(
            min_connections, max_connections, dsn
        )

    def get_connection(self):
        """Get a connection from the pool."""
        return self.pool.getconn()

    def put_connection(self, conn):
        """Return a connection to the pool."""
        try:
            conn.rollback()
        except psycopg2.Error:
            pass
        self.pool.putconn(conn)

    def execute(self, sql: str, params: tuple = ()) -> Any:
        """Execute a query and return results."""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            result = cursor.fetchall()
            cursor.close()
            conn.commit()
            return result
        finally:
            self.put_connection(conn)

    def execute_one(self, sql: str, params: tuple = ()) -> Any | None:
        """Execute a query and return a single result."""
        result = self.execute(sql, params)
        return result[0] if result else None

    def close(self):
        """Close all connections in the pool."""
        self.pool.closeall()


def _create_schema(conn_pool: PostgresConnection) -> None:
    """Create idempotent schema for vector, keyword, graph, and cache stores."""
    conn = conn_pool.get_connection()
    cursor = None
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT pg_advisory_xact_lock(%s);",
            (_SCHEMA_ADVISORY_LOCK_KEY,),
        )
        # Create pgvector extension
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;")

        # Registry of per-(model_name, dim) vector tables. Each embedding
        # model/dimension pair gets its own typed `vector(dim)` table with
        # a dedicated HNSW index (see PGVectorStore._ensure_vector_table),
        # so ANN search is always index-compatible even with multiple
        # models coexisting (e.g. during a model migration).
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS vector_tables (
                model_name TEXT NOT NULL,
                dim INT NOT NULL,
                table_name TEXT NOT NULL,
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (model_name, dim)
            );
        """)

        # Records table (denormalized metadata for full-text search)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS records (
                record_id TEXT PRIMARY KEY,
                source_kind TEXT NOT NULL,
                source_id TEXT NOT NULL,
                title TEXT NOT NULL,
                body TEXT NOT NULL,
                tsvector_body tsvector,
                workspace_id TEXT,
                created_at TIMESTAMPTZ,
                updated_at TIMESTAMPTZ,
                metadata JSONB DEFAULT '{}',
                uri TEXT,
                status TEXT DEFAULT 'active',
                CONSTRAINT records_identity_unique
                    UNIQUE (workspace_id, source_kind, source_id)
            );
        """)
        _migrate_records_schema(cursor)

        # Full-text search index
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_records_tsvector
            ON records USING gin (tsvector_body);
        """)

        # Graph edges table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS graph_edges (
                source_workspace_id TEXT NOT NULL DEFAULT '',
                source_kind TEXT NOT NULL DEFAULT 'legacy',
                source_id TEXT NOT NULL,
                target_workspace_id TEXT NOT NULL DEFAULT '',
                target_kind TEXT NOT NULL DEFAULT 'legacy',
                target_id TEXT NOT NULL,
                edge_type TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                CONSTRAINT graph_edges_identity_unique UNIQUE (
                    source_workspace_id, source_kind, source_id,
                    target_workspace_id, target_kind, target_id, edge_type
                )
            );
        """)
        _migrate_graph_schema(cursor)

        # Graph edges index
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_graph_edges_source
            ON graph_edges (source_id);
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_graph_edges_target
            ON graph_edges (target_id);
        """)

        # Cache store table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cache_store (
                key TEXT PRIMARY KEY,
                value JSONB NOT NULL,
                epoch INT NOT NULL,
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Cache epoch index
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_cache_epoch
            ON cache_store (epoch);
        """)

        # Index epoch tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS index_epoch (
                id INT PRIMARY KEY DEFAULT 1,
                epoch INT DEFAULT 0,
                CONSTRAINT only_one_row CHECK (id = 1)
            );
        """)

        # Initialize epoch if not present
        cursor.execute("SELECT COUNT(*) FROM index_epoch;")
        row = cursor.fetchone()
        if row is not None and row[0] == 0:
            cursor.execute("INSERT INTO index_epoch (epoch) VALUES (0);")

        conn.commit()
        logger.info("pgvector schema initialized successfully")
    finally:
        if cursor is not None:
            cursor.close()
        conn_pool.put_connection(conn)


class PGVectorStore:
    """Postgres + pgvector implementation of VectorStore port.

    Each (model_name, dim) pair gets its own table with a typed
    `vector(dim)` column and a dedicated HNSW index, so ANN search is
    always index-compatible -- pgvector's HNSW requires a fixed
    dimension per indexed column, which an untyped `vector` column
    cannot provide. Multiple models can coexist (e.g. during a model
    migration); each lives in its own table.

    `search()` takes `model_name`/`dim` explicitly (rather than relying on
    instance "active model" state) so concurrent callers can query
    different models safely.
    """

    def __init__(
        self,
        conn_pool: PostgresConnection,
        hnsw_ef_search: int = DEFAULT_HNSW_EF_SEARCH,
    ):
        """Initialize vector store.

        Args:
            conn_pool: PostgresConnection pool
            hnsw_ef_search: hnsw.ef_search GUC applied per query (recall/latency knob)
        """
        self.conn_pool = conn_pool
        self.hnsw_ef_search = hnsw_ef_search

    def _ensure_vector_table(self, cursor, model_name: str, dim: int) -> str:
        """Return the table name for (model_name, dim), creating it (+ HNSW index) if needed.

        Raises:
            ValueError: If model_name is already registered under a different dim.
        """
        cursor.execute(
            "SELECT dim, table_name FROM vector_tables WHERE model_name = %s;",
            (model_name,),
        )
        rows = cursor.fetchall()
        for existing_dim, existing_table in rows:
            if existing_dim != dim:
                raise ValueError(
                    f"Dimension mismatch for model {model_name}: "
                    f"expected {existing_dim}, got {dim}"
                )
            return existing_table

        table_name = _vector_table_name(model_name, dim)

        cursor.execute(
            sql.SQL(
                "CREATE TABLE IF NOT EXISTS {table} ("
                "record_id TEXT PRIMARY KEY, "
                "embedding vector({dim}) NOT NULL, "
                "created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP, "
                "updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP"
                ");"
            ).format(table=sql.Identifier(table_name), dim=sql.SQL(str(int(dim))))
        )

        cursor.execute(
            sql.SQL(
                "CREATE INDEX IF NOT EXISTS {index_name} ON {table} "
                "USING hnsw (embedding vector_cosine_ops);"
            ).format(
                index_name=sql.Identifier(f"idx_{table_name}_hnsw"),
                table=sql.Identifier(table_name),
            )
        )

        cursor.execute(
            "INSERT INTO vector_tables (model_name, dim, table_name) "
            "VALUES (%s, %s, %s) ON CONFLICT (model_name, dim) DO NOTHING;",
            (model_name, dim, table_name),
        )
        return table_name

    def upsert(self, records: list[Record], model_name: str, dim: int) -> None:
        """Upsert records with embeddings.

        Args:
            records: Records with embedding set
            model_name: Embedding model name
            dim: Vector dimensionality

        Raises:
            ValueError: If embedding dimension doesn't match stored dimension
        """
        if not records:
            return

        if dim < 1:
            raise ValueError("dim must be positive")
        for record in records:
            if record.embedding is not None and len(record.embedding) != dim:
                raise ValueError(
                    f"Embedding dimension mismatch for record {record.storage_key}: "
                    f"expected {dim}, got {len(record.embedding)}"
                )

        conn = self.conn_pool.get_connection()
        cursor = None
        try:
            cursor = conn.cursor()

            table_name = self._ensure_vector_table(cursor, model_name, dim)

            # Upsert records table
            for record in records:
                metadata_json = json.dumps(record.metadata)
                tsvector_text = f"{record.title} {record.body}"
                record_key = record.storage_key

                cursor.execute(
                    """
                    INSERT INTO records
                    (record_id, workspace_id, source_kind, source_id, title, body,
                     tsvector_body, created_at, updated_at, metadata, uri, status)
                    VALUES (%s, %s, %s, %s, %s, %s,
                            to_tsvector('english', %s), %s, %s, %s, %s, %s)
                    ON CONFLICT (record_id) DO UPDATE SET
                        workspace_id = EXCLUDED.workspace_id,
                        source_kind = EXCLUDED.source_kind,
                        source_id = EXCLUDED.source_id,
                        title = EXCLUDED.title,
                        body = EXCLUDED.body,
                        tsvector_body = to_tsvector('english', EXCLUDED.title || ' ' || EXCLUDED.body),
                        created_at = EXCLUDED.created_at,
                        updated_at = EXCLUDED.updated_at,
                        metadata = EXCLUDED.metadata,
                        uri = EXCLUDED.uri,
                        status = EXCLUDED.status;
                    """,
                    (
                        record_key,
                        record.workspace_id,
                        record.source_kind,
                        record.source_id,
                        record.title,
                        record.body,
                        tsvector_text,
                        _utc_timestamp(record.created_at),
                        _utc_timestamp(record.updated_at),
                        metadata_json,
                        record.uri,
                        record.status.value,
                    ),
                )

            # Upsert vectors into the per-model typed table
            upsert_vec_sql = sql.SQL(
                "INSERT INTO {table} (record_id, embedding) "
                "VALUES (%s, %s::vector) "
                "ON CONFLICT (record_id) DO UPDATE SET "
                "embedding = EXCLUDED.embedding, updated_at = CURRENT_TIMESTAMP;"
            ).format(table=sql.Identifier(table_name))

            vector_rows = [
                (record.storage_key, _vector_literal(record.embedding))
                for record in records
                if record.embedding is not None
            ]
            if vector_rows:
                psycopg2.extras.execute_values(
                    cursor,
                    upsert_vec_sql.as_string(cursor).replace(
                        "VALUES (%s, %s::vector)", "VALUES %s"
                    ),
                    vector_rows,
                )

            # Increment epoch
            cursor.execute("UPDATE index_epoch SET epoch = epoch + 1;")

            conn.commit()
            logger.debug(f"Upserted {len(records)} records for model {model_name}")
        finally:
            if cursor is not None:
                cursor.close()
            self.conn_pool.put_connection(conn)

    def search(
        self,
        query_vector: Vector,
        k: int,
        *,
        model_name: str,
        dim: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RecordHit | tuple[str, float]]:
        """Search for nearest neighbors using cosine similarity (ANN via HNSW).

        Args:
            query_vector: Query embedding vector
            k: Number of results to return
            model_name: Embedding model query_vector was produced with;
                selects which per-model table to search.
            dim: Dimensionality of query_vector.
            filters: Optional filters (source-kind filtering, etc.)

        Returns:
            List of (record_id, similarity_score) tuples, sorted descending
        """
        if k < 1:
            return []
        if len(query_vector) != dim:
            raise ValueError(
                f"Query vector dimension mismatch: expected {dim}, got {len(query_vector)}"
            )
        conn = self.conn_pool.get_connection()
        cursor = None
        try:
            cursor = conn.cursor()

            cursor.execute(
                "SELECT 1 FROM vector_tables WHERE model_name = %s AND dim = %s;",
                (model_name, dim),
            )
            if cursor.fetchone() is None:
                return []

            table_name = _vector_table_name(model_name, dim)

            # hnsw.ef_search cannot be a bind parameter (SET does not accept
            # protocol parameters); the value is an internally-controlled int.
            cursor.execute(f"SET LOCAL hnsw.ef_search = {int(self.hnsw_ef_search)};")

            where_parts: list[str] = []
            filter_params: list[Any] = []
            status_values = ["active"]
            if filters and "source_kinds" in filters:
                source_kinds = filters["source_kinds"]
                if not source_kinds:
                    return []
                placeholders = ",".join(["%s"] * len(source_kinds))
                where_parts.append(f"r.source_kind IN ({placeholders})")
                filter_params.extend(source_kinds)
            if filters and "statuses" in filters:
                status_values = list(filters["statuses"])
            if filters and "include_inactive" in filters and filters["include_inactive"]:
                status_values = ["active", "stale", "archived"]
            where_parts.append("r.status = ANY(%s)")
            filter_params.append(status_values)
            workspace_id = filters.get("workspace_id") if filters else None
            if workspace_id is not None:
                where_parts.append("r.workspace_id = %s")
                filter_params.append(workspace_id)
            where_clause = "AND " + " AND ".join(where_parts)

            vec_literal = _vector_literal(query_vector)

            # Order by the raw distance operator (not a wrapped/aliased
            # expression) so the planner can use the HNSW index for ANN.
            query_sql = sql.SQL(
                "SELECT r.workspace_id, r.source_kind, r.source_id, "
                "v.embedding <=> %s::vector AS distance "
                "FROM {table} v "
                "JOIN records r ON v.record_id = r.record_id "
                "WHERE 1 = 1 " + where_clause + " "
                "ORDER BY v.embedding <=> %s::vector ASC "
                "LIMIT %s;"
            ).format(table=sql.Identifier(table_name))

            params = [vec_literal, *filter_params, vec_literal, k]

            cursor.execute(query_sql, params)
            results = cursor.fetchall()
            conn.commit()
            return [
                (
                    RecordHit(
                        RecordIdentity(row[0], row[1], row[2]),
                        1.0 - float(row[3]),
                    )
                )
                for row in results
            ]
        finally:
            if cursor is not None:
                cursor.close()
            self.conn_pool.put_connection(conn)

    def delete(self, record_ids: list[str | RecordIdentity]) -> None:
        """Delete records by ID.

        Args:
            record_ids: IDs to delete
        """
        if not record_ids:
            return

        storage_ids = _canonical_ids(record_ids)
        conn = self.conn_pool.get_connection()
        cursor = None
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT table_name FROM vector_tables;")
            table_names = [row[0] for row in cursor.fetchall()]

            for table_name in table_names:
                cursor.execute(
                    sql.SQL("DELETE FROM {table} WHERE record_id = ANY(%s);").format(
                        table=sql.Identifier(table_name)
                    ),
                    (storage_ids,),
                )

            cursor.execute(
                "DELETE FROM records WHERE record_id = ANY(%s);", (storage_ids,)
            )

            # Increment epoch
            cursor.execute("UPDATE index_epoch SET epoch = epoch + 1;")

            conn.commit()
            logger.debug(f"Deleted {len(record_ids)} records")
        finally:
            if cursor is not None:
                cursor.close()
            self.conn_pool.put_connection(conn)

    def delete_for_model(
        self,
        record_ids: list[str | RecordIdentity],
        model_name: str,
        dim: int,
    ) -> None:
        """Delete records from one model while preserving other model vectors."""
        if not record_ids:
            return

        storage_ids = _canonical_ids(record_ids)
        conn = self.conn_pool.get_connection()
        cursor = None
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT table_name FROM vector_tables WHERE model_name = %s AND dim = %s;",
                (model_name, dim),
            )
            row = cursor.fetchone()
            if row is None:
                return
            table_name = row[0]

            cursor.execute(
                sql.SQL("DELETE FROM {table} WHERE record_id = ANY(%s);").format(
                    table=sql.Identifier(table_name)
                ),
                (storage_ids,),
            )

            cursor.execute("SELECT table_name FROM vector_tables;")
            table_names = [table_row[0] for table_row in cursor.fetchall()]
            remaining_checks = [
                sql.SQL(
                    "NOT EXISTS (SELECT 1 FROM {table} v "
                    "WHERE v.record_id = r.record_id)"
                ).format(table=sql.Identifier(other_table))
                for other_table in table_names
            ]
            if remaining_checks:
                cursor.execute(
                    sql.SQL("DELETE FROM records r WHERE r.record_id = ANY(%s) AND {}").format(
                        sql.SQL(" AND ").join(remaining_checks)
                    ),
                    (storage_ids,),
                )
            else:
                cursor.execute(
                    "DELETE FROM records WHERE record_id = ANY(%s);",
                    (storage_ids,),
                )

            cursor.execute("UPDATE index_epoch SET epoch = epoch + 1;")
            conn.commit()
        finally:
            if cursor is not None:
                cursor.close()
            self.conn_pool.put_connection(conn)

    def epoch(self) -> int:
        """Get current index epoch."""
        conn = self.conn_pool.get_connection()
        cursor = None
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT epoch FROM index_epoch LIMIT 1;")
            result = cursor.fetchone()
            return result[0] if result else 0
        finally:
            if cursor is not None:
                cursor.close()
            self.conn_pool.put_connection(conn)


class PGKeywordStore:
    """Postgres full-text search implementation of KeywordStore port."""

    def __init__(self, conn_pool: PostgresConnection):
        """Initialize keyword store.

        Args:
            conn_pool: PostgresConnection pool
        """
        self.conn_pool = conn_pool

    def index(self, records: list[Record]) -> None:
        """Index records for full-text search.

        Args:
            records: Records to index
        """
        if not records:
            return

        conn = self.conn_pool.get_connection()
        cursor = None
        try:
            cursor = conn.cursor()

            psycopg2.extras.execute_values(
                cursor,
                """
                UPDATE records AS r
                SET tsvector_body = to_tsvector('english', data.search_text)
                FROM (VALUES %s) AS data(record_id, search_text)
                WHERE r.record_id = data.record_id;
                """,
                [
                    (record.storage_key, f"{record.title} {record.body}")
                    for record in records
                ],
            )

            conn.commit()
            logger.debug(f"Indexed {len(records)} records for keyword search")
        finally:
            if cursor is not None:
                cursor.close()
            self.conn_pool.put_connection(conn)

    def search(
        self, query: str, k: int, filters: dict[str, Any] | None = None
    ) -> list[RecordHit | tuple[str, float]]:
        """Search for records matching the query.

        Args:
            query: Full-text search query
            k: Number of results to return
            filters: Optional filters

        Returns:
            List of (record_id, relevance_score) tuples, sorted descending
        """
        conn = self.conn_pool.get_connection()
        cursor = None
        try:
            cursor = conn.cursor()

            # Active records are the default search surface.
            where_parts: list[str] = []
            filter_params: list[Any] = []
            status_values = ["active"]
            if filters and "source_kinds" in filters:
                source_kinds = filters["source_kinds"]
                if not source_kinds:
                    return []
                placeholders = ",".join(["%s"] * len(source_kinds))
                where_parts.append(f"source_kind IN ({placeholders})")
                filter_params.extend(source_kinds)
            if filters and "statuses" in filters:
                status_values = list(filters["statuses"])
            if filters and filters.get("include_inactive"):
                status_values = ["active", "stale", "archived"]
            where_parts.append("status = ANY(%s)")
            filter_params.append(status_values)
            if filters and filters.get("workspace_id") is not None:
                where_parts.append("workspace_id = %s")
                filter_params.append(filters["workspace_id"])
            where_clause = "AND " + " AND ".join(where_parts)

            sql = f"""
                SELECT workspace_id, source_kind, source_id,
                       ts_rank(tsvector_body, plainto_tsquery('english', %s)) as relevance
                FROM records
                WHERE tsvector_body @@ plainto_tsquery('english', %s)
                {where_clause}
                ORDER BY relevance DESC
                LIMIT %s;
            """
            params = [query, query, *filter_params, k]

            cursor.execute(sql, params)
            results = cursor.fetchall()
            return [
                RecordHit(
                    RecordIdentity(row[0], row[1], row[2]),
                    float(row[3]),
                )
                for row in results
            ]
        finally:
            if cursor is not None:
                cursor.close()
            self.conn_pool.put_connection(conn)


class PGGraphStore:
    """Postgres implementation of GraphStore port."""

    def __init__(self, conn_pool: PostgresConnection):
        """Initialize graph store.

        Args:
            conn_pool: PostgresConnection pool
        """
        self.conn_pool = conn_pool

    def upsert_edges(
        self,
        edges: list[GraphEdge | tuple[str, str, str, float]],
    ) -> None:
        """Upsert edges in the graph.

        Args:
            edges: List of (source_id, target_id, edge_type, weight) tuples
        """
        if not edges:
            return

        conn = self.conn_pool.get_connection()
        cursor = None
        try:
            cursor = conn.cursor()

            rows = [
                (
                    edge.source.workspace_id or "",
                    edge.source.source_kind,
                    edge.source.source_id,
                    edge.target.workspace_id or "",
                    edge.target.source_kind,
                    edge.target.source_id,
                    edge.edge_type,
                    edge.weight,
                )
                if isinstance(edge, GraphEdge)
                else (
                    "",
                    "legacy",
                    edge[0],
                    "",
                    "legacy",
                    edge[1],
                    edge[2],
                    edge[3],
                )
                for edge in edges
            ]
            psycopg2.extras.execute_values(
                cursor,
                """
                INSERT INTO graph_edges (
                    source_workspace_id, source_kind, source_id,
                    target_workspace_id, target_kind, target_id,
                    edge_type, weight
                )
                VALUES %s
                ON CONFLICT ON CONSTRAINT graph_edges_identity_unique DO UPDATE SET
                    weight = EXCLUDED.weight,
                    updated_at = CURRENT_TIMESTAMP;
                """,
                rows,
            )

            conn.commit()
            logger.debug(f"Upserted {len(edges)} edges")
        finally:
            if cursor is not None:
                cursor.close()
            self.conn_pool.put_connection(conn)

    def neighbors(
        self,
        record_id: str | RecordIdentity,
        edge_types: list[str] | None = None,
        depth: int = 1,
    ) -> list[GraphNeighbor | tuple[str, str, float]]:
        """Retrieve neighbors of a record.

        Args:
            record_id: Starting record ID
            edge_types: Optional filter by edge type
            depth: Traversal depth (default 1 for one-hop)

        Returns:
            List of (neighbor_id, edge_type, cumulative_weight) tuples
        """
        if depth < 1:
            raise ValueError("depth must be positive")
        conn = self.conn_pool.get_connection()
        cursor = None
        try:
            cursor = conn.cursor()

            identity = (
                record_id
                if isinstance(record_id, RecordIdentity)
                else RecordIdentity(None, "legacy", record_id)
            )
            edge_filter = ""
            params: list[Any] = [
                identity.workspace_id or "",
                identity.source_kind,
                identity.source_id,
            ]
            if edge_types:
                placeholders = ",".join(["%s"] * len(edge_types))
                edge_filter = f"AND edge_type IN ({placeholders})"
                params.extend(edge_types)
            params.append(depth)
            if edge_types:
                params.extend(edge_types)

            sql = f"""
                WITH RECURSIVE walk AS (
                    SELECT source_workspace_id, source_kind, source_id,
                           target_workspace_id, target_kind, target_id,
                           edge_type, weight, 1 AS hop,
                           ARRAY[
                               source_kind || chr(31) || source_id,
                               target_kind || chr(31) || target_id
                           ] AS path
                    FROM graph_edges
                    WHERE source_workspace_id IS NOT DISTINCT FROM %s
                      AND source_kind = %s
                      AND source_id = %s
                      {edge_filter}
                    UNION ALL
                    SELECT walk.source_workspace_id, walk.source_kind,
                           walk.source_id, edge.target_workspace_id,
                           edge.target_kind, edge.target_id, edge.edge_type,
                           walk.weight * edge.weight, walk.hop + 1,
                           walk.path || (
                               edge.target_kind || chr(31) || edge.target_id
                           )
                    FROM walk
                    JOIN graph_edges edge
                      ON edge.source_workspace_id IS NOT DISTINCT FROM
                         walk.target_workspace_id
                     AND edge.source_kind = walk.target_kind
                     AND edge.source_id = walk.target_id
                    WHERE walk.hop < %s
                      AND NOT (
                          edge.target_kind || chr(31) || edge.target_id
                      ) = ANY(walk.path)
                      AND TRUE {edge_filter.replace("edge_type", "edge.edge_type")}
                ),
                ranked AS (
                    SELECT target_workspace_id, target_kind, target_id,
                           edge_type, weight,
                           ROW_NUMBER() OVER (
                               PARTITION BY target_workspace_id, target_kind,
                                            target_id
                               ORDER BY weight DESC
                           ) AS row_number
                    FROM walk
                )
                SELECT target_workspace_id, target_kind, target_id,
                       edge_type, weight
                FROM ranked
                WHERE row_number = 1
                ORDER BY weight DESC, target_kind, target_id;
            """

            cursor.execute(sql, params)
            results = cursor.fetchall()
            if isinstance(record_id, RecordIdentity):
                return [
                    GraphNeighbor(
                        RecordIdentity(row[0] or None, row[1], row[2]),
                        row[3],
                        float(row[4]),
                    )
                    for row in results
                ]
            return [(row[2], row[3], float(row[4])) for row in results]
        finally:
            if cursor is not None:
                cursor.close()
            self.conn_pool.put_connection(conn)


class PGCacheStore:
    """Postgres implementation of CacheStore port with epoch-based invalidation."""

    def __init__(self, conn_pool: PostgresConnection):
        """Initialize cache store.

        Args:
            conn_pool: PostgresConnection pool
        """
        self.conn_pool = conn_pool

    def get(self, key: str) -> Any | None:
        """Retrieve a cached value.

        Args:
            key: Cache key

        Returns:
            Cached value, or None if not found or stale
        """
        conn = self.conn_pool.get_connection()
        cursor = None
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM cache_store WHERE key = %s;", (key,))
            result = cursor.fetchone()
            if result:
                value = result[0]
                # psycopg2 automatically deserializes JSONB columns to dicts
                if isinstance(value, str):
                    return json.loads(value)
                return value
            return None
        finally:
            if cursor is not None:
                cursor.close()
            self.conn_pool.put_connection(conn)

    def set(self, key: str, value: Any, epoch: int) -> None:
        """Store a value with an associated epoch.

        Args:
            key: Cache key
            value: Value to cache
            epoch: Index epoch at cache time
        """
        conn = self.conn_pool.get_connection()
        cursor = None
        try:
            cursor = conn.cursor()
            value_json = json.dumps(value)
            cursor.execute(
                """
                INSERT INTO cache_store (key, value, epoch)
                VALUES (%s, %s, %s)
                ON CONFLICT (key) DO UPDATE SET
                    value = EXCLUDED.value,
                    epoch = EXCLUDED.epoch;
                """,
                (key, value_json, epoch),
            )
            conn.commit()
        finally:
            if cursor is not None:
                cursor.close()
            self.conn_pool.put_connection(conn)

    def invalidate_epoch(self, epoch: int) -> None:
        """Invalidate all entries from an epoch or earlier.

        Args:
            epoch: Entries with epoch <= this are discarded
        """
        conn = self.conn_pool.get_connection()
        cursor = None
        try:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM cache_store WHERE epoch <= %s;",
                (epoch,),
            )
            conn.commit()
            logger.debug(f"Invalidated cache entries for epochs <= {epoch}")
        finally:
            if cursor is not None:
                cursor.close()
            self.conn_pool.put_connection(conn)
