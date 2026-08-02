"""Postgres + pgvector store adapter implementing all four store ports.

Provides VectorStore (with HNSW ANN), KeywordStore (full-text search),
GraphStore (edge relationships), and CacheStore (epoch-based invalidation).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, ClassVar, Protocol

try:
    import psycopg2  # type: ignore[import-not-found]
    import psycopg2.extras  # type: ignore[import-not-found]
    import psycopg2.pool  # type: ignore[import-not-found]
    from psycopg2 import sql  # type: ignore[import-not-found]
except ImportError:
    psycopg2 = None  # type: ignore[assignment]
    sql = None  # type: ignore[assignment]

try:
    import psycopg  # type: ignore[import-not-found]
    import psycopg_pool  # type: ignore[import-not-found]
except ImportError:
    psycopg = None  # type: ignore[assignment]
    psycopg_pool = None  # type: ignore[assignment]

from searchkernel.domain import (
    GraphEdge,
    GraphNeighbor,
    ModelDimensionMismatchError,
    Record,
    RecordHit,
    RecordIdentity,
    Vector,
)
from searchkernel.domain.vector_filters import (
    candidate_storage_keys,
    filter_values,
    status_values,
)

logger = logging.getLogger(__name__)

_IDENT_RE = re.compile(r"[^a-z0-9_]+")

# Default HNSW query-time recall knob. Higher = better recall, more latency.
DEFAULT_HNSW_EF_SEARCH = 100
DEFAULT_HNSW_ITERATIVE_SCAN = "auto"
DEFAULT_HNSW_MAX_SCAN_TUPLES = 20_000
DEFAULT_HNSW_SCAN_MEM_MULTIPLIER = 1.0
DEFAULT_VECTOR_OVERFETCH_MULTIPLIER = 2.0
DEFAULT_VECTOR_MAX_SCAN_ROUNDS = 4
_SCHEMA_ADVISORY_LOCK_KEY = 907341005
_ITERATIVE_SCAN_MODES = {"auto", "off", "strict_order", "relaxed_order"}


@dataclass(frozen=True, slots=True)
class PGVectorFeatureSupport:
    extension_version: str | None
    iterative_scan: bool

    @classmethod
    def from_extension_version(
        cls,
        extension_version: str | None,
    ) -> PGVectorFeatureSupport:
        if not extension_version:
            return cls(None, False)
        numbers = re.match(r"^(\d+)\.(\d+)", extension_version)
        version = (
            (int(numbers.group(1)), int(numbers.group(2)))
            if numbers is not None
            else (0, 0)
        )
        return cls(extension_version, version >= (0, 8))


def bounded_scan_limits(
    requested_k: int,
    *,
    max_scan_tuples: int,
    max_scan_rounds: int,
    overfetch_multiplier: float,
) -> list[int]:
    """Return bounded ANN limits for adaptive filtered retrieval."""
    if requested_k < 1:
        return []
    limits: list[int] = []
    current = min(requested_k, max_scan_tuples)
    for _ in range(max_scan_rounds):
        limits.append(current)
        if current >= max_scan_tuples:
            break
        current = min(
            max_scan_tuples,
            max(current + 1, math.ceil(current * overfetch_multiplier)),
        )
    return limits


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
    while multiple models coexist.
    """
    return f"vectors__{_sanitize_model_name(model_name)}__{dim}"


def _vector_literal(vec: Vector) -> str:
    """Serialize a Python vector to pgvector's `[v1,v2,...]` text format."""
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"


def _string_filter_values(
    filters: Mapping[str, Any],
    *names: str,
) -> list[str] | None:
    for name in names:
        if name in filters and filters[name] is not None:
            return [
                value.value if hasattr(value, "value") else str(value)
                for value in filter_values(filters[name])
            ]
    return None


def _path_variants(value: Any) -> set[str]:
    normalized = str(value).replace("\\", "/")
    variants = {normalized}
    leaf = normalized.rsplit("/", 1)[-1]
    if "." in leaf:
        stem = leaf.rsplit(".", 1)[0]
        parent = normalized.rsplit("/", 1)[0]
        variants.add(f"{parent}/{stem}" if "/" in normalized else stem)
        variants.add(stem)
    variants.add(leaf)
    return variants


def build_pgvector_filter_sql(
    filters: Mapping[str, Any] | None,
    *,
    record_alias: str = "r",
) -> tuple[list[str], list[Any]]:
    """Build SQL predicates for the canonical vector filter contract.

    Supports generic metadata field filtering via the 'metadata_equals' dict,
    where each key-value pair becomes a metadata->>'key' = value filter.
    """
    filters = filters or {}
    clauses: list[str] = []
    parameters: list[Any] = []
    status_filter = sorted(status_values(filters))
    if not status_filter:
        return ["FALSE"], []
    clauses.append(f"{record_alias}.status = ANY(%s)")
    parameters.append(status_filter)

    workspace_id = filters.get("workspace_id")
    if workspace_id is not None:
        clauses.append(f"{record_alias}.workspace_id = %s")
        parameters.append(workspace_id)

    source_kinds = _string_filter_values(
        filters, "source_kinds", "source_kind", "source_filter"
    )
    if source_kinds is not None:
        if not source_kinds:
            return ["FALSE"], []
        clauses.append(f"{record_alias}.source_kind = ANY(%s)")
        parameters.append(source_kinds)

    candidate_value = filters.get("candidate_ids")
    if candidate_value is None:
        candidate_value = filters.get("candidate_storage_keys")
    if candidate_value is not None:
        candidate_keys = candidate_storage_keys(candidate_value)
        if not candidate_keys:
            return ["FALSE"], []
        clauses.append(f"{record_alias}.record_id = ANY(%s)")
        parameters.append(sorted(candidate_keys))

    project_expr = f"{record_alias}.metadata->>'project_id'"
    project_values = _string_filter_values(
        filters, "project_ids", "project_id", "project_filter"
    )
    if project_values == [] and "project_filter" in filters:
        project_values = None
    if project_values is not None:
        if not project_values:
            return ["FALSE"], []
        clauses.append(f"{project_expr} = ANY(%s)")
        parameters.append(project_values)
    excluded_projects = _string_filter_values(
        filters, "excluded_projects", "excluded_project_ids"
    )
    if excluded_projects:
        clauses.append(f"({project_expr} IS NULL OR {project_expr} <> ALL(%s))")
        parameters.append(excluded_projects)

    metadata_equals = filters.get("metadata_equals")
    if metadata_equals is not None:
        for field, value in metadata_equals.items():
            if value is not None:
                field_expr = f"{record_alias}.metadata->>'%s'" % field
                clauses.append(f"{field_expr} = %s")
                parameters.append(str(value))

    path_expr = (
        f"COALESCE(NULLIF({record_alias}.metadata->>'file_path', ''), "
        f"NULLIF({record_alias}.metadata->>'path', ''), "
        f"NULLIF({record_alias}.metadata->>'source_file', ''), "
        f"NULLIF({record_alias}.uri, ''))"
    )
    document_expr = (
        f"COALESCE(NULLIF({record_alias}.metadata->>'doc_id', ''), "
        f"{record_alias}.source_id)"
    )
    path_values = _string_filter_values(
        filters,
        "paths",
        "file_paths",
        "source_files",
        "path",
        "file_path",
        "source_file",
    )
    if path_values is not None:
        expanded = sorted(
            set().union(*(_path_variants(value) for value in path_values))
        )
        if not expanded:
            return ["FALSE"], []
        clauses.append(
            f"({path_expr} = ANY(%s) OR {document_expr} = ANY(%s))"
        )
        parameters.extend([expanded, expanded])

    excluded_paths = _string_filter_values(
        filters,
        "excluded_files",
        "excluded_paths",
        "excluded_file_paths",
        "excluded_source_files",
    )
    if excluded_paths is not None:
        expanded = sorted(
            set().union(*(_path_variants(value) for value in excluded_paths))
        )
        if expanded:
            clauses.append(
                f"(COALESCE({path_expr}, '') <> ALL(%s) "
                f"AND {document_expr} <> ALL(%s) "
                f"AND regexp_replace(COALESCE({path_expr}, ''), '^.*/', '') "
                f"<> ALL(%s))"
            )
            parameters.extend([expanded, expanded, expanded])

    document_values = _string_filter_values(
        filters, "document_ids", "document_id", "doc_ids", "doc_id"
    )
    if document_values is not None:
        if not document_values:
            return ["FALSE"], []
        clauses.append(f"{document_expr} = ANY(%s)")
        parameters.append(document_values)
    excluded_documents = _string_filter_values(
        filters,
        "excluded_documents",
        "excluded_document_ids",
        "excluded_doc_ids",
    )
    if excluded_documents is not None:
        expanded = sorted(
            set().union(*(_path_variants(value) for value in excluded_documents))
        )
        if expanded:
            clauses.append(
                f"{document_expr} <> ALL(%s) AND {record_alias}.source_id <> ALL(%s)"
            )
            parameters.extend([expanded, expanded])

    return clauses, parameters


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


def _delete_graph_edges_for_identities(
    cursor: Any,
    identities: Sequence[RecordIdentity],
) -> bool:
    """Delete graph edges incident to the supplied record identities."""
    if not identities:
        return False
    rows = [
        (identity.workspace_id or "", identity.source_kind, identity.source_id)
        for identity in identities
    ]
    psycopg2.extras.execute_values(
        cursor,
        """
        DELETE FROM graph_edges AS edge
        USING (VALUES %s) AS deleted(workspace_id, source_kind, source_id)
        WHERE (edge.source_workspace_id, edge.source_kind, edge.source_id) =
              (deleted.workspace_id, deleted.source_kind, deleted.source_id)
           OR (edge.target_workspace_id, edge.target_kind, edge.target_id) =
              (deleted.workspace_id, deleted.source_kind, deleted.source_id);
        """,
        rows,
    )
    return bool(cursor.rowcount)


def _create_record_indexes(cursor) -> None:
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_records_workspace "
        "ON records (workspace_id);"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_records_vector_filters "
        "ON records (workspace_id, source_kind, status, record_id);"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_records_project_filter "
        "ON records ((metadata->>'project_id'));"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_records_document_filter "
        "ON records ((metadata->>'doc_id'));"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_records_path_filter "
        "ON records ((COALESCE("
        "NULLIF(metadata->>'file_path', ''), "
        "NULLIF(metadata->>'path', ''), "
        "NULLIF(metadata->>'source_file', ''), "
        "NULLIF(uri, '')"
        ")));"
    )


def _create_graph_indexes(cursor) -> None:
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_graph_edges_source_identity "
        "ON graph_edges (source_workspace_id, source_kind, source_id);"
    )


class _PostgresSession(Protocol):
    def cursor(self) -> Any: ...

    def commit(self) -> None: ...

    def rollback(self) -> None: ...


class _PostgresConnectionLike(Protocol):
    def get_connection(self) -> _PostgresSession: ...

    def put_connection(self, conn: _PostgresSession) -> None: ...


class _PostgresEpochLane:
    """Own epoch SQL shared by the PostgreSQL storage lanes."""

    _LANE_COLUMNS: ClassVar[dict[str, str]] = {
        "keyword": "keyword_epoch",
        "vector": "vector_epoch",
        "graph": "graph_epoch",
    }

    @staticmethod
    def bump(
        cursor: Any,
        *,
        keyword: bool = False,
        vector: bool = False,
        graph: bool = False,
    ) -> None:
        if not any((keyword, vector, graph)):
            return
        cursor.execute(
            """
            UPDATE index_epoch
            SET epoch = epoch + 1,
                keyword_epoch = keyword_epoch + %s,
                vector_epoch = vector_epoch + %s,
                graph_epoch = graph_epoch + %s;
            """,
            (int(keyword), int(vector), int(graph)),
        )

    def read_all(self, conn_pool: _PostgresConnectionLike) -> dict[str, int]:
        conn = conn_pool.get_connection()
        cursor = None
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT keyword_epoch, vector_epoch, graph_epoch "
                "FROM index_epoch LIMIT 1;"
            )
            row = cursor.fetchone()
            if row is None:
                return {lane: 0 for lane in self._LANE_COLUMNS}
            return {
                lane: int(row[index])
                for index, lane in enumerate(self._LANE_COLUMNS)
            }
        finally:
            if cursor is not None:
                cursor.close()
            conn_pool.put_connection(conn)

    def read(self, conn_pool: _PostgresConnectionLike, lane: str) -> int:
        return self.read_all(conn_pool)[lane]

    @staticmethod
    def read_total(conn_pool: _PostgresConnectionLike) -> int:
        conn = conn_pool.get_connection()
        cursor = None
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT epoch FROM index_epoch LIMIT 1;")
            row = cursor.fetchone()
            return int(row[0]) if row else 0
        finally:
            if cursor is not None:
                cursor.close()
            conn_pool.put_connection(conn)


_POSTGRES_EPOCH_LANE = _PostgresEpochLane()


class PostgresConnection:
    """Thread-safe Postgres connection pool."""

    def __init__(self, dsn: str, min_connections: int = 2, max_connections: int = 10):
        """Initialize connection pool.

        Args:
            dsn: PostgreSQL connection string
            min_connections: Minimum idle connections in pool
            max_connections: Maximum connections in pool

        Raises:
            ImportError: If psycopg2 is not installed
        """
        if psycopg2 is None or psycopg2.pool is None:
            raise ImportError(
                "psycopg2 is required for PostgresConnection. "
                "Install with: pip install 'andnp-searchkernel[pgvector]'"
            )
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


class Psycopg3Connection:
    """Thread-safe Postgres connection pool using psycopg3.

    Provides the same interface as PostgresConnection but uses psycopg3's
    native connection pool (psycopg_pool.ConnectionPool) instead of psycopg2.
    Defers import of psycopg/psycopg_pool until instantiation, so users who
    only use this class don't need psycopg2 installed.
    """

    def __init__(self, dsn: str, min_connections: int = 2, max_connections: int = 10):
        """Initialize connection pool.

        Args:
            dsn: PostgreSQL connection string
            min_connections: Minimum idle connections in pool
            max_connections: Maximum connections in pool

        Raises:
            ImportError: If psycopg or psycopg_pool is not installed
        """
        if psycopg_pool is None:
            raise ImportError(
                "psycopg3 and psycopg_pool are required for Psycopg3Connection. "
                "Install with: pip install 'andnp-searchkernel[pgvector-psycopg3]'"
            )

        self.dsn = dsn
        self.pool = psycopg_pool.ConnectionPool(
            dsn, min_size=min_connections, max_size=max_connections
        )

    def get_connection(self):
        """Get a connection from the pool."""
        return self.pool.getconn()

    def put_connection(self, conn):
        """Return a connection to the pool."""
        try:
            conn.rollback()
        except psycopg.Error as e:  # pyright: ignore[union-attr]
            logger.debug("Connection rollback failed during pool return: %s", e)
        self.pool.putconn(conn)

    def execute(self, sql: str, params: tuple = ()) -> Any:
        """Execute a query and return results."""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            result = cursor.fetchall() if cursor.description is not None else []
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
        self.pool.close()


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
        # models coexisting.
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
                indexed_text TEXT,
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
        _create_record_indexes(cursor)

        # Full-text search index
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_records_tsvector
            ON records USING gin (tsvector_body);
        """)

        # Graph edges table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS graph_edges (
                source_workspace_id TEXT NOT NULL DEFAULT '',
                source_kind TEXT NOT NULL,
                source_id TEXT NOT NULL,
                target_workspace_id TEXT NOT NULL DEFAULT '',
                target_kind TEXT NOT NULL,
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
        _create_graph_indexes(cursor)

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
                keyword_epoch INT DEFAULT 0,
                vector_epoch INT DEFAULT 0,
                graph_epoch INT DEFAULT 0,
                CONSTRAINT only_one_row CHECK (id = 1)
            );
        """)
        cursor.execute(
            "ALTER TABLE index_epoch ADD COLUMN IF NOT EXISTS keyword_epoch INT DEFAULT 0;"
        )
        cursor.execute(
            "ALTER TABLE index_epoch ADD COLUMN IF NOT EXISTS vector_epoch INT DEFAULT 0;"
        )
        cursor.execute(
            "ALTER TABLE index_epoch ADD COLUMN IF NOT EXISTS graph_epoch INT DEFAULT 0;"
        )

        # Initialize epoch if not present
        cursor.execute("SELECT COUNT(*) FROM index_epoch;")
        row = cursor.fetchone()
        if row is not None and row[0] == 0:
            cursor.execute(
                "INSERT INTO index_epoch "
                "(epoch, keyword_epoch, vector_epoch, graph_epoch) "
                "VALUES (0, 0, 0, 0);"
            )

        conn.commit()
        logger.info("pgvector schema initialized successfully")
    finally:
        if cursor is not None:
            cursor.close()
        conn_pool.put_connection(conn)


def _sql_module_for(conn_pool: _PostgresConnectionLike):
    """Return the SQL builder module matching the connection pool's driver.

    For psycopg3 connections, returns psycopg.sql; otherwise returns psycopg2.sql.
    This ensures all Composed objects created by PGVectorStore match the cursor's driver.
    """
    if isinstance(conn_pool, Psycopg3Connection):
        return psycopg.sql
    return sql


class PGVectorStore:
    """Postgres + pgvector implementation of VectorStore port.

    Each (model_name, dim) pair gets its own table with a typed
    `vector(dim)` column and a dedicated HNSW index, so ANN search is
    always index-compatible -- pgvector's HNSW requires a fixed
    dimension per indexed column, which an untyped `vector` column
    cannot provide. Multiple models can coexist; each lives in its own table.

    `search()` takes `model_name`/`dim` explicitly (rather than relying on
    instance "active model" state) so concurrent callers can query
    different models safely.
    """

    def __init__(
        self,
        conn_pool: _PostgresConnectionLike,
        hnsw_ef_search: int = DEFAULT_HNSW_EF_SEARCH,
        *,
        hnsw_iterative_scan: str = DEFAULT_HNSW_ITERATIVE_SCAN,
        hnsw_max_scan_tuples: int = DEFAULT_HNSW_MAX_SCAN_TUPLES,
        hnsw_scan_mem_multiplier: float = DEFAULT_HNSW_SCAN_MEM_MULTIPLIER,
        overfetch_multiplier: float = DEFAULT_VECTOR_OVERFETCH_MULTIPLIER,
        max_scan_rounds: int = DEFAULT_VECTOR_MAX_SCAN_ROUNDS,
    ):
        """Initialize vector store.

        Args:
            conn_pool: PostgresConnection pool
            hnsw_ef_search: hnsw.ef_search GUC applied per query (recall/latency knob)
            hnsw_iterative_scan: auto, off, strict_order, or relaxed_order
            hnsw_max_scan_tuples: maximum tuples visited by iterative scans
            hnsw_scan_mem_multiplier: iterative scan memory multiplier
            overfetch_multiplier: bounded filtered-search expansion factor
            max_scan_rounds: maximum filtered-search expansion rounds
        """
        if not isinstance(hnsw_ef_search, int) or isinstance(hnsw_ef_search, bool):
            raise TypeError("hnsw_ef_search must be an integer")
        if hnsw_ef_search < 1:
            raise ValueError("hnsw_ef_search must be positive")
        if hnsw_iterative_scan not in _ITERATIVE_SCAN_MODES:
            raise ValueError(
                "hnsw_iterative_scan must be auto, off, strict_order, or relaxed_order"
            )
        if (
            not isinstance(hnsw_max_scan_tuples, int)
            or isinstance(hnsw_max_scan_tuples, bool)
            or hnsw_max_scan_tuples < 1
        ):
            raise ValueError("hnsw_max_scan_tuples must be a positive integer")
        if (
            not isinstance(overfetch_multiplier, (int, float))
            or isinstance(overfetch_multiplier, bool)
        ):
            raise TypeError("overfetch_multiplier must be numeric")
        if not math.isfinite(overfetch_multiplier) or overfetch_multiplier < 1.0:
            raise ValueError("overfetch_multiplier must be finite and >= 1")
        if (
            not isinstance(max_scan_rounds, int)
            or isinstance(max_scan_rounds, bool)
            or max_scan_rounds < 1
        ):
            raise ValueError("max_scan_rounds must be a positive integer")
        if (
            not isinstance(hnsw_scan_mem_multiplier, (int, float))
            or isinstance(hnsw_scan_mem_multiplier, bool)
        ):
            raise TypeError("hnsw_scan_mem_multiplier must be numeric")
        if (
            not math.isfinite(hnsw_scan_mem_multiplier)
            or hnsw_scan_mem_multiplier < 1.0
        ):
            raise ValueError("hnsw_scan_mem_multiplier must be finite and >= 1")
        self.conn_pool = conn_pool
        self._sql = _sql_module_for(conn_pool)
        self.hnsw_ef_search = hnsw_ef_search
        self.hnsw_iterative_scan = hnsw_iterative_scan
        self.hnsw_max_scan_tuples = hnsw_max_scan_tuples
        self.hnsw_scan_mem_multiplier = hnsw_scan_mem_multiplier
        self.overfetch_multiplier = float(overfetch_multiplier)
        self.max_scan_rounds = max_scan_rounds
        self._feature_support: PGVectorFeatureSupport | None = None
        self._last_search_diagnostics: dict[str, Any] = {}

    @property
    def feature_support(self) -> PGVectorFeatureSupport | None:
        return self._feature_support

    @property
    def last_search_diagnostics(self) -> dict[str, Any]:
        return dict(self._last_search_diagnostics)

    def detect_feature_support(self) -> PGVectorFeatureSupport:
        """Read feature support from the installed server extension."""
        conn = self.conn_pool.get_connection()
        cursor = None
        try:
            cursor = conn.cursor()
            support = self._detect_feature_support(cursor)
            conn.commit()
            return support
        finally:
            if cursor is not None:
                cursor.close()
            self.conn_pool.put_connection(conn)

    def _detect_feature_support(self, cursor) -> PGVectorFeatureSupport:
        if self._feature_support is not None:
            return self._feature_support
        cursor.execute(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector';"
        )
        row = cursor.fetchone()
        version = str(row[0]) if row and row[0] is not None else None
        self._feature_support = PGVectorFeatureSupport.from_extension_version(version)
        return self._feature_support

    def _configure_hnsw(self, cursor) -> PGVectorFeatureSupport:
        support = self._detect_feature_support(cursor)
        cursor.execute(f"SET LOCAL hnsw.ef_search = {self.hnsw_ef_search};")
        if self.hnsw_iterative_scan != "off" and support.iterative_scan:
            mode = (
                "strict_order"
                if self.hnsw_iterative_scan == "auto"
                else self.hnsw_iterative_scan
            )
            cursor.execute(f"SET LOCAL hnsw.iterative_scan = '{mode}';")
            cursor.execute(
                f"SET LOCAL hnsw.max_scan_tuples = {self.hnsw_max_scan_tuples};"
            )
            cursor.execute(
                "SET LOCAL hnsw.scan_mem_multiplier = "
                f"{self.hnsw_scan_mem_multiplier!r};"
            )
        return support

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
                raise ModelDimensionMismatchError(
                    f"Dimension mismatch for model {model_name}: "
                    f"expected {existing_dim}, got {dim}"
                )
            return existing_table

        table_name = _vector_table_name(model_name, dim)

        cursor.execute(
            self._sql.SQL(
                "CREATE TABLE IF NOT EXISTS {table} ("
                "record_id TEXT PRIMARY KEY, "
                "embedding vector({dim}) NOT NULL, "
                "created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP, "
                "updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP"
                ");"
            ).format(table=self._sql.Identifier(table_name), dim=self._sql.SQL(str(int(dim))))
        )

        cursor.execute(
            self._sql.SQL(
                "CREATE INDEX IF NOT EXISTS {index_name} ON {table} "
                "USING hnsw (embedding vector_cosine_ops);"
            ).format(
                index_name=self._sql.Identifier(f"idx_{table_name}_hnsw"),
                table=self._sql.Identifier(table_name),
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
                indexed_text = record.indexed_text or record.body
                tsvector_text = f"{record.title} {indexed_text}"
                record_key = record.storage_key

                cursor.execute(
                    """
                    INSERT INTO records
                    (record_id, workspace_id, source_kind, source_id, title, body,
                     indexed_text, tsvector_body, created_at, updated_at, metadata, uri, status)
                    VALUES (%s, %s, %s, %s, %s, %s, %s,
                            to_tsvector('english', %s), %s, %s, %s, %s, %s)
                    ON CONFLICT (record_id) DO UPDATE SET
                        workspace_id = EXCLUDED.workspace_id,
                        source_kind = EXCLUDED.source_kind,
                        source_id = EXCLUDED.source_id,
                        title = EXCLUDED.title,
                        body = EXCLUDED.body,
                        indexed_text = EXCLUDED.indexed_text,
                        tsvector_body = to_tsvector(
                            'english',
                            EXCLUDED.title || ' ' ||
                            COALESCE(NULLIF(EXCLUDED.indexed_text, ''), EXCLUDED.body)
                        ),
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
                        record.indexed_text,
                        tsvector_text,
                        _utc_timestamp(record.created_at),
                        _utc_timestamp(record.updated_at),
                        metadata_json,
                        record.uri,
                        record.status.value,
                    ),
                )

            # Upsert vectors into the per-model typed table
            upsert_vec_sql = self._sql.SQL(
                "INSERT INTO {table} (record_id, embedding) "
                "VALUES (%s, %s::vector) "
                "ON CONFLICT (record_id) DO UPDATE SET "
                "embedding = EXCLUDED.embedding, updated_at = CURRENT_TIMESTAMP;"
            ).format(table=self._sql.Identifier(table_name))

            vector_rows = [
                (record.storage_key, _vector_literal(record.embedding))
                for record in records
                if record.embedding is not None
            ]
            if vector_rows:
                if isinstance(self.conn_pool, Psycopg3Connection):
                    insert_sql = (
                        "INSERT INTO " + self._sql.Identifier(table_name).as_string(cursor)
                        + " (record_id, embedding) VALUES (%s, %s::vector) "
                        "ON CONFLICT (record_id) DO UPDATE SET "
                        "embedding = EXCLUDED.embedding, updated_at = CURRENT_TIMESTAMP;"
                    )
                    cursor.executemany(insert_sql, vector_rows)
                else:
                    psycopg2.extras.execute_values(
                        cursor,
                        upsert_vec_sql.as_string(cursor).replace(
                            "VALUES (%s, %s::vector)", "VALUES %s"
                        ),
                        vector_rows,
                    )

            _POSTGRES_EPOCH_LANE.bump(
                cursor, keyword=True, vector=bool(vector_rows)
            )

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
    ) -> list[RecordHit]:
        """Search for nearest neighbors using cosine similarity (ANN via HNSW).

        Args:
            query_vector: Query embedding vector
            k: Number of results to return
            model_name: Embedding model query_vector was produced with;
                selects which per-model table to search.
            dim: Dimensionality of query_vector.
            filters: Optional filters (source-kind filtering, etc.)

        Returns:
            List of RecordHit values, sorted descending
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
                self._last_search_diagnostics = {
                    "requested_k": k,
                    "returned": 0,
                    "scan_rounds": 0,
                    "under_returned": True,
                    "scan_bound_hit": False,
                }
                return []

            table_name = _vector_table_name(model_name, dim)

            feature_support = self._configure_hnsw(cursor)

            where_parts, filter_params = build_pgvector_filter_sql(filters)
            where_clause = "AND " + " AND ".join(where_parts)

            vec_literal = _vector_literal(query_vector)

            # Order by the raw distance operator (not a wrapped/aliased
            # expression) so the planner can use the HNSW index for ANN.
            query_sql = self._sql.SQL(
                "SELECT r.workspace_id, r.source_kind, r.source_id, "
                "v.embedding <=> %s::vector AS distance "
                "FROM {table} v "
                "JOIN records r ON v.record_id = r.record_id "
                "WHERE 1 = 1 " + where_clause + " "
                "ORDER BY v.embedding <=> %s::vector ASC, v.record_id ASC "
                "LIMIT %s;"
            ).format(table=self._sql.Identifier(table_name))

            results: list[Any] = []
            scan_limits = bounded_scan_limits(
                k,
                max_scan_tuples=self.hnsw_max_scan_tuples,
                max_scan_rounds=self.max_scan_rounds,
                overfetch_multiplier=self.overfetch_multiplier,
            )
            scan_rounds = 0
            last_scan_limit = 0
            for scan_limit in scan_limits:
                scan_rounds += 1
                last_scan_limit = scan_limit
                params = [vec_literal, *filter_params, vec_literal, scan_limit]
                cursor.execute(query_sql, params)
                results = cursor.fetchall()
                if len(results) >= k or scan_limit >= self.hnsw_max_scan_tuples:
                    break
            self._last_search_diagnostics = {
                "requested_k": k,
                "returned": min(len(results), k),
                "scan_rounds": scan_rounds,
                "scan_limit": last_scan_limit,
                "scan_bound_hit": last_scan_limit >= self.hnsw_max_scan_tuples,
                "under_returned": len(results) < k,
                "iterative_scan": feature_support.iterative_scan,
                "extension_version": feature_support.extension_version,
            }
            conn.commit()
            return [
                (
                    RecordHit(
                        RecordIdentity(row[0], row[1], row[2]),
                        1.0 - float(row[3]),
                    )
                )
                for row in results[:k]
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
            graph_changed = _delete_graph_edges_for_identities(
                cursor,
                [RecordIdentity.from_storage_key(storage_id) for storage_id in storage_ids],
            )
            cursor.execute("SELECT DISTINCT table_name FROM vector_tables;")
            table_names = [row[0] for row in cursor.fetchall()]

            for table_name in table_names:
                cursor.execute(
                    self._sql.SQL("DELETE FROM {table} WHERE record_id = ANY(%s);").format(
                        table=self._sql.Identifier(table_name)
                    ),
                    (storage_ids,),
                )

            cursor.execute(
                "DELETE FROM records WHERE record_id = ANY(%s);", (storage_ids,)
            )

            _POSTGRES_EPOCH_LANE.bump(
                cursor,
                keyword=True,
                vector=True,
                graph=graph_changed,
            )

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
                self._sql.SQL("DELETE FROM {table} WHERE record_id = ANY(%s);").format(
                    table=self._sql.Identifier(table_name)
                ),
                (storage_ids,),
            )

            cursor.execute("SELECT table_name FROM vector_tables;")
            table_names = [table_row[0] for table_row in cursor.fetchall()]
            remaining_checks = [
                self._sql.SQL(
                    "NOT EXISTS (SELECT 1 FROM {table} v "
                    "WHERE v.record_id = r.record_id)"
                ).format(table=self._sql.Identifier(other_table))
                for other_table in table_names
            ]
            if remaining_checks:
                remaining_clause = self._sql.SQL(" AND ").join(remaining_checks)
                cursor.execute(
                    self._sql.SQL(
                        "SELECT r.workspace_id, r.source_kind, r.source_id "
                        "FROM records r "
                        "WHERE r.record_id = ANY(%s) AND {}"
                    ).format(remaining_clause),
                    (storage_ids,),
                )
                deleted_identities = [
                    RecordIdentity(workspace_id or None, source_kind, source_id)
                    for workspace_id, source_kind, source_id in cursor.fetchall()
                ]
                graph_changed = _delete_graph_edges_for_identities(
                    cursor, deleted_identities
                )
                cursor.execute(
                    self._sql.SQL(
                        "DELETE FROM records r WHERE r.record_id = ANY(%s) AND {}"
                    ).format(remaining_clause),
                    (storage_ids,),
                )
            else:
                graph_changed = False
                cursor.execute(
                    "DELETE FROM records WHERE record_id = ANY(%s);",
                    (storage_ids,),
                )

            _POSTGRES_EPOCH_LANE.bump(
                cursor,
                keyword=True,
                vector=True,
                graph=graph_changed,
            )
            conn.commit()
        finally:
            if cursor is not None:
                cursor.close()
            self.conn_pool.put_connection(conn)

    def epoch(self) -> int:
        """Get current index epoch."""
        return _POSTGRES_EPOCH_LANE.read_total(self.conn_pool)

    def vector_epoch(self) -> int:
        return _POSTGRES_EPOCH_LANE.read(self.conn_pool, "vector")

    def epochs(self) -> dict[str, int]:
        return _POSTGRES_EPOCH_LANE.read_all(self.conn_pool)


class PGKeywordStore:
    """Postgres full-text search implementation of KeywordStore port."""

    def __init__(self, conn_pool: _PostgresConnectionLike):
        """Initialize keyword store.

        Args:
            conn_pool: PostgresConnection or Psycopg3Connection pool
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

            rows = [
                (
                    record.storage_key,
                    f"{record.title} {record.indexed_text or record.body}",
                )
                for record in records
            ]

            if isinstance(self.conn_pool, Psycopg3Connection):
                cursor.executemany(
                    """
                    UPDATE records AS r
                    SET tsvector_body = to_tsvector('english', %s)
                    WHERE r.record_id = %s;
                    """,
                    [(text, record_id) for record_id, text in rows],
                )
            else:
                psycopg2.extras.execute_values(
                    cursor,
                    """
                    UPDATE records AS r
                    SET tsvector_body = to_tsvector('english', data.search_text)
                    FROM (VALUES %s) AS data(record_id, search_text)
                    WHERE r.record_id = data.record_id;
                    """,
                    rows,
                )

            _POSTGRES_EPOCH_LANE.bump(cursor, keyword=True)
            conn.commit()
            logger.debug(f"Indexed {len(records)} records for keyword search")
        finally:
            if cursor is not None:
                cursor.close()
            self.conn_pool.put_connection(conn)

    def keyword_epoch(self) -> int:
        return _POSTGRES_EPOCH_LANE.read(self.conn_pool, "keyword")

    def epoch(self) -> int:
        return _POSTGRES_EPOCH_LANE.read_total(self.conn_pool)

    def epochs(self) -> dict[str, int]:
        return _POSTGRES_EPOCH_LANE.read_all(self.conn_pool)

    def search(
        self, query: str, k: int, filters: dict[str, Any] | None = None
    ) -> list[RecordHit]:
        """Search for records matching the query.

        Args:
            query: Full-text search query
            k: Number of results to return
            filters: Optional filters

        Returns:
            List of RecordHit values, sorted descending
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
        edges: Sequence[GraphEdge],
    ) -> None:
        """Upsert edges in the graph.

        Args:
            edges: GraphEdge values with complete endpoint identities
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

            _POSTGRES_EPOCH_LANE.bump(cursor, graph=True)
            conn.commit()
            logger.debug(f"Upserted {len(edges)} edges")
        finally:
            if cursor is not None:
                cursor.close()
            self.conn_pool.put_connection(conn)

    def delete_edges(
        self,
        edges: Sequence[GraphEdge],
    ) -> None:
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
                )
                for edge in edges
            ]
            cursor.executemany(
                """
                DELETE FROM graph_edges
                WHERE source_workspace_id = %s
                  AND source_kind = %s
                  AND source_id = %s
                  AND target_workspace_id = %s
                  AND target_kind = %s
                  AND target_id = %s
                  AND edge_type = %s;
                """,
                rows,
            )
            if cursor.rowcount:
                _POSTGRES_EPOCH_LANE.bump(cursor, graph=True)
            conn.commit()
        finally:
            if cursor is not None:
                cursor.close()
            self.conn_pool.put_connection(conn)

    def graph_epoch(self) -> int:
        return _POSTGRES_EPOCH_LANE.read(self.conn_pool, "graph")

    def epochs(self) -> dict[str, int]:
        return _POSTGRES_EPOCH_LANE.read_all(self.conn_pool)

    def neighbors(
        self,
        record_id: RecordIdentity,
        edge_types: list[str] | None = None,
        depth: int = 1,
    ) -> Sequence[GraphNeighbor]:
        """Retrieve neighbors of a record.

        Args:
            record_id: Starting record identity
            edge_types: Optional filter by edge type
            depth: Traversal depth (default 1 for one-hop)

        Returns:
            GraphNeighbor values with complete neighbor identities
        """
        if depth < 1:
            raise ValueError("depth must be positive")
        conn = self.conn_pool.get_connection()
        cursor = None
        try:
            cursor = conn.cursor()

            edge_filter = ""
            params: list[Any] = [
                record_id.workspace_id or "",
                record_id.source_kind,
                record_id.source_id,
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
                               jsonb_build_array(
                                   NULLIF(source_workspace_id, ''),
                                   source_kind,
                                   source_id
                               )::text,
                               jsonb_build_array(
                                   NULLIF(target_workspace_id, ''),
                                   target_kind,
                                   target_id
                               )::text
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
                               jsonb_build_array(
                                   NULLIF(edge.target_workspace_id, ''),
                                   edge.target_kind,
                                   edge.target_id
                               )::text
                           )
                    FROM walk
                    JOIN graph_edges edge
                      ON edge.source_workspace_id IS NOT DISTINCT FROM
                         walk.target_workspace_id
                     AND edge.source_kind = walk.target_kind
                     AND edge.source_id = walk.target_id
                    WHERE walk.hop < %s
                      AND NOT (
                          jsonb_build_array(
                              NULLIF(edge.target_workspace_id, ''),
                              edge.target_kind,
                              edge.target_id
                          )::text
                      ) = ANY(walk.path)
                      AND TRUE {edge_filter.replace("edge_type", "edge.edge_type")}
                ),
                ranked AS (
                    SELECT target_workspace_id, target_kind, target_id,
                           edge_type, weight,
                           ROW_NUMBER() OVER (
                               PARTITION BY target_workspace_id, target_kind,
                                            target_id
                               ORDER BY weight DESC, edge_type
                           ) AS row_number
                    FROM walk
                )
                SELECT target_workspace_id, target_kind, target_id,
                       edge_type, weight
                FROM ranked
                WHERE row_number = 1
                ORDER BY weight DESC, target_workspace_id, target_kind,
                         target_id, edge_type;
            """

            cursor.execute(sql, params)
            results = cursor.fetchall()
            return [
                GraphNeighbor(
                    RecordIdentity(row[0] or None, row[1], row[2]),
                    row[3],
                    float(row[4]),
                )
                for row in results
            ]
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
