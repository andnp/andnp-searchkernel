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
from typing import Any, LiteralString, Protocol

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
    from psycopg import sql as psycopg3_sql
except ImportError:
    psycopg = None  # type: ignore[assignment]
    psycopg3_sql = None
    psycopg_pool = None  # type: ignore[assignment]


def _require_psycopg2():
    if psycopg2 is None:
        raise ImportError(
            "psycopg2 is required for PostgreSQL operations. "
            "Install with: pip install 'andnp-searchkernel[pgvector]'"
        )
    return psycopg2


def _require_psycopg3():
    if psycopg is None:
        raise ImportError(
            "psycopg3 is required for PostgreSQL operations. "
            "Install with: pip install 'andnp-searchkernel[pgvector-psycopg3]'"
        )
    return psycopg


def _require_psycopg2_sql():
    if sql is None:
        raise ImportError(
            "psycopg2 is required for PostgreSQL SQL composition. "
            "Install with: pip install 'andnp-searchkernel[pgvector]'"
        )
    return sql


def _require_psycopg3_sql():
    if psycopg3_sql is None:
        raise ImportError(
            "psycopg3 is required for PostgreSQL SQL composition. "
            "Install with: pip install 'andnp-searchkernel[pgvector-psycopg3]'"
        )
    return psycopg3_sql

from searchkernel.adapters.stores.postgres_epochs import (
    _POSTGRES_EPOCH_LANE,
    _PostgresConnectionLike,
)
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
    compile_source_scoped_filters,
    filter_values,
    status_values,
)
from searchkernel.indices import keyword_scoring as _keyword_scoring
from searchkernel.indices.vector_revision import record_embedding_revision
from searchkernel.ports.keyword_scoring import KeywordArtifactScorer

logger = logging.getLogger(__name__)

_IDENT_RE = re.compile(r"[^a-z0-9_]+")
_METADATA_FIELD_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

# Default HNSW query-time recall knob. Higher = better recall, more latency.
DEFAULT_HNSW_EF_SEARCH = 100
DEFAULT_HNSW_ITERATIVE_SCAN = "auto"
DEFAULT_HNSW_MAX_SCAN_TUPLES = 20_000
DEFAULT_HNSW_SCAN_MEM_MULTIPLIER = 1.0
DEFAULT_VECTOR_OVERFETCH_MULTIPLIER = 2.0
DEFAULT_VECTOR_MAX_SCAN_ROUNDS = 4
_SCHEMA_ADVISORY_LOCK_KEY = 907341005
_ITERATIVE_SCAN_MODES = {"auto", "off", "strict_order", "relaxed_order"}
_VECTOR_WRITE_BATCH_SIZE = 100
_KEYWORD_WRITE_BATCH_SIZE = 100


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
    components: list[str] = []
    for value in vec:
        try:
            component = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("embedding values must be finite numbers") from exc
        if not math.isfinite(component):
            raise ValueError("embedding values must be finite numbers")
        components.append(repr(component))
    return "[" + ",".join(components) + "]"


def _parse_vector_literal(value: str) -> Vector:
    """Parse pgvector's `[v1,v2,...]` text format back into a Python vector."""
    return [float(component) for component in value.strip("[]").split(",")]


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
    where each key-value pair becomes a metadata->>'key' = value filter, and
    via the 'metadata_in' dict, where each key maps to a list of allowed
    values (metadata->>'key' = ANY(values)); fields in 'metadata_in' are
    ANDed together, while the values for a single field are ORed.

    'exclude_storage_keys' (symmetric to 'candidate_ids'/
    'candidate_storage_keys') REQUIRES its values to already be canonical
    `record:`-prefixed storage keys, or `RecordIdentity` instances - a bare
    external id (e.g. an issue key like "ENG-123") is NOT canonicalized by
    this function and is silently dropped, exactly like the existing
    'candidate_ids'/'candidate_storage_keys' inclusion filter. Callers must
    build the canonical storage key themselves (e.g. via
    `RecordIdentity(workspace_id, source_kind, source_id).storage_key`)
    before passing it in.
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

    for source_filter in compile_source_scoped_filters(filters):
        if source_filter.workspace_ids is not None:
            if not source_filter.workspace_ids:
                clauses.append(f"{record_alias}.source_kind <> %s")
                parameters.append(source_filter.source_kind)
            else:
                clauses.append(
                    f"({record_alias}.source_kind <> %s OR "
                    f"{record_alias}.workspace_id = ANY(%s))"
                )
                parameters.extend(
                    [source_filter.source_kind, list(source_filter.workspace_ids)]
                )
        for field in source_filter.metadata_non_empty:
            clauses.append(
                f"({record_alias}.source_kind <> %s OR ("
                f"jsonb_typeof({record_alias}.metadata -> %s) = 'array' AND "
                f"jsonb_array_length({record_alias}.metadata -> %s) > 0))"
            )
            parameters.extend([source_filter.source_kind, field, field])
        for field, allowed_values in source_filter.metadata_contains_any:
            if not allowed_values:
                clauses.append(f"{record_alias}.source_kind <> %s")
                parameters.append(source_filter.source_kind)
                continue
            clauses.append(
                f"({record_alias}.source_kind <> %s OR EXISTS ("
                "SELECT 1 FROM jsonb_array_elements("
                f"CASE WHEN jsonb_typeof({record_alias}.metadata -> %s) = 'array' "
                f"THEN {record_alias}.metadata -> %s ELSE '[]'::jsonb END"
                ") AS scoped_value(value) "
                "WHERE jsonb_typeof(scoped_value.value) = 'string' "
                "AND scoped_value.value #>> '{}' = ANY(%s)))"
            )
            parameters.extend(
                [source_filter.source_kind, field, field, list(allowed_values)]
            )

    candidate_value = filters.get("candidate_ids")
    if candidate_value is None:
        candidate_value = filters.get("candidate_storage_keys")
    if candidate_value is not None:
        candidate_keys = candidate_storage_keys(candidate_value)
        if not candidate_keys:
            return ["FALSE"], []
        clauses.append(f"{record_alias}.record_id = ANY(%s)")
        parameters.append(sorted(candidate_keys))

    exclude_value = filters.get("exclude_storage_keys")
    if exclude_value is not None:
        excluded_keys = sorted(candidate_storage_keys(exclude_value))
        if excluded_keys:
            clauses.append(f"{record_alias}.record_id <> ALL(%s)")
            parameters.append(excluded_keys)

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
                if not _METADATA_FIELD_RE.match(field):
                    raise ValueError(
                        f"metadata_equals field {field!r} must match "
                        f"{_METADATA_FIELD_RE.pattern!r}"
                    )
                field_expr = f"{record_alias}.metadata->>'{field}'"
                clauses.append(f"{field_expr} = %s")
                parameters.append(str(value))

    metadata_in = filters.get("metadata_in")
    if metadata_in is not None:
        for field, values in metadata_in.items():
            if not _METADATA_FIELD_RE.match(field):
                raise ValueError(
                    f"metadata_in field {field!r} must match "
                    f"{_METADATA_FIELD_RE.pattern!r}"
                )
            allowed_values = [str(value) for value in values]
            if not allowed_values:
                return ["FALSE"], []
            field_expr = f"{record_alias}.metadata->>'{field}'"
            clauses.append(f"{field_expr} = ANY(%s)")
            parameters.append(allowed_values)

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
    _require_psycopg2().extras.execute_values(
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
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_graph_edges_target_identity "
        "ON graph_edges (target_workspace_id, target_kind, target_id);"
    )


def create_schema(conn_pool: _PostgresConnectionLike) -> None:
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


_create_schema = create_schema


class PostgresConnection:
    """Thread-safe Postgres connection pool."""

    def __init__(self, dsn: str, min_connections: int = 2, max_connections: int = 10):
        """Initialize connection pool."""
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
        except _require_psycopg2().Error:
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
    """Thread-safe Postgres connection pool using psycopg3."""

    def __init__(self, dsn: str, min_connections: int = 2, max_connections: int = 10):
        """Initialize connection pool."""
        if psycopg_pool is None:
            raise ImportError(
                "psycopg3 and psycopg_pool are required for Psycopg3Connection. "
                "Install with: pip install 'andnp-searchkernel[pgvector-psycopg3]'"
            )

        self.dsn = dsn
        self.pool = psycopg_pool.ConnectionPool(
            dsn,
            min_size=min_connections,
            max_size=max_connections,
            open=True,
            kwargs={"prepare_threshold": None},
        )

    def get_connection(self):
        """Get a connection from the pool."""
        return self.pool.getconn()

    def put_connection(self, conn):
        """Return a connection to the pool."""
        try:
            conn.rollback()
        except _require_psycopg3().Error as e:
            logger.debug("Connection rollback failed during pool return: %s", e)
        self.pool.putconn(conn)

    def execute(self, sql: str, params: tuple = ()) -> Any:
        """Execute a query and return results."""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(sql.encode("utf-8"), params)
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




class _SQLFragment(Protocol):
    value: object

    def format(
        self, *args: _SQLFragment, **kwargs: _SQLFragment
    ) -> _SQLFragment: ...

    def join(self, values: Sequence[_SQLFragment]) -> _SQLFragment: ...

    def as_string(self, cursor) -> str: ...


class _SQLBuilder(Protocol):
    def SQL(self, value: LiteralString) -> _SQLFragment: ...

    def Identifier(self, *values: str) -> _SQLFragment: ...

    def Integer(self, value: int) -> _SQLFragment: ...


def _psycopg2_composable(value: object):
    composable_type = _require_psycopg2_sql().Composable
    if not isinstance(value, composable_type):
        raise TypeError("psycopg2 SQL fragments cannot be mixed with psycopg3")
    return value


def _psycopg2_sql(value: object):
    sql_type = _require_psycopg2_sql().SQL
    if not isinstance(value, sql_type):
        raise TypeError("psycopg2 SQL formatting requires an SQL template")
    return value


def _psycopg3_composable(value: object):
    composable_type = _require_psycopg3_sql().Composable
    if not isinstance(value, composable_type):
        raise TypeError("psycopg3 SQL fragments cannot be mixed with psycopg2")
    return value


def _psycopg3_sql(value: object):
    sql_type = _require_psycopg3_sql().SQL
    if not isinstance(value, sql_type):
        raise TypeError("psycopg3 SQL formatting requires an SQL template")
    return value


class _Psycopg2Fragment:
    def __init__(self, value: object) -> None:
        self.value = value

    def format(
        self, *args: _SQLFragment, **kwargs: _SQLFragment
    ) -> _SQLFragment:
        positional = [_psycopg2_composable(fragment.value) for fragment in args]
        named = {
            name: _psycopg2_composable(fragment.value)
            for name, fragment in kwargs.items()
        }
        return _Psycopg2Fragment(
            _psycopg2_sql(self.value).format(*positional, **named)
        )

    def join(self, values: Sequence[_SQLFragment]) -> _SQLFragment:
        return _Psycopg2Fragment(
            _psycopg2_sql(self.value).join(
                [_psycopg2_composable(value.value) for value in values]
            )
        )

    def as_string(self, cursor) -> str:
        return _psycopg2_composable(self.value).as_string(cursor)


class _Psycopg3Fragment:
    def __init__(self, value: object) -> None:
        self.value = value

    def format(
        self, *args: _SQLFragment, **kwargs: _SQLFragment
    ) -> _SQLFragment:
        positional = [_psycopg3_composable(fragment.value) for fragment in args]
        named = {
            name: _psycopg3_composable(fragment.value)
            for name, fragment in kwargs.items()
        }
        return _Psycopg3Fragment(
            _psycopg3_sql(self.value).format(*positional, **named)
        )

    def join(self, values: Sequence[_SQLFragment]) -> _SQLFragment:
        return _Psycopg3Fragment(
            _psycopg3_sql(self.value).join(
                [_psycopg3_composable(value.value) for value in values]
            )
        )

    def as_string(self, cursor) -> str:
        return _psycopg3_composable(self.value).as_string(cursor)


def _execute(
    cursor,
    statement: str | _SQLFragment,
    params: object | None = None,
) -> None:
    statement_value: object = statement
    if isinstance(statement, (_Psycopg2Fragment, _Psycopg3Fragment)):
        statement_value = statement.value
    if params is None:
        cursor.execute(statement_value)
    else:
        cursor.execute(statement_value, params)


class _Psycopg2SQLBuilder:
    def SQL(self, value: LiteralString) -> _SQLFragment:
        return _Psycopg2Fragment(_require_psycopg2_sql().SQL(value))

    def Identifier(self, *values: str) -> _SQLFragment:
        return _Psycopg2Fragment(_require_psycopg2_sql().Identifier(*values))

    def Integer(self, value: int) -> _SQLFragment:
        return _Psycopg2Fragment(_require_psycopg2_sql().SQL(str(value)))


class _Psycopg3SQLBuilder:
    def SQL(self, value: LiteralString) -> _SQLFragment:
        return _Psycopg3Fragment(_require_psycopg3_sql().SQL(value))

    def Identifier(self, *values: str) -> _SQLFragment:
        return _Psycopg3Fragment(_require_psycopg3_sql().Identifier(*values))

    def Integer(self, value: int) -> _SQLFragment:
        return _Psycopg3Fragment(
            _require_psycopg3_sql().SQL("{}").format(
                _require_psycopg3_sql().Literal(value)
            )
        )


def _sql_module_for(conn_pool: _PostgresConnectionLike) -> _SQLBuilder:
    """Return a driver-specific SQL builder for the connection pool."""
    if isinstance(conn_pool, Psycopg3Connection):
        return _Psycopg3SQLBuilder()
    return _Psycopg2SQLBuilder()


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
        self._vector_tables: dict[str, tuple[int, str]] = {}
        self._revision_ready_tables: set[str] = set()
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
        cached_table = self._vector_tables.get(model_name)
        if cached_table is not None:
            existing_dim, existing_table = cached_table
            if existing_dim != dim:
                raise ModelDimensionMismatchError(
                    f"Dimension mismatch for model {model_name}: "
                    f"expected {existing_dim}, got {dim}"
                )
            _execute(
                cursor,
                self._sql.SQL(
                    "ALTER TABLE {table} ADD COLUMN IF NOT EXISTS revision TEXT;"
                ).format(table=self._sql.Identifier(existing_table))
            )
            return existing_table

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
            _execute(
                cursor,
                self._sql.SQL(
                    "ALTER TABLE {table} ADD COLUMN IF NOT EXISTS revision TEXT;"
                ).format(table=self._sql.Identifier(existing_table))
            )
            return existing_table

        table_name = _vector_table_name(model_name, dim)

        _execute(
            cursor,
            self._sql.SQL(
                "CREATE TABLE IF NOT EXISTS {table} ("
                "record_id TEXT PRIMARY KEY, "
                "embedding vector({dim}) NOT NULL, "
                "created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP, "
                "updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP"
                ");"
            ).format(
                table=self._sql.Identifier(table_name),
                dim=self._sql.Integer(dim),
            )
        )
        _execute(
            cursor,
            self._sql.SQL(
                "ALTER TABLE {table} ADD COLUMN IF NOT EXISTS revision TEXT;"
            ).format(table=self._sql.Identifier(table_name))
        )

        _execute(
            cursor,
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

        The records and vector rows commit together, while the existing
        keyword projection remains owned by :meth:`PGKeywordStore.index`.

        Args:
            records: Records with embedding set
            model_name: Embedding model name
            dim: Vector dimensionality

        Raises:
            ValueError: If a record has no embedding or its dimension is invalid
        """
        if not records:
            return

        if dim < 1:
            raise ValueError("dim must be positive")
        for record in records:
            if record.embedding is None:
                raise ValueError(
                    f"Record {record.storage_key} must have an embedding"
                )
            if len(record.embedding) != dim:
                raise ValueError(
                    f"Embedding dimension mismatch for record {record.storage_key}: "
                    f"expected {dim}, got {len(record.embedding)}"
                )
        records = list({record.storage_key: record for record in records}.values())

        conn = self.conn_pool.get_connection()
        cursor = None
        try:
            cursor = conn.cursor()

            table_name = self._ensure_vector_table(cursor, model_name, dim)

            # Upsert records table in one driver-native bulk operation.
            record_rows = []
            for record in records:
                metadata_json = json.dumps(record.metadata)
                indexed_text = record.indexed_text or record.body
                tsvector_text = f"{record.title} {indexed_text}"
                record_key = record.storage_key
                record_rows.append(
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
                    )
                )

            record_insert_sql = """
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
                    tsvector_body = records.tsvector_body,
                    created_at = EXCLUDED.created_at,
                    updated_at = EXCLUDED.updated_at,
                    metadata = EXCLUDED.metadata,
                    uri = EXCLUDED.uri,
                    status = EXCLUDED.status;
                """
            if isinstance(self.conn_pool, Psycopg3Connection):
                cursor.executemany(record_insert_sql, record_rows)
            else:
                _require_psycopg2().extras.execute_values(
                    cursor,
                    record_insert_sql.replace(
                        "VALUES (%s, %s, %s, %s, %s, %s, %s,\n"
                        "                        to_tsvector('english', %s), %s, %s, %s, %s, %s)",
                        "VALUES %s",
                    ),
                    record_rows,
                    template=(
                        "(%s, %s, %s, %s, %s, %s, %s, "
                        "to_tsvector('english', %s), %s, %s, %s, %s, %s)"
                    ),
                )

            # Upsert only vector rows whose revision or payload differs.
            vector_rows = []
            for record in records:
                embedding = record.embedding
                if embedding is None:
                    raise ValueError(
                        f"Record {record.storage_key} must have an embedding"
                    )
                vector_rows.append(
                    (
                        record.storage_key,
                        _vector_literal(embedding),
                        record_embedding_revision(record, model_name, dim),
                    )
                )
            table_sql = self._sql.Identifier(table_name).as_string(cursor)
            vector_insert_sql = f"""
                INSERT INTO {table_sql} AS existing (record_id, embedding, revision)
                VALUES (%s, %s::vector, %s)
                ON CONFLICT (record_id) DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    revision = EXCLUDED.revision,
                    updated_at = CURRENT_TIMESTAMP
                WHERE existing.revision IS DISTINCT FROM EXCLUDED.revision
                   OR vector_dims(existing.embedding) IS DISTINCT FROM
                      vector_dims(EXCLUDED.embedding)
                   OR existing.embedding::text IS DISTINCT FROM
                      EXCLUDED.embedding::text
                RETURNING record_id;
                """
            changed_vector_ids: set[str] = set()
            if vector_rows:
                if isinstance(self.conn_pool, Psycopg3Connection):
                    for offset in range(0, len(vector_rows), _VECTOR_WRITE_BATCH_SIZE):
                        batch = vector_rows[offset : offset + _VECTOR_WRITE_BATCH_SIZE]
                        values = ", ".join(
                            "(%s, %s::vector, %s)" for _ in batch
                        )
                        batch_sql = vector_insert_sql.replace(
                            "VALUES (%s, %s::vector, %s)", f"VALUES {values}"
                        )
                        cursor.execute(
                            batch_sql,
                            tuple(value for row in batch for value in row),
                        )
                        changed_vector_ids.update(
                            row[0] for row in cursor.fetchall()
                        )
                else:
                    changed_rows = _require_psycopg2().extras.execute_values(
                        cursor,
                        vector_insert_sql.replace(
                            "VALUES (%s, %s::vector, %s)", "VALUES %s"
                        ),
                        vector_rows,
                        page_size=_VECTOR_WRITE_BATCH_SIZE,
                        fetch=True,
                    )
                    changed_vector_ids.update(row[0] for row in changed_rows)

            _POSTGRES_EPOCH_LANE.bump(
                cursor,
                vector=bool(changed_vector_ids),
            )

            conn.commit()
            self._vector_tables[model_name] = (dim, table_name)
            logger.debug(f"Upserted {len(records)} records for model {model_name}")
        except Exception:
            conn.rollback()
            raise
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

            cached_table = self._vector_tables.get(model_name)
            if cached_table is None:
                cursor.execute(
                    "SELECT 1 FROM vector_tables WHERE model_name = %s AND dim = %s;",
                    (model_name, dim),
                )
                if cursor.fetchone() is None:
                    conn.rollback()
                    self._last_search_diagnostics = {
                        "requested_k": k,
                        "returned": 0,
                        "scan_rounds": 0,
                        "under_returned": True,
                        "scan_bound_hit": False,
                    }
                    return []
                table_name = _vector_table_name(model_name, dim)
            else:
                registered_dim, table_name = cached_table
                if registered_dim != dim:
                    conn.rollback()
                    self._last_search_diagnostics = {
                        "requested_k": k,
                        "returned": 0,
                        "scan_rounds": 0,
                        "under_returned": True,
                        "scan_bound_hit": False,
                    }
                    return []

            feature_support = self._configure_hnsw(cursor)

            where_parts, filter_params = build_pgvector_filter_sql(filters)
            where_clause = "AND " + " AND ".join(where_parts)

            vec_literal = _vector_literal(query_vector)

            # Order by the raw distance operator (not a wrapped/aliased
            # expression) so the planner can use the HNSW index for ANN.
            query_sql = f"""
                SELECT r.workspace_id, r.source_kind, r.source_id,
                       v.embedding <=> %s::vector AS distance
                FROM {table_name} v
                JOIN records r ON v.record_id = r.record_id
                WHERE 1 = 1 {where_clause}
                ORDER BY v.embedding <=> %s::vector ASC, v.record_id ASC
                LIMIT %s;
                """

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
                if isinstance(self.conn_pool, Psycopg3Connection):
                    cursor.execute(query_sql.encode("utf-8"), params)
                else:
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
            self._vector_tables[model_name] = (dim, table_name)
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
                _execute(
                    cursor,
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

            _execute(
                cursor,
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
                _execute(
                    cursor,
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
                _execute(
                    cursor,
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

    def get_many(self, records: list[Record], model_name: str, dim: int) -> dict[str, Vector]:
        """Return stored embeddings that are still valid for the given records.

        A record's embedding is included only when its stored revision matches
        the record's current embedding revision -- i.e. nothing about the
        record has changed since it was last embedded. Records with no stored
        vector, or a stale one, are omitted so callers know to embed them
        fresh instead of reusing stale data.
        """
        if not records:
            return {}
        if dim < 1:
            raise ValueError("dim must be positive")

        conn = self.conn_pool.get_connection()
        cursor = None
        try:
            cursor = conn.cursor()
            cached_table = self._vector_tables.get(model_name)
            if cached_table is None:
                cursor.execute(
                    "SELECT dim, table_name FROM vector_tables WHERE model_name = %s;",
                    (model_name,),
                )
                row = cursor.fetchone()
                if row is None:
                    return {}
                existing_dim, table_name = row
                self._vector_tables[model_name] = (existing_dim, table_name)
            else:
                existing_dim, table_name = cached_table
            if existing_dim != dim:
                raise ModelDimensionMismatchError(
                    f"Dimension mismatch for model {model_name}: "
                    f"expected {existing_dim}, got {dim}"
                )
            # A table created before revision tracking existed has no such
            # column; add it (idempotent) rather than assume every existing
            # deployment has already upserted since that column was added.
            if table_name not in self._revision_ready_tables:
                _execute(
                    cursor,
                    self._sql.SQL(
                        "ALTER TABLE {table} ADD COLUMN IF NOT EXISTS revision TEXT;"
                    ).format(table=self._sql.Identifier(table_name))
                )

            storage_keys = [record.storage_key for record in records]
            _execute(
                cursor,
                self._sql.SQL(
                    "SELECT record_id, embedding::text, revision FROM {table} "
                    "WHERE record_id = ANY(%s);"
                ).format(table=self._sql.Identifier(table_name)),
                (storage_keys,),
            )
            stored = {row[0]: (row[1], row[2]) for row in cursor.fetchall()}
            conn.commit()
            self._revision_ready_tables.add(table_name)
        finally:
            if cursor is not None:
                cursor.close()
            self.conn_pool.put_connection(conn)

        result: dict[str, Vector] = {}
        for record in records:
            entry = stored.get(record.storage_key)
            if entry is None:
                continue
            embedding_text, revision = entry
            if revision != record_embedding_revision(record, model_name, dim):
                continue
            result[record.storage_key] = _parse_vector_literal(embedding_text)
        return result


def _postgres_tsquery(query: str) -> tuple[str, str]:
    """Choose a parameterized PostgreSQL query shape for lexical intent."""
    stripped = query.strip()
    if stripped.startswith('"') and stripped.endswith('"'):
        return "phraseto_tsquery('english', %s)", stripped[1:-1]
    if stripped.endswith("*") and re.fullmatch(r"[\w]+\*", stripped):
        return "to_tsquery('english', %s)", f"{stripped[:-1]}:*"
    return "plainto_tsquery('english', %s)", query


class PGKeywordStore:
    """Postgres full-text search implementation of KeywordStore port."""

    def __init__(
        self,
        conn_pool: _PostgresConnectionLike,
        *,
        artifact_scorer: KeywordArtifactScorer | None = None,
        keyword_overfetch_multiplier: float = 4.0,
    ):
        """Initialize keyword store.

        Args:
            conn_pool: PostgresConnection or Psycopg3Connection pool
            artifact_scorer: Optional identifier-aware reranker. Unlike
                LocalRecordBackend, this deliberately has no filesystem
                default -- generic Postgres consumers may want a scorer
                tuned to their own identifier shape (e.g. Jira keys) rather
                than file paths.
            keyword_overfetch_multiplier: How much to widen the base-relevance
                SQL LIMIT when a scorer identifies the query, so a row that
                the scorer would boost outside the base top-k can still
                surface. Mirrors LocalRecordBackend's default of the same name.
        """
        self.conn_pool = conn_pool
        self._artifact_scorer = artifact_scorer
        self._keyword_overfetch_multiplier = keyword_overfetch_multiplier

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

            records = list({record.storage_key: record for record in records}.values())
            rows = [
                (
                    record.storage_key,
                    record.title,
                    record.indexed_text or record.body,
                    " ".join(
                        (record.uri or "", json.dumps(record.metadata, sort_keys=True))
                    ),
                )
                for record in records
            ]

            weighted_tsvector = """
                setweight(to_tsvector('english', {title}), 'A') ||
                setweight(to_tsvector('english', {body}), 'B') ||
                setweight(to_tsvector('english', {rest}), 'D')
            """
            projection_sql = weighted_tsvector.format(
                title="data.title", body="data.body", rest="data.rest"
            )
            update_sql = f"""
                UPDATE records AS r
                SET tsvector_body = projection.tsvector_body
                FROM (VALUES %s) AS data(record_id, title, body, rest)
                CROSS JOIN LATERAL (
                    SELECT {projection_sql} AS tsvector_body
                ) AS projection
                WHERE r.record_id = data.record_id
                  AND r.tsvector_body IS DISTINCT FROM projection.tsvector_body
                RETURNING r.record_id;
                """

            changed_keyword_ids: set[str] = set()
            if isinstance(self.conn_pool, Psycopg3Connection):
                for offset in range(0, len(rows), _KEYWORD_WRITE_BATCH_SIZE):
                    batch = rows[offset : offset + _KEYWORD_WRITE_BATCH_SIZE]
                    values = ", ".join("(%s, %s, %s, %s)" for _ in batch)
                    batch_sql = update_sql.replace("VALUES %s", f"VALUES {values}")
                    cursor.execute(
                        batch_sql,
                        tuple(value for row in batch for value in row),
                    )
                    changed_keyword_ids.update(row[0] for row in cursor.fetchall())
            else:
                changed_rows = _require_psycopg2().extras.execute_values(
                    cursor,
                    update_sql,
                    rows,
                    page_size=_KEYWORD_WRITE_BATCH_SIZE,
                    fetch=True,
                )
                changed_keyword_ids.update(row[0] for row in changed_rows)

            _POSTGRES_EPOCH_LANE.bump(cursor, keyword=bool(changed_keyword_ids))
            conn.commit()
            logger.debug(f"Indexed {len(records)} records for keyword search")
        except Exception:
            conn.rollback()
            raise
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

            where_parts, filter_params = build_pgvector_filter_sql(
                filters, record_alias="r"
            )
            where_clause = "AND " + " AND ".join(where_parts)

            needs_artifact_rerank = (
                self._artifact_scorer is not None
                and self._artifact_scorer.looks_like_identifier_query(query)
            )
            limit = k
            if needs_artifact_rerank:
                limit = max(k, math.ceil(k * self._keyword_overfetch_multiplier))

            query_expression, query_value = _postgres_tsquery(query)
            projection = (
                "r.workspace_id, r.source_kind, r.source_id, "
                "ts_rank(r.tsvector_body, search_query.query) AS relevance, "
                "r.record_id, r.title, r.body, r.indexed_text, r.uri, r.metadata"
                if needs_artifact_rerank
                else "r.workspace_id, r.source_kind, r.source_id, "
                "ts_rank(r.tsvector_body, search_query.query) AS relevance"
            )
            sql = f"""
                WITH search_query AS (
                    SELECT {query_expression} AS query
                )
                SELECT {projection}
                FROM records AS r
                CROSS JOIN search_query
                WHERE r.tsvector_body @@ search_query.query
                {where_clause}
                ORDER BY relevance DESC, r.record_id ASC
                LIMIT %s;
            """
            params = [query_value, *filter_params, limit]

            cursor.execute(sql, params)
            results = cursor.fetchall()
            hits = [
                RecordHit(
                    RecordIdentity(row[0], row[1], row[2]),
                    float(row[3]),
                )
                for row in results
            ]
            if needs_artifact_rerank:
                assert self._artifact_scorer is not None
                boosted: list[RecordHit] = []
                for row, hit in zip(results, hits, strict=True):
                    metadata = row[9] or {}
                    boost = self._artifact_scorer.score(
                        query,
                        title=row[5],
                        body=row[6],
                        indexed_text=row[7],
                        headers=_keyword_scoring.metadata_keyword_text(metadata),
                        uri=row[8] or _keyword_scoring.metadata_uri(metadata),
                    )
                    boosted.append(RecordHit(hit.identity, hit.score + boost))
                boosted.sort(key=lambda item: (-item.score, item.storage_key))
                hits = boosted[:k]
            return hits
        finally:
            if cursor is not None:
                cursor.close()
            self.conn_pool.put_connection(conn)


class PGGraphStore:
    """Postgres implementation of GraphStore port."""

    def __init__(self, conn_pool: _PostgresConnectionLike):
        """Initialize graph store.

        Args:
            conn_pool: PostgreSQL connection pool
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
            statement = """
                INSERT INTO graph_edges (
                    source_workspace_id, source_kind, source_id,
                    target_workspace_id, target_kind, target_id,
                    edge_type, weight
                )
                VALUES %s
                ON CONFLICT ON CONSTRAINT graph_edges_identity_unique DO UPDATE SET
                    weight = EXCLUDED.weight,
                    updated_at = CURRENT_TIMESTAMP;
            """
            if isinstance(self.conn_pool, Psycopg3Connection):
                cursor.executemany(
                    statement.replace("VALUES %s", "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"),
                    rows,
                )
            else:
                _require_psycopg2().extras.execute_values(cursor, statement, rows)

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
        max_neighbors: int | None = None,
    ) -> Sequence[GraphNeighbor]:
        return self._neighbors_direction(
            record_id,
            edge_types,
            depth,
            max_neighbors,
            incoming=False,
        )

    def incoming_neighbors(
        self,
        record_id: RecordIdentity,
        edge_types: list[str] | None = None,
        depth: int = 1,
        max_neighbors: int | None = None,
    ) -> Sequence[GraphNeighbor]:
        return self._neighbors_direction(
            record_id,
            edge_types,
            depth,
            max_neighbors,
            incoming=True,
        )

    def _neighbors_direction(
        self,
        record_id: RecordIdentity,
        edge_types: list[str] | None = None,
        depth: int = 1,
        max_neighbors: int | None = None,
        *,
        incoming: bool,
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
        if max_neighbors is not None and max_neighbors <= 0:
            raise ValueError("max_neighbors must be positive")
        seed_workspace = "source_workspace_id" if not incoming else "target_workspace_id"
        seed_kind = "source_kind" if not incoming else "target_kind"
        seed_id = "source_id" if not incoming else "target_id"
        neighbor_workspace = "target_workspace_id" if not incoming else "source_workspace_id"
        neighbor_kind = "target_kind" if not incoming else "source_kind"
        neighbor_id = "target_id" if not incoming else "source_id"
        walk_workspace = "target_workspace_id" if not incoming else "source_workspace_id"
        walk_kind = "target_kind" if not incoming else "source_kind"
        walk_id = "target_id" if not incoming else "source_id"
        join_workspace = "source_workspace_id" if not incoming else "target_workspace_id"
        join_kind = "source_kind" if not incoming else "target_kind"
        join_id = "source_id" if not incoming else "target_id"
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
            if max_neighbors is not None:
                params.append(max_neighbors)

            limit_clause = "LIMIT %s" if max_neighbors is not None else ""

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
                    WHERE {seed_workspace} IS NOT DISTINCT FROM %s
                      AND {seed_kind} = %s
                      AND {seed_id} = %s
                      {edge_filter}
                    UNION ALL
                    SELECT walk.source_workspace_id, walk.source_kind,
                           walk.source_id, edge.target_workspace_id,
                           edge.target_kind, edge.target_id, edge.edge_type,
                           walk.weight * edge.weight, walk.hop + 1,
                           walk.path || (
                               jsonb_build_array(
                                   NULLIF(edge.{neighbor_workspace}, ''),
                                   edge.{neighbor_kind},
                                   edge.{neighbor_id}
                               )::text
                           )
                    FROM walk
                    JOIN graph_edges edge
                      ON edge.{join_workspace} IS NOT DISTINCT FROM
                         walk.{walk_workspace}
                     AND edge.{join_kind} = walk.{walk_kind}
                     AND edge.{join_id} = walk.{walk_id}
                    WHERE walk.hop < %s
                      AND NOT (
                          jsonb_build_array(
                              NULLIF(edge.{neighbor_workspace}, ''),
                              edge.{neighbor_kind},
                              edge.{neighbor_id}
                          )::text
                      ) = ANY(walk.path)
                      AND TRUE {edge_filter.replace("edge_type", "edge.edge_type")}
                ),
                ranked AS (
                    SELECT {neighbor_workspace}, {neighbor_kind}, {neighbor_id},
                           edge_type, weight,
                           ROW_NUMBER() OVER (
                               PARTITION BY {neighbor_workspace}, {neighbor_kind},
                                            {neighbor_id}
                               ORDER BY weight DESC, edge_type
                           ) AS row_number
                    FROM walk
                )
                SELECT {neighbor_workspace}, {neighbor_kind}, {neighbor_id},
                       edge_type, weight
                FROM ranked
                WHERE row_number = 1
                ORDER BY weight DESC, {neighbor_workspace}, {neighbor_kind},
                         {neighbor_id}, edge_type
                {limit_clause};
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


    def neighbors_many(
        self,
        identities: Sequence[RecordIdentity],
        *,
        depth: int,
        max_neighbors: int | None = None,
    ) -> dict[str, list[GraphNeighbor]]:
        return self._neighbors_many_direction(
            identities,
            depth=depth,
            max_neighbors=max_neighbors,
            incoming=False,
        )

    def incoming_neighbors_many(
        self,
        identities: Sequence[RecordIdentity],
        *,
        depth: int,
        max_neighbors: int | None = None,
    ) -> dict[str, list[GraphNeighbor]]:
        return self._neighbors_many_direction(
            identities,
            depth=depth,
            max_neighbors=max_neighbors,
            incoming=True,
        )

    def _neighbors_many_direction(
        self,
        identities: Sequence[RecordIdentity],
        *,
        depth: int,
        max_neighbors: int | None,
        incoming: bool,
    ) -> dict[str, list[GraphNeighbor]]:
        """Retrieve neighbors for all seeds with one recursive query."""
        if depth < 1:
            raise ValueError("depth must be positive")
        if max_neighbors is not None and max_neighbors <= 0:
            raise ValueError("max_neighbors must be positive")
        seed_identities_by_key: dict[str, RecordIdentity] = {}
        for identity in identities:
            seed_identities_by_key.setdefault(identity.storage_key, identity)
        seed_identities = list(seed_identities_by_key.values())
        result: dict[str, list[GraphNeighbor]] = {
            identity.storage_key: [] for identity in seed_identities
        }
        if not seed_identities:
            return result

        conn = self.conn_pool.get_connection()
        cursor = None
        try:
            cursor = conn.cursor()
            values = ", ".join(["(%s, %s, %s)"] * len(seed_identities))
            seed_params = [
                value
                for identity in seed_identities
                for value in (
                    identity.workspace_id or "",
                    identity.source_kind,
                    identity.source_id,
                )
            ]
            params: list[Any] = seed_params + [depth]
            limit_clause = ""
            if max_neighbors is not None:
                limit_clause = "WHERE neighbor_number <= %s"
                params.append(max_neighbors)
            seed_workspace = "target_workspace_id" if incoming else "source_workspace_id"
            seed_kind = "target_kind" if incoming else "source_kind"
            seed_id = "target_id" if incoming else "source_id"
            neighbor_workspace = "source_workspace_id" if incoming else "target_workspace_id"
            neighbor_kind = "source_kind" if incoming else "target_kind"
            neighbor_id = "source_id" if incoming else "target_id"
            sql = f"""
                WITH RECURSIVE seeds(workspace_id, source_kind, source_id) AS (
                    VALUES {values}
                ), walk AS (
                    SELECT s.workspace_id AS seed_workspace_id,
                           s.source_kind AS seed_kind, s.source_id AS seed_id,
                           e.{neighbor_workspace}, e.{neighbor_kind}, e.{neighbor_id},
                           e.edge_type, e.weight, 1 AS hop,
                           ARRAY[concat_ws(E'\\x1f', e.source_workspace_id,
                                           e.source_kind, e.source_id),
                                 concat_ws(E'\\x1f', e.target_workspace_id,
                                           e.target_kind, e.target_id)] AS path
                    FROM seeds s
                    JOIN graph_edges e
                      ON e.{seed_workspace} = s.workspace_id
                     AND e.{seed_kind} = s.source_kind
                     AND e.{seed_id} = s.source_id
                    UNION ALL
                    SELECT w.seed_workspace_id, w.seed_kind, w.seed_id,
                           e.{neighbor_workspace}, e.{neighbor_kind}, e.{neighbor_id},
                           e.edge_type, w.weight * e.weight, w.hop + 1,
                           w.path || ARRAY[concat_ws(E'\\x1f',
                                                     e.source_workspace_id,
                                                     e.source_kind, e.source_id),
                                           concat_ws(E'\\x1f',
                                                     e.target_workspace_id,
                                                     e.target_kind, e.target_id)]
                    FROM walk w
                    JOIN graph_edges e
                      ON e.{seed_workspace} = w.{neighbor_workspace}
                     AND e.{seed_kind} = w.{neighbor_kind}
                     AND e.{seed_id} = w.{neighbor_id}
                    WHERE w.hop < %s
                      AND NOT concat_ws(E'\\x1f', e.{neighbor_workspace},
                                        e.{neighbor_kind}, e.{neighbor_id}) = ANY(w.path)
                ), best AS (
                    SELECT *, ROW_NUMBER() OVER (
                        PARTITION BY seed_workspace_id, seed_kind, seed_id,
                                     {neighbor_workspace}, {neighbor_kind}, {neighbor_id}
                        ORDER BY weight DESC, edge_type
                    ) AS target_number
                    FROM walk
                ), ranked AS (
                    SELECT *, ROW_NUMBER() OVER (
                        PARTITION BY seed_workspace_id, seed_kind, seed_id
                        ORDER BY weight DESC, {neighbor_workspace},
                                 {neighbor_kind}, {neighbor_id}, edge_type
                    ) AS neighbor_number
                    FROM best
                    WHERE target_number = 1
                )
                SELECT seed_workspace_id, seed_kind, seed_id,
                       {neighbor_workspace}, {neighbor_kind}, {neighbor_id},
                       edge_type, weight
                FROM ranked
                {limit_clause}
                ORDER BY seed_workspace_id, seed_kind, seed_id,
                         weight DESC, {neighbor_workspace}, {neighbor_kind},
                         {neighbor_id}, edge_type;
            """
            cursor.execute(sql, params)
            for row in cursor.fetchall():
                seed_key = RecordIdentity(
                    row[0] or None, row[1], row[2]
                ).storage_key
                result.setdefault(seed_key, []).append(
                    GraphNeighbor(
                        RecordIdentity(row[3] or None, row[4], row[5]),
                        row[6],
                        float(row[7]),
                    )
                )
            return result
        finally:
            if cursor is not None:
                cursor.close()
            self.conn_pool.put_connection(conn)


class PGCacheStore:
    """Postgres implementation of CacheStore port with epoch-based invalidation."""

    def __init__(self, conn_pool: _PostgresConnectionLike):
        """Initialize cache store.

        Args:
            conn_pool: PostgreSQL connection pool
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
