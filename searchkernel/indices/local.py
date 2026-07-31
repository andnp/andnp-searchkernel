"""SQLite-backed record stores for the local search backend.

The local backend deliberately speaks the same record-oriented ports as the
pgvector adapter.  Chunk-oriented FAISS/SQLite indices remain available for
the ingestion surface, but query execution uses canonical record identities.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from searchkernel.domain import (
    GraphEdge,
    GraphNeighbor,
    Record,
    RecordHit,
    RecordIdentity,
    RecordStatus,
    Vector,
)
from searchkernel.storage.db import DatabaseManager

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_LOCAL_KEYWORD_SCHEMA = "local_records_fts"
_LOCAL_KEYWORD_SCHEMA_VERSION = 1
_LOCAL_FTS_TABLE = "local_records_fts"
_LOCAL_FTS_COLUMNS = ("title", "body", "uri", "keywords")
_FALLBACK_SCAN_MAX_ROWS = 10_000


class _EphemeralDatabase:
    def __init__(self) -> None:
        self._connection = sqlite3.connect(":memory:", check_same_thread=False)
        self._connection.row_factory = sqlite3.Row

    def get_connection(self) -> sqlite3.Connection:
        return self._connection


class _Database(Protocol):
    def get_connection(self) -> sqlite3.Connection: ...


class LocalRecordBackend:
    """Shared durable state for the local vector, keyword, and graph stores."""

    def __init__(
        self,
        db_path: Path | None = None,
        *,
        db_manager: DatabaseManager | None = None,
    ) -> None:
        if db_manager is not None and db_path is not None:
            raise ValueError("pass db_path or db_manager, not both")
        self._db = db_manager or (
            DatabaseManager(db_path) if db_path is not None else _EphemeralDatabase()
        )
        self._lock = threading.RLock()
        self._fts5_available = False
        self._keyword_search_diagnostic = (
            "FTS5 indexed lexical search has not been initialized"
        )
        self._initialize_schema()

    @property
    def db_manager(self) -> _Database:
        return self._db

    def _initialize_schema(self) -> None:
        conn = self._db.get_connection()
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS local_records (
                storage_key TEXT PRIMARY KEY,
                workspace_id TEXT,
                source_kind TEXT NOT NULL,
                source_id TEXT NOT NULL,
                title TEXT NOT NULL,
                body TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata TEXT NOT NULL,
                uri TEXT,
                keywords TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_local_records_identity
                ON local_records (workspace_id, source_kind, source_id);
            CREATE INDEX IF NOT EXISTS idx_local_records_status
                ON local_records (status);
            CREATE INDEX IF NOT EXISTS idx_local_records_workspace
                ON local_records (workspace_id);
            CREATE TABLE IF NOT EXISTS local_vectors (
                storage_key TEXT NOT NULL,
                model_name TEXT NOT NULL,
                dim INTEGER NOT NULL,
                embedding TEXT NOT NULL,
                PRIMARY KEY (storage_key, model_name, dim)
            );
            CREATE TABLE IF NOT EXISTS local_graph_edges (
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                edge_type TEXT NOT NULL,
                weight REAL NOT NULL,
                PRIMARY KEY (source_id, target_id, edge_type)
            );
            CREATE INDEX IF NOT EXISTS idx_local_graph_source
                ON local_graph_edges (source_id);
            CREATE TABLE IF NOT EXISTS system_state (
                key TEXT PRIMARY KEY,
                value TEXT
            );
            """
        )
        self._ensure_local_record_column(conn, "keywords", "TEXT NOT NULL DEFAULT ''")
        self._initialize_keyword_schema(conn)
        conn.commit()

    @staticmethod
    def _ensure_local_record_column(
        conn: sqlite3.Connection,
        column: str,
        definition: str,
    ) -> None:
        columns = {
            row[1] for row in conn.execute("PRAGMA table_info(local_records)")
        }
        if column not in columns:
            conn.execute(f"ALTER TABLE local_records ADD COLUMN {column} {definition}")

    @staticmethod
    def _metadata_keyword_text(metadata: dict[str, Any]) -> str:
        values: list[str] = []
        for key in ("tags", "keywords", "source_keywords", "aliases"):
            value = metadata.get(key)
            if value is None:
                continue
            if isinstance(value, (list, tuple, set)):
                values.extend(str(item) for item in value)
            else:
                values.append(str(value))
        return " ".join(" ".join(value.strip().lower().split()) for value in values if value)

    @staticmethod
    def _record_uri(record: Record) -> str:
        if record.uri:
            return record.uri
        for key in ("uri", "source_file", "file_path", "path"):
            value = record.metadata.get(key)
            if value:
                return str(value)
        return ""

    @classmethod
    def _migrate_keyword_columns(cls, conn: sqlite3.Connection) -> None:
        rows = conn.execute(
            "SELECT rowid, metadata, keywords FROM local_records"
        ).fetchall()
        for row in rows:
            try:
                metadata = json.loads(row["metadata"])
            except (TypeError, ValueError):
                metadata = {}
            if not isinstance(metadata, dict):
                metadata = {}
            keywords = cls._metadata_keyword_text(metadata)
            if row["keywords"] != keywords:
                conn.execute(
                    "UPDATE local_records SET keywords = ? WHERE rowid = ?",
                    (keywords, row["rowid"]),
                )

    @staticmethod
    def _fts_table_columns(conn: sqlite3.Connection) -> tuple[str, ...] | None:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (_LOCAL_FTS_TABLE,),
        ).fetchone()
        if row is None:
            return None
        try:
            return tuple(
                column[1]
                for column in conn.execute(f"PRAGMA table_info({_LOCAL_FTS_TABLE})")
            )
        except sqlite3.DatabaseError:
            return None

    def _initialize_keyword_schema(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS local_keyword_schema (
                name TEXT PRIMARY KEY,
                version INTEGER NOT NULL
            )
            """
        )
        self._migrate_keyword_columns(conn)

        table_columns = self._fts_table_columns(conn)
        needs_rebuild = table_columns != _LOCAL_FTS_COLUMNS
        if table_columns is not None and needs_rebuild:
            conn.execute(f"DROP TABLE IF EXISTS {_LOCAL_FTS_TABLE}")

        if table_columns != _LOCAL_FTS_COLUMNS:
            try:
                conn.execute(
                    f"""
                    CREATE VIRTUAL TABLE {_LOCAL_FTS_TABLE} USING fts5(
                        title,
                        body,
                        uri,
                        keywords,
                        content='local_records',
                        content_rowid='rowid',
                        tokenize='unicode61'
                    )
                    """
                )
            except sqlite3.OperationalError as exc:
                if "fts5" not in str(exc).lower():
                    raise
                self._fts5_available = False
                self._keyword_search_diagnostic = (
                    "FTS5 indexed lexical search is unavailable; "
                    "using the bounded SQLite scan fallback"
                )
            else:
                self._fts5_available = True
                needs_rebuild = True
        else:
            self._fts5_available = True

        version_row = conn.execute(
            "SELECT version FROM local_keyword_schema WHERE name = ?",
            (_LOCAL_KEYWORD_SCHEMA,),
        ).fetchone()
        if self._fts5_available and (
            needs_rebuild
            or version_row is None
            or version_row[0] != _LOCAL_KEYWORD_SCHEMA_VERSION
        ):
            conn.execute(
                f"INSERT INTO {_LOCAL_FTS_TABLE}({_LOCAL_FTS_TABLE}) VALUES ('rebuild')"
            )

        conn.execute(
            """
            INSERT INTO local_keyword_schema (name, version)
            VALUES (?, ?)
            ON CONFLICT(name) DO UPDATE SET version = excluded.version
            """,
            (_LOCAL_KEYWORD_SCHEMA, _LOCAL_KEYWORD_SCHEMA_VERSION),
        )
        if self._fts5_available:
            self._keyword_search_diagnostic = "FTS5 indexed lexical search is active"

    def _bump_epoch(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            INSERT INTO system_state (key, value) VALUES ('local_record_epoch', '1')
            ON CONFLICT(key) DO UPDATE SET value = CAST(value AS INTEGER) + 1
            """
        )

    @staticmethod
    def _record_values(record: Record) -> tuple[Any, ...]:
        return (
            record.storage_key,
            record.workspace_id,
            record.source_kind,
            record.source_id,
            record.title,
            record.body,
            record.created_at.isoformat(),
            record.updated_at.isoformat(),
            json.dumps(record.metadata, sort_keys=True),
            LocalRecordBackend._record_uri(record),
            LocalRecordBackend._metadata_keyword_text(record.metadata),
            record.status.value,
        )

    @staticmethod
    def _delete_fts_row(
        conn: sqlite3.Connection,
        row: sqlite3.Row,
    ) -> None:
        conn.execute(
            f"""
            INSERT INTO {_LOCAL_FTS_TABLE}
                ({_LOCAL_FTS_TABLE}, rowid, title, body, uri, keywords)
            VALUES ('delete', ?, ?, ?, ?, ?)
            """,
            (row["rowid"], row["title"], row["body"], row["uri"], row["keywords"]),
        )

    @staticmethod
    def _insert_fts_row(
        conn: sqlite3.Connection,
        row: sqlite3.Row,
    ) -> None:
        conn.execute(
            f"""
            INSERT INTO {_LOCAL_FTS_TABLE}
                (rowid, title, body, uri, keywords)
            VALUES (?, ?, ?, ?, ?)
            """,
            (row["rowid"], row["title"], row["body"], row["uri"], row["keywords"]),
        )

    def _upsert_records(self, records: Iterable[Record]) -> None:
        rows = list(records)
        if not rows:
            return
        with self._lock:
            conn = self._db.get_connection()
            try:
                for record in rows:
                    old_row = conn.execute(
                        """
                        SELECT rowid, title, body, uri, keywords
                        FROM local_records
                        WHERE storage_key = ?
                        """,
                        (record.storage_key,),
                    ).fetchone()
                    if old_row is not None and self._fts5_available:
                        self._delete_fts_row(conn, old_row)
                    conn.execute(
                        """
                        INSERT INTO local_records (
                            storage_key, workspace_id, source_kind, source_id, title, body,
                            created_at, updated_at, metadata, uri, keywords, status
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(storage_key) DO UPDATE SET
                            workspace_id = excluded.workspace_id,
                            source_kind = excluded.source_kind,
                            source_id = excluded.source_id,
                            title = excluded.title,
                            body = excluded.body,
                            created_at = excluded.created_at,
                            updated_at = excluded.updated_at,
                            metadata = excluded.metadata,
                            uri = excluded.uri,
                            keywords = excluded.keywords,
                            status = excluded.status
                        """,
                        self._record_values(record),
                    )
                    if self._fts5_available:
                        new_row = conn.execute(
                            """
                            SELECT rowid, title, body, uri, keywords
                            FROM local_records
                            WHERE storage_key = ?
                            """,
                            (record.storage_key,),
                        ).fetchone()
                        assert new_row is not None
                        self._insert_fts_row(conn, new_row)
                self._bump_epoch(conn)
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def index(self, records: list[Record]) -> None:
        """Index records for keyword retrieval."""
        self._upsert_records(records)

    def upsert(self, records: list[Record], model_name: str, dim: int) -> None:
        """Persist records and their optional model-specific embeddings."""
        if dim < 1:
            raise ValueError("dim must be positive")
        rows = list(records)
        for record in rows:
            if record.embedding is not None and len(record.embedding) != dim:
                raise ValueError(
                    f"embedding dimension mismatch for {record.storage_key}: "
                    f"expected {dim}, got {len(record.embedding)}"
                )
        self._upsert_records(rows)
        if not rows:
            return
        with self._lock:
            conn = self._db.get_connection()
            conn.executemany(
                """
                INSERT INTO local_vectors (storage_key, model_name, dim, embedding)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(storage_key, model_name, dim) DO UPDATE SET
                    embedding = excluded.embedding
                """,
                [
                    (
                        record.storage_key,
                        model_name,
                        dim,
                        json.dumps(record.embedding),
                    )
                    for record in rows
                    if record.embedding is not None
                ],
            )
            self._bump_epoch(conn)
            conn.commit()

    @staticmethod
    def _status_values(filters: dict[str, Any] | None) -> set[str]:
        if filters and filters.get("include_inactive"):
            return {"active", "stale", "archived"}
        values = filters.get("statuses") if filters else None
        return {str(value) for value in values} if values is not None else {"active"}

    @classmethod
    def _matches(
        cls,
        row: sqlite3.Row,
        filters: dict[str, Any] | None,
    ) -> bool:
        filters = filters or {}
        if row["status"] not in cls._status_values(filters):
            return False
        workspace_id = filters.get("workspace_id")
        if workspace_id is not None and row["workspace_id"] != workspace_id:
            return False
        source_kinds = filters.get("source_kinds")
        if source_kinds is not None and row["source_kind"] not in source_kinds:
            return False
        candidate_ids = filters.get("candidate_ids")
        return candidate_ids is None or row["storage_key"] in candidate_ids

    def _record_rows(self) -> list[sqlite3.Row]:
        conn = self._db.get_connection()
        conn.row_factory = sqlite3.Row
        return conn.execute("SELECT * FROM local_records").fetchall()

    def search_keyword(
        self,
        query: str,
        k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RecordHit]:
        if k < 1 or not query.strip():
            return []
        terms = [term.lower() for term in _TOKEN_RE.findall(query)]
        if not terms:
            return []
        hits: list[RecordHit] = []
        with self._lock:
            for row in self._record_rows():
                if not self._matches(row, filters):
                    continue
                haystack = f"{row['title']} {row['body']}".lower()
                score = sum(haystack.count(term) for term in terms)
                if score:
                    hits.append(
                        RecordHit(
                            RecordIdentity(
                                row["workspace_id"],
                                row["source_kind"],
                                row["source_id"],
                            ),
                            float(score),
                        )
                    )
        hits.sort(key=lambda item: (-item.score, item.storage_key))
        return hits[:k]

    def search_vector(
        self,
        query_vector: Vector,
        k: int,
        *,
        model_name: str,
        dim: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RecordHit]:
        if k < 1:
            return []
        if len(query_vector) != dim:
            raise ValueError(
                f"query vector dimension mismatch: expected {dim}, got {len(query_vector)}"
            )
        query = np.asarray(query_vector, dtype=np.float32)
        query_norm = float(np.linalg.norm(query))
        if query_norm == 0:
            return []
        with self._lock:
            conn = self._db.get_connection()
            rows = conn.execute(
                """
                SELECT r.*, v.embedding
                FROM local_records r
                JOIN local_vectors v ON v.storage_key = r.storage_key
                WHERE v.model_name = ? AND v.dim = ?
                """,
                (model_name, dim),
            ).fetchall()
            hits: list[RecordHit] = []
            for row in rows:
                if not self._matches(row, filters):
                    continue
                vector = np.asarray(json.loads(row["embedding"]), dtype=np.float32)
                norm = float(np.linalg.norm(vector))
                score = float(np.dot(query, vector) / (query_norm * norm)) if norm else 0.0
                hits.append(
                    RecordHit(
                        RecordIdentity(
                            row["workspace_id"],
                            row["source_kind"],
                            row["source_id"],
                        ),
                        score,
                    )
                )
        hits.sort(key=lambda item: (-item.score, item.storage_key))
        return hits[:k]

    def hydrate_record(
        self,
        record_id: RecordIdentity | str,
        *,
        source_kind: str | None = None,
        workspace_id: str | None = None,
    ) -> Record | None:
        storage_key = (
            record_id.storage_key if isinstance(record_id, RecordIdentity) else record_id
        )
        if isinstance(record_id, RecordIdentity):
            source_kind = record_id.source_kind
            workspace_id = record_id.workspace_id
        with self._lock:
            conn = self._db.get_connection()
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM local_records WHERE storage_key = ?",
                (storage_key,),
            ).fetchone()
            if row is None:
                candidates = conn.execute(
                    """
                    SELECT * FROM local_records
                    WHERE source_id = ?
                      AND (? IS NULL OR source_kind = ?)
                      AND (? IS NULL OR workspace_id = ?)
                    ORDER BY storage_key
                    """,
                    (
                        record_id.source_id if isinstance(record_id, RecordIdentity) else record_id,
                        source_kind,
                        source_kind,
                        workspace_id,
                        workspace_id,
                    ),
                ).fetchall()
                if len(candidates) != 1:
                    return None
                row = candidates[0]
            return Record(
                workspace_id=row["workspace_id"],
                source_kind=row["source_kind"],
                source_id=row["source_id"],
                title=row["title"],
                body=row["body"],
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
                metadata=json.loads(row["metadata"]),
                uri=row["uri"],
                status=RecordStatus(row["status"]),
            )

    def delete(self, record_ids: list[str]) -> None:
        if not record_ids:
            return
        with self._lock:
            conn = self._db.get_connection()
            for record_id in record_ids:
                record = self.hydrate_record(record_id)
                if record is None:
                    continue
                old_row = conn.execute(
                    """
                    SELECT rowid, title, body, uri, keywords
                    FROM local_records
                    WHERE storage_key = ?
                    """,
                    (record.storage_key,),
                ).fetchone()
                if old_row is not None and self._fts5_available:
                    self._delete_fts_row(conn, old_row)
                conn.execute(
                    "DELETE FROM local_vectors WHERE storage_key = ?",
                    (record.storage_key,),
                )
                conn.execute(
                    "DELETE FROM local_records WHERE storage_key = ?",
                    (record.storage_key,),
                )
            self._bump_epoch(conn)
            conn.commit()

    @property
    def keyword_index_available(self) -> bool:
        return self._fts5_available

    @property
    def keyword_search_diagnostic(self) -> str:
        return self._keyword_search_diagnostic

    def check_keyword_index(self) -> bool:
        """Return whether the external-content keyword index matches records."""
        if not self._fts5_available:
            return False
        with self._lock:
            conn = self._db.get_connection()
            try:
                conn.execute(
                    f"INSERT INTO {_LOCAL_FTS_TABLE}({_LOCAL_FTS_TABLE}) "
                    "VALUES ('integrity-check')"
                )
                missing = conn.execute(
                    f"""
                    SELECT EXISTS(
                        SELECT 1
                        FROM local_records r
                        LEFT JOIN {_LOCAL_FTS_TABLE} f ON f.rowid = r.rowid
                        WHERE f.rowid IS NULL
                    )
                    OR EXISTS(
                        SELECT 1
                        FROM {_LOCAL_FTS_TABLE} f
                        LEFT JOIN local_records r ON r.rowid = f.rowid
                        WHERE r.rowid IS NULL
                    )
                    """
                ).fetchone()[0]
            except sqlite3.DatabaseError:
                return False
            return not bool(missing)

    def rebuild_keyword_index(self) -> None:
        """Rebuild the external-content keyword index from local records."""
        with self._lock:
            conn = self._db.get_connection()
            if not self._fts5_available:
                self._keyword_search_diagnostic = (
                    "FTS5 indexed lexical search is unavailable; "
                    "the keyword index cannot be rebuilt"
                )
                return
            try:
                conn.execute(
                    f"INSERT INTO {_LOCAL_FTS_TABLE}({_LOCAL_FTS_TABLE}) VALUES ('rebuild')"
                )
                conn.commit()
            except sqlite3.DatabaseError:
                conn.rollback()
                conn.execute(f"DROP TABLE IF EXISTS {_LOCAL_FTS_TABLE}")
                conn.execute(
                    f"""
                    CREATE VIRTUAL TABLE {_LOCAL_FTS_TABLE} USING fts5(
                        title,
                        body,
                        uri,
                        keywords,
                        content='local_records',
                        content_rowid='rowid',
                        tokenize='unicode61'
                    )
                    """
                )
                conn.execute(
                    f"INSERT INTO {_LOCAL_FTS_TABLE}({_LOCAL_FTS_TABLE}) VALUES ('rebuild')"
                )
                conn.commit()

    def epoch(self) -> int:
        with self._lock:
            conn = self._db.get_connection()
            row = conn.execute(
                "SELECT value FROM system_state WHERE key = 'local_record_epoch'"
            ).fetchone()
            return int(row[0]) if row else 0

    def upsert_edges(
        self,
        edges: Sequence[GraphEdge | tuple[str, str, str, float]],
    ) -> None:
        rows = [
            (
                edge.source.storage_key,
                edge.target.storage_key,
                edge.edge_type,
                edge.weight,
            )
            if isinstance(edge, GraphEdge)
            else edge
            for edge in edges
        ]
        with self._lock:
            conn = self._db.get_connection()
            conn.executemany(
                """
                INSERT INTO local_graph_edges (source_id, target_id, edge_type, weight)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(source_id, target_id, edge_type) DO UPDATE SET
                    weight = excluded.weight
                """,
                rows,
            )
            conn.commit()

    def neighbors(
        self,
        record_id: RecordIdentity | str,
        edge_types: list[str] | None = None,
        depth: int = 1,
    ) -> list[GraphNeighbor]:
        if depth < 1:
            raise ValueError("depth must be positive")
        allowed = set(edge_types) if edge_types else None
        identity_key = (
            record_id.storage_key if isinstance(record_id, RecordIdentity) else record_id
        )
        frontier = {identity_key}
        best: dict[str, tuple[str, float]] = {}
        with self._lock:
            conn = self._db.get_connection()
            for hop in range(depth):
                if not frontier:
                    break
                placeholders = ",".join("?" for _ in frontier)
                rows = conn.execute(
                    f"""
                    SELECT source_id, target_id, edge_type, weight
                    FROM local_graph_edges
                    WHERE source_id IN ({placeholders})
                    """,
                    tuple(frontier),
                ).fetchall()
                next_frontier: set[str] = set()
                for row in rows:
                    if allowed is not None and row["edge_type"] not in allowed:
                        continue
                    if row["target_id"] == identity_key:
                        continue
                    weight = float(row["weight"])
                    if hop:
                        parent_weight = best.get(row["source_id"], ("", 1.0))[1]
                        weight *= parent_weight
                    current = best.get(row["target_id"])
                    if current is None or weight > current[1]:
                        best[row["target_id"]] = (row["edge_type"], weight)
                    next_frontier.add(row["target_id"])
                frontier = next_frontier
        return sorted(
            (
                GraphNeighbor(
                    self._identity_from_storage_key(target_id),
                    edge_type,
                    weight,
                )
                for target_id, (edge_type, weight) in best.items()
            ),
            key=lambda item: (-item.weight, item.identity.storage_key, item.edge_type),
        )

    @staticmethod
    def _identity_from_storage_key(storage_key: str) -> RecordIdentity:
        if storage_key.startswith("record:"):
            try:
                workspace_id, source_kind, source_id = json.loads(storage_key[7:])
            except (TypeError, ValueError):
                pass
            else:
                if (
                    (workspace_id is None or isinstance(workspace_id, str))
                    and isinstance(source_kind, str)
                    and isinstance(source_id, str)
                ):
                    return RecordIdentity(workspace_id, source_kind, source_id)
        return RecordIdentity(None, "legacy", storage_key)

class LocalVectorStore:
    """VectorStore implementation backed by the local record database."""

    def __init__(self, backend: LocalRecordBackend):
        self._backend = backend

    def upsert(self, records: list[Record], model_name: str, dim: int) -> None:
        self._backend.upsert(records, model_name, dim)

    def search(
        self,
        query_vector: Vector,
        k: int,
        *,
        model_name: str,
        dim: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RecordHit]:
        return self._backend.search_vector(
            query_vector, k, model_name=model_name, dim=dim, filters=filters
        )

    def delete(self, record_ids: list[str]) -> None:
        self._backend.delete(record_ids)

    def epoch(self) -> int:
        return self._backend.epoch()


class LocalKeywordStore:
    """KeywordStore implementation backed by SQLite record rows."""

    def __init__(self, backend: LocalRecordBackend):
        self._backend = backend

    def index(self, records: list[Record]) -> None:
        self._backend.index(records)

    def search(
        self, query: str, k: int, filters: dict[str, Any] | None = None
    ) -> list[RecordHit]:
        return self._backend.search_keyword(query, k, filters)


class LocalGraphStore:
    """GraphStore implementation backed by SQLite graph rows."""

    def __init__(self, backend: LocalRecordBackend):
        self._backend = backend

    def upsert_edges(
        self,
        edges: Sequence[GraphEdge | tuple[str, str, str, float]],
    ) -> None:
        self._backend.upsert_edges(edges)

    def neighbors(
        self,
        record_id: RecordIdentity | str,
        edge_types: list[str] | None = None,
        depth: int = 1,
    ) -> list[GraphNeighbor]:
        return self._backend.neighbors(record_id, edge_types, depth)


FAISSVectorStore = LocalVectorStore
SQLiteKeywordStore = LocalKeywordStore
SQLiteGraphStore = LocalGraphStore

__all__ = [
    "FAISSVectorStore",
    "LocalGraphStore",
    "LocalKeywordStore",
    "LocalRecordBackend",
    "LocalVectorStore",
    "SQLiteGraphStore",
    "SQLiteKeywordStore",
]
