"""SQLite-backed record stores for the local search backend.

The local backend deliberately speaks the same record-oriented ports as the
pgvector adapter.  Chunk-oriented FAISS/SQLite indices remain available for
the ingestion surface, but query execution uses canonical record identities.
"""

from __future__ import annotations

import asyncio
import json
import math
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
from searchkernel.domain.vector_filters import (
    metadata_mapping,
    record_matches_vector_filters,
)
from searchkernel.indices import keyword_scoring as _keyword_scoring
from searchkernel.indices.faiss_local import FAISSLocalVectorStore
from searchkernel.indices.local_vectors import (
    NORMALIZATION_POLICY,
    VECTOR_FORMAT_VERSION,
    PackedVectorCodec,
    VectorSnapshot,
)
from searchkernel.storage.db import DatabaseManager, SQLiteTuning

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_LOCAL_KEYWORD_SCHEMA = "local_records_fts"
_LOCAL_KEYWORD_SCHEMA_VERSION = 1
_LOCAL_FTS_TABLE = "local_records_fts"
_LOCAL_FTS_COLUMNS = ("title", "body", "uri", "keywords")
_FALLBACK_SCAN_MAX_ROWS = 10_000


class _EphemeralDatabase:
    def __init__(self, tuning: SQLiteTuning | None = None) -> None:
        self._connection = sqlite3.connect(":memory:", check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        (tuning or SQLiteTuning()).apply(self._connection)
        self._connection.execute("PRAGMA foreign_keys=ON")

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
        sqlite_tuning: SQLiteTuning | None = None,
        keyword_overfetch_multiplier: float = 4.0,
        vector_engine: str = "exact",
        faiss_threshold: int = 50_000,
        vector_snapshot_max_rows: int = 100_000,
    ) -> None:
        if db_manager is not None and db_path is not None:
            raise ValueError("pass db_path or db_manager, not both")
        if db_manager is not None and sqlite_tuning is not None:
            raise ValueError("sqlite_tuning belongs to a new database manager")
        if not math.isfinite(keyword_overfetch_multiplier) or keyword_overfetch_multiplier < 1.0:
            raise ValueError("keyword_overfetch_multiplier must be finite and at least 1")
        if vector_engine not in {"exact", "faiss", "auto"}:
            raise ValueError("vector_engine must be exact, faiss, or auto")
        if faiss_threshold < 1:
            raise ValueError("faiss_threshold must be positive")
        if vector_snapshot_max_rows < 1:
            raise ValueError("vector_snapshot_max_rows must be positive")
        self._db = db_manager or (
            DatabaseManager(db_path, tuning=sqlite_tuning)
            if db_path is not None
            else _EphemeralDatabase(sqlite_tuning)
        )
        self._lock = threading.RLock()
        self._snapshot_lock = threading.RLock()
        self._vector_snapshots: dict[tuple[str, int], VectorSnapshot] = {}
        self._keyword_overfetch_multiplier = keyword_overfetch_multiplier
        self._vector_engine = vector_engine
        self._faiss_threshold = faiss_threshold
        self._vector_snapshot_max_rows = vector_snapshot_max_rows
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
        conn.row_factory = sqlite3.Row
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
            CREATE TABLE IF NOT EXISTS local_vectors_v2 (
                storage_key TEXT NOT NULL,
                encoder_namespace TEXT NOT NULL,
                dim INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                format_version INTEGER NOT NULL,
                normalization_policy TEXT NOT NULL,
                PRIMARY KEY (storage_key, encoder_namespace, dim),
                FOREIGN KEY (storage_key) REFERENCES local_records(storage_key)
                    ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_local_vectors_v2_namespace
                ON local_vectors_v2 (encoder_namespace, dim);
            CREATE TABLE IF NOT EXISTS local_vector_schema (
                name TEXT PRIMARY KEY,
                version INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS local_graph_edges (
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                edge_type TEXT NOT NULL,
                weight REAL NOT NULL,
                PRIMARY KEY (source_id, target_id, edge_type),
                FOREIGN KEY (source_id) REFERENCES local_records(storage_key)
                    ON DELETE CASCADE,
                FOREIGN KEY (target_id) REFERENCES local_records(storage_key)
                    ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS system_state (
                key TEXT PRIMARY KEY,
                value TEXT
            );
            """
        )
        self._ensure_local_record_column(conn, "keywords", "TEXT NOT NULL DEFAULT ''")
        self._initialize_keyword_schema(conn)
        self._migrate_legacy_vectors(conn)
        self._initialize_graph_schema(conn)
        conn.commit()

    @staticmethod
    def _canonical_graph_storage_key(storage_key: str) -> str:
        if not isinstance(storage_key, str):
            raise TypeError("graph endpoints must be canonical storage keys")
        try:
            identity = RecordIdentity.from_storage_key(storage_key)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid graph storage key: {storage_key!r}"
            ) from exc
        if not identity.source_kind or not identity.source_id:
            raise ValueError(f"invalid graph storage key: {storage_key!r}")
        if identity.storage_key != storage_key:
            raise ValueError(f"non-canonical graph storage key: {storage_key!r}")
        return storage_key

    @classmethod
    def _initialize_graph_schema(cls, conn: sqlite3.Connection) -> None:
        foreign_keys = {
            row[3]
            for row in conn.execute("PRAGMA foreign_key_list(local_graph_edges)")
        }
        if foreign_keys != {"source_id", "target_id"}:
            conn.execute("DROP INDEX IF EXISTS idx_local_graph_source")
            conn.execute("DROP INDEX IF EXISTS idx_local_graph_target")
            conn.execute("DROP INDEX IF EXISTS idx_local_graph_source_type")
            conn.execute(
                "ALTER TABLE local_graph_edges RENAME TO local_graph_edges_legacy"
            )
            conn.execute(
                """
                CREATE TABLE local_graph_edges (
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    edge_type TEXT NOT NULL,
                    weight REAL NOT NULL,
                    PRIMARY KEY (source_id, target_id, edge_type),
                    FOREIGN KEY (source_id) REFERENCES local_records(storage_key)
                        ON DELETE CASCADE,
                    FOREIGN KEY (target_id) REFERENCES local_records(storage_key)
                        ON DELETE CASCADE
                )
                """
            )
            record_keys = {
                row[0]
                for row in conn.execute(
                    "SELECT storage_key FROM local_records"
                ).fetchall()
            }
            legacy_rows = conn.execute(
                """
                SELECT e.source_id, e.target_id, e.edge_type, e.weight
                FROM local_graph_edges_legacy
                """
            ).fetchall()
            for row in legacy_rows:
                try:
                    source_id = cls._canonical_graph_storage_key(row[0])
                    target_id = cls._canonical_graph_storage_key(row[1])
                except ValueError:
                    continue
                if source_id not in record_keys or target_id not in record_keys:
                    continue
                conn.execute(
                    """
                    INSERT OR IGNORE INTO local_graph_edges
                        (source_id, target_id, edge_type, weight)
                    VALUES (?, ?, ?, ?)
                    """,
                    (source_id, target_id, row[2], row[3]),
                )
            conn.execute("DROP TABLE local_graph_edges_legacy")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_local_graph_source "
            "ON local_graph_edges (source_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_local_graph_target "
            "ON local_graph_edges (target_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_local_graph_source_type "
            "ON local_graph_edges (source_id, edge_type)"
        )

    @staticmethod
    def _migrate_legacy_vectors(conn: sqlite3.Connection) -> None:
        columns = {
            row[1] for row in conn.execute("PRAGMA table_info(local_vectors)")
        }
        if not columns:
            return
        rows = conn.execute(
            "SELECT storage_key, model_name, dim, embedding FROM local_vectors"
        ).fetchall()
        if not rows:
            conn.execute(
                """
                INSERT INTO local_vector_schema (name, version)
                VALUES ('local_vectors', ?)
                ON CONFLICT(name) DO UPDATE SET version = excluded.version
                """,
                (VECTOR_FORMAT_VERSION,),
            )
            return
        for row in rows:
            packed = PackedVectorCodec.migrate_json(
                row["embedding"],
                int(row["dim"]),
                context=(
                    f"legacy embedding for {row['storage_key']} "
                    f"namespace {row['model_name']!r}"
                ),
            )
            conn.execute(
                """
                INSERT INTO local_vectors_v2 (
                    storage_key, encoder_namespace, dim, embedding,
                    format_version, normalization_policy
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(storage_key, encoder_namespace, dim) DO UPDATE SET
                    embedding = excluded.embedding,
                    format_version = excluded.format_version,
                    normalization_policy = excluded.normalization_policy
                """,
                (
                    row["storage_key"],
                    row["model_name"],
                    int(row["dim"]),
                    packed,
                    VECTOR_FORMAT_VERSION,
                    NORMALIZATION_POLICY,
                ),
            )
        conn.execute("DELETE FROM local_vectors")
        conn.execute(
            """
            INSERT INTO local_vector_schema (name, version)
            VALUES ('local_vectors', ?)
            ON CONFLICT(name) DO UPDATE SET version = excluded.version
            """,
            (VECTOR_FORMAT_VERSION,),
        )

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
            if isinstance(value, set):
                values.extend(str(item) for item in sorted(value, key=str))
            elif isinstance(value, (list, tuple)):
                values.extend(str(item) for item in value)
            else:
                values.append(str(value))
        return " ".join(" ".join(value.strip().lower().split()) for value in values if value)

    @staticmethod
    def _metadata_uri(metadata: dict[str, Any]) -> str:
        for key in ("uri", "source_file", "file_path", "path"):
            value = metadata.get(key)
            if value:
                return str(value)
        return ""

    @classmethod
    def _record_uri(cls, record: Record) -> str:
        return record.uri or cls._metadata_uri(record.metadata)

    @classmethod
    def _migrate_keyword_columns(cls, conn: sqlite3.Connection) -> None:
        rows = conn.execute(
            "SELECT rowid, metadata, uri, keywords FROM local_records"
        ).fetchall()
        for row in rows:
            try:
                metadata = json.loads(row["metadata"])
            except (TypeError, ValueError):
                metadata = {}
            if not isinstance(metadata, dict):
                metadata = {}
            keywords = cls._metadata_keyword_text(metadata)
            uri = row["uri"] or cls._metadata_uri(metadata)
            if row["keywords"] != keywords or row["uri"] != uri:
                conn.execute(
                    "UPDATE local_records SET uri = ?, keywords = ? WHERE rowid = ?",
                    (uri, keywords, row["rowid"]),
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

    @staticmethod
    def _bump_epoch(
        conn: sqlite3.Connection,
        *,
        keyword: bool = False,
        vector: bool = False,
        graph: bool = False,
    ) -> None:
        lanes = {
            "local_record_epoch": True,
            "local_keyword_epoch": keyword,
            "local_vector_epoch": vector,
            "local_graph_epoch": graph,
        }
        for key, enabled in lanes.items():
            if not enabled:
                continue
            conn.execute(
                """
                INSERT INTO system_state (key, value) VALUES (?, '1')
                ON CONFLICT(key) DO UPDATE SET value = CAST(value AS INTEGER) + 1
                """,
                (key,),
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

    def _write_records(
        self,
        conn: sqlite3.Connection,
        rows: list[Record],
    ) -> None:
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

    def _upsert_records(self, records: Iterable[Record]) -> None:
        rows = list(records)
        if not rows:
            return
        with self._lock:
            conn = self._db.get_connection()
            try:
                self._write_records(conn, rows)
                self._bump_epoch(conn, keyword=True)
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
        packed_vectors: list[tuple[str, bytes]] = []
        for record in rows:
            if record.embedding is not None:
                packed_vectors.append(
                    (
                        record.storage_key,
                        PackedVectorCodec.encode(
                            record.embedding,
                            dim,
                            context=f"embedding for {record.storage_key}",
                        ),
                    )
                )
        if not rows:
            return
        with self._lock:
            conn = self._db.get_connection()
            try:
                self._write_records(conn, rows)
                conn.executemany(
                    """
                    INSERT INTO local_vectors_v2 (
                        storage_key, encoder_namespace, dim, embedding,
                        format_version, normalization_policy
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(storage_key, encoder_namespace, dim) DO UPDATE SET
                        embedding = excluded.embedding,
                        format_version = excluded.format_version,
                        normalization_policy = excluded.normalization_policy
                    """,
                    [
                        (
                            storage_key,
                            model_name,
                            dim,
                            embedding,
                            VECTOR_FORMAT_VERSION,
                            NORMALIZATION_POLICY,
                        )
                        for storage_key, embedding in packed_vectors
                    ],
                )
                self._bump_epoch(
                    conn,
                    keyword=True,
                    vector=bool(packed_vectors),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    @staticmethod
    def _status_values(filters: dict[str, Any] | None) -> set[str]:
        if filters and filters.get("include_inactive"):
            return {"active", "stale", "archived"}
        if filters and filters.get("status") is not None:
            value = filters["status"]
            return {value.value if isinstance(value, RecordStatus) else str(value)}
        values = filters.get("statuses") if filters else None
        if values is None:
            return {"active"}
        return {
            value.value if isinstance(value, RecordStatus) else str(value)
            for value in LocalRecordBackend._filter_values(values)
        }

    @staticmethod
    def _filter_values(value: Any) -> list[Any]:
        if isinstance(value, (str, RecordStatus)):
            return [value]
        return list(value)

    @classmethod
    def _matches(
        cls,
        row: sqlite3.Row,
        filters: dict[str, Any] | None,
    ) -> bool:
        return record_matches_vector_filters(
            storage_key=row["storage_key"],
            source_id=row["source_id"],
            workspace_id=row["workspace_id"],
            source_kind=row["source_kind"],
            status=row["status"],
            metadata=metadata_mapping(row["metadata"]),
            uri=row["uri"],
            filters=filters,
        )

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
        if self._fts5_available:
            return self._search_keyword_fts(query, k, filters)
        return self._search_keyword_fallback(query, k, filters)

    @staticmethod
    def _keyword_filter_sql(
        filters: dict[str, Any] | None,
    ) -> tuple[list[str], list[Any]]:
        filters = filters or {}
        statuses = sorted(LocalRecordBackend._status_values(filters))
        clauses = ["r.status IN ({})".format(
            ", ".join("?" for _ in statuses)
        )]
        parameters: list[Any] = statuses

        workspace_id = filters.get("workspace_id")
        if workspace_id is not None:
            clauses.append("r.workspace_id = ?")
            parameters.append(workspace_id)

        source_kinds = filters.get("source_kinds")
        if source_kinds is None and filters.get("source_kind") is not None:
            source_kinds = [filters["source_kind"]]
        if source_kinds is not None:
            source_kinds = LocalRecordBackend._filter_values(source_kinds)
            if not source_kinds:
                return ["0"], []
            clauses.append("r.source_kind IN ({})".format(
                ", ".join("?" for _ in source_kinds)
            ))
            parameters.extend(source_kinds)

        candidate_keys = filters.get("candidate_ids")
        if candidate_keys is None:
            candidate_keys = filters.get("candidate_storage_keys")
        if candidate_keys is not None:
            candidate_keys = LocalRecordBackend._filter_values(candidate_keys)
            if not candidate_keys:
                return ["0"], []
            clauses.append("r.storage_key IN ({})".format(
                ", ".join("?" for _ in candidate_keys)
            ))
            parameters.extend(candidate_keys)
        return clauses, parameters

    def _search_keyword_fts(
        self,
        query: str,
        k: int,
        filters: dict[str, Any] | None,
    ) -> list[RecordHit]:
        match_query = _keyword_scoring.sanitize_fts_query(query)
        if match_query == '""':
            return []
        if _keyword_scoring.looks_like_artifact_query(query):
            match_query = '"' + match_query.replace('"', "") + '"'
        clauses, parameters = self._keyword_filter_sql(filters)
        clauses.insert(0, f"{_LOCAL_FTS_TABLE} MATCH ?")
        parameters.insert(0, match_query)
        needs_artifact_rerank = _keyword_scoring.looks_like_artifact_query(query)
        limit = k
        if needs_artifact_rerank:
            limit = max(k, math.ceil(k * self._keyword_overfetch_multiplier))
        with self._lock:
            conn = self._db.get_connection()
            rows = conn.execute(
                f"""
                SELECT
                    r.storage_key,
                    r.workspace_id,
                    r.source_kind,
                    r.source_id,
                    r.title,
                    r.body,
                    r.uri,
                    r.keywords,
                    -bm25({_LOCAL_FTS_TABLE}, 5.0, 1.0, 4.0, 2.0) AS score
                FROM {_LOCAL_FTS_TABLE}
                JOIN local_records r ON r.rowid = {_LOCAL_FTS_TABLE}.rowid
                WHERE {" AND ".join(clauses)}
                ORDER BY score DESC, r.storage_key ASC
                LIMIT ?
                """,
                (*parameters, limit),
            ).fetchall()
        hits: list[RecordHit] = []
        for row in rows:
            score = float(row["score"])
            if needs_artifact_rerank:
                normalized_query = _keyword_scoring.normalize_artifact_value(query)
                score += _keyword_scoring.score_field_aware_match(
                    query,
                    content=row["body"],
                    title=row["title"],
                    headers=row["keywords"],
                    source_file=row["uri"] or "",
                )
                score += _keyword_scoring.score_artifact_match(
                    normalized_query,
                    Path(normalized_query).name,
                    row["body"],
                    row["title"],
                    row["keywords"],
                    row["uri"] or "",
                )
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

    def _search_keyword_fallback(
        self,
        query: str,
        k: int,
        filters: dict[str, Any] | None,
    ) -> list[RecordHit]:
        terms = [term.lower() for term in _TOKEN_RE.findall(query)]
        if not terms:
            return []
        clauses, parameters = self._keyword_filter_sql(filters)
        with self._lock:
            conn = self._db.get_connection()
            rows = conn.execute(
                f"""
                SELECT storage_key, workspace_id, source_kind, source_id, title, body
                FROM local_records r
                WHERE {" AND ".join(clause.replace("r.", "") for clause in clauses)}
                LIMIT ?
                """,
                (*parameters, _FALLBACK_SCAN_MAX_ROWS + 1),
            ).fetchall()
        if len(rows) > _FALLBACK_SCAN_MAX_ROWS:
            self._keyword_search_diagnostic = (
                "FTS5 indexed lexical search is unavailable; "
                f"scan fallback refuses corpora over {_FALLBACK_SCAN_MAX_ROWS} rows"
            )
            return []
        hits: list[RecordHit] = []
        for row in rows:
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
        query = PackedVectorCodec.normalize(
            query_vector, dim, context="query vector"
        )
        if self.vector_count(model_name, dim) > self._vector_snapshot_max_rows:
            return self._search_vector_blocks(
                query,
                k,
                model_name=model_name,
                dim=dim,
                filters=filters,
            )
        snapshot = self._get_vector_snapshot(model_name, dim)
        eligible = snapshot.filter_mask(
            filters,
            status_values=self._status_values(filters),
            filter_values=self._filter_values,
        )
        positions = np.flatnonzero(eligible)
        if not len(positions):
            return []
        scores = snapshot.matrix[positions] @ query
        selected = self._select_top_positions(
            positions,
            scores,
            snapshot.storage_keys,
            k,
        )
        return [
            RecordHit(
                RecordIdentity.from_storage_key(snapshot.storage_keys[position]),
                float(scores[index]),
            )
            for index, position in selected
        ]

    def _search_vector_blocks(
        self,
        query: np.ndarray,
        k: int,
        *,
        model_name: str,
        dim: int,
        filters: dict[str, Any] | None,
    ) -> list[RecordHit]:
        best_keys: list[str] = []
        best_scores: list[float] = []
        offset = 0
        while True:
            with self._lock:
                conn = self._db.get_connection()
                rows = conn.execute(
                    """
                    SELECT r.storage_key, r.workspace_id, r.source_kind, r.source_id,
                           r.status, r.metadata, r.uri, v.embedding, v.format_version,
                           v.normalization_policy
                    FROM local_records r
                    JOIN local_vectors_v2 v ON v.storage_key = r.storage_key
                    WHERE v.encoder_namespace = ? AND v.dim = ?
                    ORDER BY r.storage_key
                    LIMIT ? OFFSET ?
                    """,
                    (model_name, dim, self._vector_snapshot_max_rows, offset),
                ).fetchall()
            if not rows:
                break
            eligible_rows = [row for row in rows if self._matches(row, filters)]
            if eligible_rows:
                matrix = np.vstack(
                    [
                        PackedVectorCodec.decode(
                            row["embedding"],
                            dim,
                            context=f"stored embedding for {row['storage_key']}",
                        )
                        for row in eligible_rows
                    ]
                )
                scores = matrix @ query
                best_keys.extend(row["storage_key"] for row in eligible_rows)
                best_scores.extend(float(score) for score in scores)
                if len(best_keys) > k:
                    positions = np.arange(len(best_keys))
                    selected = self._select_top_positions(
                        positions,
                        np.asarray(best_scores),
                        tuple(best_keys),
                        k,
                    )
                    best_keys = [best_keys[position] for _, position in selected]
                    best_scores = [best_scores[position] for _, position in selected]
            offset += len(rows)
        ordered = sorted(
            zip(best_keys, best_scores, strict=True),
            key=lambda item: (-item[1], item[0]),
        )
        return [
            RecordHit(RecordIdentity.from_storage_key(storage_key), score)
            for storage_key, score in ordered[:k]
        ]

    @property
    def vector_engine(self) -> str:
        return self._vector_engine

    @property
    def faiss_threshold(self) -> int:
        return self._faiss_threshold

    def vector_count(self, model_name: str, dim: int) -> int:
        with self._lock:
            conn = self._db.get_connection()
            row = conn.execute(
                """
                SELECT COUNT(*)
                FROM local_vectors_v2
                WHERE encoder_namespace = ? AND dim = ?
                """,
                (model_name, dim),
            ).fetchone()
            return int(row[0]) if row else 0

    def _get_vector_snapshot(self, model_name: str, dim: int) -> VectorSnapshot:
        key = (model_name, dim)
        with self._snapshot_lock:
            with self._lock:
                conn = self._db.get_connection()
                current_epoch = self._epoch_locked(conn)
                cached = self._vector_snapshots.get(key)
                if cached is not None and cached.epoch == current_epoch:
                    return cached
                rows = conn.execute(
                    """
                    SELECT r.storage_key, r.workspace_id, r.source_kind, r.source_id,
                           r.status, r.metadata, r.uri, v.embedding, v.format_version,
                           v.normalization_policy
                    FROM local_records r
                    JOIN local_vectors_v2 v ON v.storage_key = r.storage_key
                    WHERE v.encoder_namespace = ? AND v.dim = ?
                    ORDER BY r.storage_key
                    """,
                    (model_name, dim),
                ).fetchall()
                snapshot_epoch = self._epoch_locked(conn)
            snapshot = VectorSnapshot.from_rows(
                rows,
                encoder_namespace=model_name,
                dim=dim,
                epoch=snapshot_epoch,
            )
            self._vector_snapshots[key] = snapshot
            return snapshot

    @staticmethod
    def _select_top_positions(
        positions: np.ndarray,
        scores: np.ndarray,
        storage_keys: tuple[str, ...],
        k: int,
    ) -> list[tuple[int, int]]:
        if len(positions) <= k:
            candidates = np.arange(len(positions))
        else:
            partition = np.argpartition(-scores, k - 1)[:k]
            threshold = float(np.min(scores[partition]))
            above = np.flatnonzero(scores > threshold)
            ties = np.flatnonzero(scores == threshold)
            ties = ties[
                np.argsort(
                    np.asarray(
                        [storage_keys[int(positions[index])] for index in ties],
                        dtype=str,
                    ),
                    kind="stable",
                )
            ]
            candidates = np.concatenate((above, ties[: max(0, k - len(above))]))
        ordered = sorted(
            (int(index) for index in candidates),
            key=lambda index: (
                -float(scores[index]),
                storage_keys[int(positions[index])],
            ),
        )
        return [(index, int(positions[index])) for index in ordered[:k]]

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

    def hydrate_records(
        self,
        identities: Sequence[RecordIdentity],
    ) -> dict[str, Record | None]:
        """Hydrate canonical identities with one record query."""
        keys = list(dict.fromkeys(identity.storage_key for identity in identities))
        if not keys:
            return {}
        placeholders = ",".join("?" for _ in keys)
        with self._lock:
            conn = self._db.get_connection()
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"SELECT * FROM local_records WHERE storage_key IN ({placeholders})",
                keys,
            ).fetchall()
        records = {
            row["storage_key"]: Record(
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
            for row in rows
        }
        return {key: records.get(key) for key in keys}

    def delete(self, record_ids: list[str]) -> None:
        if not record_ids:
            return
        record_ids = list(dict.fromkeys(record_ids))
        with self._lock:
            conn = self._db.get_connection()
            try:
                existing_records = 0
                existing_vectors = 0
                deleted_graph_edges = 0
                for record_id in record_ids:
                    record = self.hydrate_record(record_id)
                    if record is None:
                        continue
                    existing_records += 1
                    existing_vectors += int(
                        conn.execute(
                            """
                            SELECT EXISTS(
                                SELECT 1 FROM local_vectors_v2
                                WHERE storage_key = ?
                            ) OR EXISTS(
                                SELECT 1 FROM local_vectors
                                WHERE storage_key = ?
                            )
                            """,
                            (record.storage_key, record.storage_key),
                        ).fetchone()[0]
                    )
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
                        "DELETE FROM local_vectors_v2 WHERE storage_key = ?",
                        (record.storage_key,),
                    )
                    conn.execute(
                        "DELETE FROM local_vectors WHERE storage_key = ?",
                        (record.storage_key,),
                    )
                    conn.execute(
                        "DELETE FROM local_records WHERE storage_key = ?",
                        (record.storage_key,),
                    )
                    deleted_graph_edges += conn.execute(
                        "DELETE FROM local_graph_edges "
                        "WHERE source_id = ? OR target_id = ?",
                        (record.storage_key, record.storage_key),
                    ).rowcount
                self._bump_epoch(
                    conn,
                    keyword=existing_records > 0,
                    vector=existing_vectors > 0,
                    graph=deleted_graph_edges > 0,
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise

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
                conn.execute("DROP TABLE IF EXISTS local_records_fts_vocab")
                conn.execute(
                    """
                    CREATE VIRTUAL TABLE local_records_fts_vocab
                    USING fts5vocab('local_records_fts', 'instance')
                    """
                )
                missing = conn.execute(
                    """
                    SELECT EXISTS(
                        SELECT 1
                        FROM local_records r
                        WHERE (
                            r.title != ''
                            OR r.body != ''
                            OR COALESCE(r.uri, '') != ''
                            OR r.keywords != ''
                        )
                        AND NOT EXISTS(
                            SELECT 1
                            FROM local_records_fts_vocab v
                            WHERE CAST(v.doc AS INTEGER) = r.rowid
                        )
                    )
                    OR EXISTS(
                        SELECT 1
                        FROM local_records_fts_vocab v
                        WHERE NOT EXISTS(
                            SELECT 1
                            FROM local_records r
                            WHERE r.rowid = CAST(v.doc AS INTEGER)
                        )
                    )
                    """
                ).fetchone()[0]
            except sqlite3.DatabaseError:
                return False
            finally:
                conn.execute("DROP TABLE IF EXISTS local_records_fts_vocab")
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
            return self._epoch_locked(conn)

    def keyword_epoch(self) -> int:
        return self._lane_epoch("local_keyword_epoch")

    def vector_epoch(self) -> int:
        return self._lane_epoch("local_vector_epoch")

    def graph_epoch(self) -> int:
        return self._lane_epoch("local_graph_epoch")

    def epochs(self) -> dict[str, int]:
        with self._lock:
            conn = self._db.get_connection()
            return {
                "keyword": self._lane_epoch_locked(conn, "local_keyword_epoch"),
                "vector": self._lane_epoch_locked(conn, "local_vector_epoch"),
                "graph": self._lane_epoch_locked(conn, "local_graph_epoch"),
            }

    @staticmethod
    def _epoch_locked(conn: sqlite3.Connection) -> int:
        row = conn.execute(
            "SELECT value FROM system_state WHERE key = 'local_record_epoch'"
        ).fetchone()
        return int(row[0]) if row else 0

    def _lane_epoch(self, key: str) -> int:
        with self._lock:
            conn = self._db.get_connection()
            return self._lane_epoch_locked(conn, key)

    @staticmethod
    def _lane_epoch_locked(conn: sqlite3.Connection, key: str) -> int:
        row = conn.execute(
            "SELECT value FROM system_state WHERE key = ?",
            (key,),
        ).fetchone()
        return int(row[0]) if row else 0

    def upsert_edges(
        self,
        edges: Sequence[GraphEdge | tuple[str, str, str, float]],
    ) -> None:
        rows: list[tuple[str, str, str, float]] = []
        for edge in edges:
            if isinstance(edge, GraphEdge):
                row = (
                    edge.source.storage_key,
                    edge.target.storage_key,
                    edge.edge_type,
                    edge.weight,
                )
            else:
                if len(edge) != 4:
                    raise ValueError("graph edges require four values")
                row = (edge[0], edge[1], edge[2], float(edge[3]))
            source_id = self._canonical_graph_storage_key(row[0])
            target_id = self._canonical_graph_storage_key(row[1])
            if not row[2] or not math.isfinite(row[3]):
                raise ValueError("graph edges require a finite weight and edge type")
            rows.append((source_id, target_id, row[2], row[3]))
        if not rows:
            return
        with self._lock:
            conn = self._db.get_connection()
            try:
                endpoint_keys = sorted(
                    {source_id for source_id, _, _, _ in rows}
                    | {target_id for _, target_id, _, _ in rows}
                )
                placeholders = ",".join("?" for _ in endpoint_keys)
                existing_keys = {
                    row[0]
                    for row in conn.execute(
                        f"""
                        SELECT storage_key
                        FROM local_records
                        WHERE storage_key IN ({placeholders})
                        """,
                        endpoint_keys,
                    ).fetchall()
                }
                missing_keys = set(endpoint_keys) - existing_keys
                if missing_keys:
                    raise ValueError(
                        "graph edge endpoints are not indexed: "
                        + ", ".join(sorted(missing_keys))
                    )
                changes_before = conn.total_changes
                conn.executemany(
                    """
                    INSERT INTO local_graph_edges (source_id, target_id, edge_type, weight)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(source_id, target_id, edge_type) DO UPDATE SET
                        weight = excluded.weight
                    WHERE local_graph_edges.weight IS NOT excluded.weight
                    """,
                    rows,
                )
                if conn.total_changes > changes_before:
                    self._bump_epoch(conn, graph=True)
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def delete_edges(
        self,
        edges: Sequence[GraphEdge | tuple[str, str, str, float]],
    ) -> None:
        rows: list[tuple[str, str, str]] = []
        for edge in edges:
            if isinstance(edge, GraphEdge):
                row = (
                    edge.source.storage_key,
                    edge.target.storage_key,
                    edge.edge_type,
                )
            else:
                if len(edge) < 3:
                    raise ValueError("graph edges require at least three values")
                row = (edge[0], edge[1], edge[2])
            rows.append(
                (
                    self._canonical_graph_storage_key(row[0]),
                    self._canonical_graph_storage_key(row[1]),
                    row[2],
                )
            )
        if not rows:
            return
        with self._lock:
            conn = self._db.get_connection()
            try:
                changes_before = conn.total_changes
                conn.executemany(
                    """
                    DELETE FROM local_graph_edges
                    WHERE source_id = ? AND target_id = ? AND edge_type = ?
                    """,
                    rows,
                )
                if conn.total_changes > changes_before:
                    self._bump_epoch(conn, graph=True)
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def graph_integrity_errors(self) -> list[str]:
        with self._lock:
            conn = self._db.get_connection()
            rows = conn.execute(
                """
                SELECT e.source_id, e.target_id
                FROM local_graph_edges e
                LEFT JOIN local_records source_record
                    ON source_record.storage_key = e.source_id
                LEFT JOIN local_records target_record
                    ON target_record.storage_key = e.target_id
                WHERE source_record.storage_key IS NULL
                   OR target_record.storage_key IS NULL
                ORDER BY e.source_id, e.target_id
                """
            ).fetchall()
            errors = [
                f"dangling graph edge: {row['source_id']} -> {row['target_id']}"
                for row in rows
            ]
            for row in conn.execute(
                "SELECT source_id, target_id FROM local_graph_edges "
                "ORDER BY source_id, target_id"
            ):
                for column in ("source_id", "target_id"):
                    try:
                        self._canonical_graph_storage_key(row[column])
                    except ValueError as exc:
                        errors.append(str(exc))
            return sorted(set(errors))

    def check_graph_integrity(self) -> bool:
        return not self.graph_integrity_errors()

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
        identity_key = self._canonical_graph_storage_key(identity_key)
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
                    SELECT e.source_id, e.target_id, e.edge_type, e.weight
                    FROM local_graph_edges e
                    JOIN local_records target_record
                        ON target_record.storage_key = e.target_id
                    WHERE e.source_id IN ({placeholders})
                    """,
                    tuple(frontier),
                ).fetchall()
                next_frontier: set[str] = set()
                for row in rows:
                    if allowed is not None and row["edge_type"] not in allowed:
                        continue
                    try:
                        target_id = self._canonical_graph_storage_key(row["target_id"])
                    except ValueError:
                        continue
                    if row["target_id"] == identity_key:
                        continue
                    weight = float(row["weight"])
                    if hop:
                        parent_weight = best.get(row["source_id"], ("", 1.0))[1]
                        weight *= parent_weight
                    current = best.get(target_id)
                    if current is None or weight > current[1]:
                        best[target_id] = (row["edge_type"], weight)
                    next_frontier.add(target_id)
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

    def neighbors_many(
        self,
        identities: Sequence[RecordIdentity],
        *,
        depth: int,
    ) -> dict[str, list[GraphNeighbor]]:
        """Retrieve neighbors for multiple seeds with one query per hop."""
        if depth < 1:
            raise ValueError("depth must be positive")
        seed_keys = list(dict.fromkeys(identity.storage_key for identity in identities))
        frontiers = {seed_key: {seed_key} for seed_key in seed_keys}
        best_by_seed: dict[str, dict[str, tuple[str, float]]] = {
            seed_key: {} for seed_key in seed_keys
        }
        with self._lock:
            conn = self._db.get_connection()
            for hop in range(depth):
                owners: dict[str, list[str]] = {}
                for seed_key, frontier in frontiers.items():
                    for source_key in frontier:
                        owners.setdefault(source_key, []).append(seed_key)
                if not owners:
                    break
                placeholders = ",".join("?" for _ in owners)
                rows = conn.execute(
                    f"""
                    SELECT e.source_id, e.target_id, e.edge_type, e.weight
                    FROM local_graph_edges e
                    JOIN local_records target_record
                        ON target_record.storage_key = e.target_id
                    WHERE e.source_id IN ({placeholders})
                    """,
                    tuple(owners),
                ).fetchall()
                next_frontiers = {seed_key: set() for seed_key in seed_keys}
                for row in rows:
                    try:
                        target_id = self._canonical_graph_storage_key(row["target_id"])
                    except ValueError:
                        continue
                    for seed_key in owners.get(row["source_id"], ()):
                        if target_id == seed_key:
                            continue
                        if hop:
                            parent_weight = best_by_seed[seed_key].get(
                                row["source_id"], ("", 1.0)
                            )[1]
                            weight = float(row["weight"]) * parent_weight
                        else:
                            weight = float(row["weight"])
                        current = best_by_seed[seed_key].get(target_id)
                        if current is None or weight > current[1]:
                            best_by_seed[seed_key][target_id] = (
                                row["edge_type"],
                                weight,
                            )
                        next_frontiers[seed_key].add(target_id)
                frontiers = next_frontiers
        return {
            seed_key: sorted(
                (
                    GraphNeighbor(
                        self._identity_from_storage_key(target_id),
                        edge_type,
                        weight,
                    )
                    for target_id, (edge_type, weight) in best.items()
                ),
                key=lambda item: (
                    -item.weight,
                    item.identity.storage_key,
                    item.edge_type,
                ),
            )
            for seed_key, best in best_by_seed.items()
        }

    @staticmethod
    def _identity_from_storage_key(storage_key: str) -> RecordIdentity:
        LocalRecordBackend._canonical_graph_storage_key(storage_key)
        return RecordIdentity.from_storage_key(storage_key)

class LocalVectorStore:
    """VectorStore implementation backed by the local record database."""

    def __init__(
        self,
        backend: LocalRecordBackend,
        *,
        engine: str | None = None,
        faiss_path: Path | None = None,
    ):
        self._backend = backend
        self._engine = engine or backend.vector_engine
        if self._engine not in {"exact", "faiss", "auto"}:
            raise ValueError("engine must be exact, faiss, or auto")
        self._faiss_path = faiss_path
        self._faiss_store: Any | None = None
        self._last_engine_name = (
            "faiss" if self._engine == "faiss" else "sqlite-exact"
        )

    @property
    def engine_name(self) -> str:
        return self._last_engine_name

    def _selected_store(
        self,
        model_name: str,
        dim: int,
    ) -> Any | None:
        use_faiss = self._engine == "faiss" or (
            self._engine == "auto"
            and self._backend.vector_count(model_name, dim)
            >= self._backend.faiss_threshold
        )
        if not use_faiss:
            self._last_engine_name = "sqlite-exact"
            return None
        self._last_engine_name = "faiss"
        if self._faiss_store is None:
            from searchkernel.indices.faiss_local import FAISSLocalVectorStore

            self._faiss_store = FAISSLocalVectorStore(
                self._backend,
                index_path=self._faiss_path,
            )
        return self._faiss_store

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
        faiss_store = self._selected_store(model_name, dim)
        if faiss_store is not None:
            return faiss_store.search(
                query_vector,
                k,
                model_name=model_name,
                dim=dim,
                filters=filters,
            )
        return self._backend.search_vector(
            query_vector, k, model_name=model_name, dim=dim, filters=filters
        )

    async def async_search(
        self,
        query_vector: Vector,
        k: int,
        *,
        model_name: str,
        dim: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RecordHit]:
        return await asyncio.to_thread(
            self.search,
            query_vector,
            k,
            model_name=model_name,
            dim=dim,
            filters=filters,
        )

    def delete(self, record_ids: list[str]) -> None:
        self._backend.delete(record_ids)

    def epoch(self) -> int:
        return self._backend.epoch()

    def vector_epoch(self) -> int:
        return self._backend.vector_epoch()

    def epochs(self) -> dict[str, int]:
        return self._backend.epochs()


class SQLiteExactVectorStore(LocalVectorStore):
    """Truthful name for the default packed exact vector engine."""

    def __init__(self, backend: LocalRecordBackend):
        super().__init__(backend, engine="exact")


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

    @property
    def keyword_index_available(self) -> bool:
        return self._backend.keyword_index_available

    @property
    def keyword_search_diagnostic(self) -> str:
        return self._backend.keyword_search_diagnostic

    def check_keyword_index(self) -> bool:
        return self._backend.check_keyword_index()

    def rebuild_keyword_index(self) -> None:
        self._backend.rebuild_keyword_index()

    def keyword_epoch(self) -> int:
        return self._backend.keyword_epoch()

    def epochs(self) -> dict[str, int]:
        return self._backend.epochs()


class LocalGraphStore:
    """GraphStore implementation backed by SQLite graph rows."""

    def __init__(self, backend: LocalRecordBackend):
        self._backend = backend

    def upsert_edges(
        self,
        edges: Sequence[GraphEdge | tuple[str, str, str, float]],
    ) -> None:
        self._backend.upsert_edges(edges)

    def graph_epoch(self) -> int:
        return self._backend.graph_epoch()

    def graph_integrity_errors(self) -> list[str]:
        return self._backend.graph_integrity_errors()

    def check_graph_integrity(self) -> bool:
        return self._backend.check_graph_integrity()

    def delete_edges(
        self,
        edges: Sequence[GraphEdge | tuple[str, str, str, float]],
    ) -> None:
        self._backend.delete_edges(edges)

    def epochs(self) -> dict[str, int]:
        return self._backend.epochs()

    def neighbors(
        self,
        record_id: RecordIdentity | str,
        edge_types: list[str] | None = None,
        depth: int = 1,
    ) -> list[GraphNeighbor]:
        return self._backend.neighbors(record_id, edge_types, depth)

    def neighbors_many(
        self,
        identities: Sequence[RecordIdentity],
        *,
        depth: int,
    ) -> dict[str, list[GraphNeighbor]]:
        return self._backend.neighbors_many(identities, depth=depth)


FAISSVectorStore = FAISSLocalVectorStore
SQLiteKeywordStore = LocalKeywordStore
SQLiteGraphStore = LocalGraphStore

__all__ = [
    "FAISSLocalVectorStore",
    "FAISSVectorStore",
    "LocalGraphStore",
    "LocalKeywordStore",
    "LocalRecordBackend",
    "LocalVectorStore",
    "SQLiteExactVectorStore",
    "SQLiteGraphStore",
    "SQLiteKeywordStore",
]
