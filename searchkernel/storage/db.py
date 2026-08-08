from __future__ import annotations

import logging
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Self

logger = logging.getLogger(__name__)


class SQLiteDatabase(Protocol):
    """Connection provider used by SQLite-backed storage components."""

    def get_connection(self) -> sqlite3.Connection: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class SQLiteTuning:
    """Validated connection and checkpoint settings for local SQLite stores."""

    busy_timeout_ms: int = 5_000
    page_size: int | None = None
    cache_size: int | None = None
    mmap_size: int = 0
    temp_store: str | int = "default"
    checkpoint_policy: str = "passive"
    checkpoint_interval: int = 1_000

    def __post_init__(self) -> None:
        if (
            not isinstance(self.busy_timeout_ms, int)
            or isinstance(self.busy_timeout_ms, bool)
            or self.busy_timeout_ms < 0
        ):
            raise ValueError("busy_timeout_ms must be non-negative")
        if self.page_size is not None and (
            not isinstance(self.page_size, int)
            or isinstance(self.page_size, bool)
            or self.page_size < 512
            or self.page_size > 65_536
            or self.page_size & (self.page_size - 1)
        ):
            raise ValueError("page_size must be a power of two between 512 and 65536")
        if self.cache_size is not None and (
            not isinstance(self.cache_size, int)
            or isinstance(self.cache_size, bool)
            or self.cache_size == 0
        ):
            raise ValueError("cache_size must be non-zero when configured")
        if (
            not isinstance(self.mmap_size, int)
            or isinstance(self.mmap_size, bool)
            or self.mmap_size < 0
        ):
            raise ValueError("mmap_size must be non-negative")
        if isinstance(self.temp_store, str):
            temp_store = self.temp_store.lower()
            if temp_store not in {"default", "file", "memory"}:
                raise ValueError("temp_store must be default, file, or memory")
        elif (
            not isinstance(self.temp_store, int)
            or isinstance(self.temp_store, bool)
            or self.temp_store not in {0, 1, 2}
        ):
            raise ValueError("temp_store must be 0, 1, or 2")
        if not isinstance(self.checkpoint_policy, str) or self.checkpoint_policy not in {
            "none",
            "manual",
            "passive",
            "full",
            "restart",
            "truncate",
        }:
            raise ValueError("unsupported checkpoint_policy")
        if (
            not isinstance(self.checkpoint_interval, int)
            or isinstance(self.checkpoint_interval, bool)
            or self.checkpoint_interval < 0
        ):
            raise ValueError("checkpoint_interval must be non-negative")

    @property
    def temp_store_value(self) -> int:
        if isinstance(self.temp_store, int):
            return self.temp_store
        return {"default": 0, "file": 1, "memory": 2}[self.temp_store.lower()]

    @property
    def auto_checkpoint(self) -> int:
        if self.checkpoint_policy in {"none", "manual"}:
            return 0
        return self.checkpoint_interval

    def apply(self, conn: sqlite3.Connection) -> None:
        if self.page_size is not None:
            conn.execute(f"PRAGMA page_size = {self.page_size}")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute(f"PRAGMA busy_timeout = {self.busy_timeout_ms}")
        if self.cache_size is not None:
            conn.execute(f"PRAGMA cache_size = {self.cache_size}")
        conn.execute(f"PRAGMA mmap_size = {self.mmap_size}")
        conn.execute(f"PRAGMA temp_store = {self.temp_store_value}")
        conn.execute(f"PRAGMA wal_autocheckpoint = {self.auto_checkpoint}")

    def checkpoint(
        self,
        conn: sqlite3.Connection,
        *,
        policy: str | None = None,
    ) -> tuple[int, int, int]:
        selected_policy = policy or self.checkpoint_policy
        if selected_policy not in {
            "none",
            "manual",
            "passive",
            "full",
            "restart",
            "truncate",
        }:
            raise ValueError("unsupported checkpoint policy")
        if selected_policy == "manual":
            selected_policy = "passive"
        if selected_policy == "none":
            return (0, 0, 0)
        row = conn.execute(
            f"PRAGMA wal_checkpoint({selected_policy})"
        ).fetchone()
        if row is None:
            return (0, 0, 0)
        return (int(row[0]), int(row[1]), int(row[2]))


class _ManagedConnection(sqlite3.Connection):
    def __del__(self) -> None:
        try:
            self.close()
        except (AttributeError, sqlite3.Error):
            return


class InMemorySQLiteDatabase:
    """Single-connection SQLite provider for ephemeral local stores."""

    def __init__(self, tuning: SQLiteTuning | None = None) -> None:
        self._connection = sqlite3.connect(":memory:", check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        (tuning or SQLiteTuning()).apply(self._connection)
        self._connection.execute("PRAGMA foreign_keys=ON")

    def get_connection(self) -> sqlite3.Connection:
        return self._connection

    def close(self) -> None:
        self._connection.close()


class DatabaseManager:
    """Thread-safe SQLite database manager with WAL mode."""

    def __init__(
        self,
        db_path: Path,
        *,
        tuning: SQLiteTuning | None = None,
    ) -> None:
        self._db_path = db_path
        self._tuning = tuning or SQLiteTuning()
        self._local = threading.local()
        self._connections: dict[int, sqlite3.Connection] = {}
        self._connections_lock = threading.Lock()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # Initialize schema via a temporary connection
        conn = self._open_connection()
        self._initialize_schema_on(conn)
        conn.close()

    def _open_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self._db_path),
            timeout=self._tuning.busy_timeout_ms / 1_000,
            check_same_thread=False,
            factory=_ManagedConnection,
        )
        self._tuning.apply(conn)
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.row_factory = sqlite3.Row
        return conn

    def get_connection(self) -> sqlite3.Connection:
        """Return a per-thread SQLite connection."""
        conn = getattr(self._local, "connection", None)
        if conn is None:
            thread_id = threading.get_ident()
            with self._connections_lock:
                conn = self._connections.get(thread_id)
                if conn is None:
                    conn = self._open_connection()
                    self._connections[thread_id] = conn
                self._local.connection = conn
        return conn

    def _initialize_schema_on(self, conn: sqlite3.Connection) -> None:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS kv_store (
                key TEXT PRIMARY KEY,
                value TEXT
            );

            CREATE TABLE IF NOT EXISTS graph_nodes (
                node_id TEXT PRIMARY KEY,
                metadata TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS graph_edges (
                source TEXT NOT NULL,
                target TEXT NOT NULL,
                edge_type TEXT NOT NULL DEFAULT 'related_to',
                edge_context TEXT DEFAULT '',
                PRIMARY KEY (source, target, edge_type),
                FOREIGN KEY (source) REFERENCES graph_nodes(node_id) ON DELETE CASCADE,
                FOREIGN KEY (target) REFERENCES graph_nodes(node_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_graph_edges_source ON graph_edges(source);
            CREATE INDEX IF NOT EXISTS idx_graph_edges_target ON graph_edges(target);
            CREATE INDEX IF NOT EXISTS idx_graph_edges_type ON graph_edges(edge_type);

            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                task_name TEXT NOT NULL,
                data TEXT DEFAULT '{}',
                status TEXT NOT NULL DEFAULT 'pending',
                created_at REAL NOT NULL,
                updated_at REAL
            );

            CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);

            CREATE TABLE IF NOT EXISTS system_state (
                key TEXT PRIMARY KEY,
                value TEXT
            );
        """)
        try:
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS search_index USING fts5(
                    chunk_id UNINDEXED,
                    doc_id UNINDEXED,
                    content,
                    title,
                    headers,
                    tags,
                    source_file UNINDEXED
                )
                """
            )
        except sqlite3.OperationalError as exc:
            if "fts5" not in str(exc).lower():
                raise
            logger.info("SQLite FTS5 is unavailable; keyword index is disabled")
        conn.commit()

    def initialize_schema(self) -> None:
        self._initialize_schema_on(self.get_connection())

    @property
    def tuning(self) -> SQLiteTuning:
        return self._tuning

    def checkpoint(self, *, policy: str | None = None) -> tuple[int, int, int]:
        return self._tuning.checkpoint(self.get_connection(), policy=policy)

    def close(self) -> None:
        with self._connections_lock:
            connections = tuple(self._connections.values())
            self._connections.clear()
            self._local.connection = None
        for conn in connections:
            conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except (AttributeError, sqlite3.Error):
            return
