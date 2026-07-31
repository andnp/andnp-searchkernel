from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path
from typing import Self

logger = logging.getLogger(__name__)


class _ManagedConnection(sqlite3.Connection):
    def __del__(self) -> None:
        try:
            self.close()
        except (AttributeError, sqlite3.Error):
            return


class DatabaseManager:
    """Thread-safe SQLite database manager with WAL mode."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
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
            check_same_thread=False,
            factory=_ManagedConnection,
        )
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
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

            CREATE VIRTUAL TABLE IF NOT EXISTS search_index USING fts5(
                chunk_id UNINDEXED,
                doc_id UNINDEXED,
                content,
                title,
                headers,
                tags,
                source_file UNINDEXED
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
        conn.commit()

    def initialize_schema(self) -> None:
        self._initialize_schema_on(self.get_connection())

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
