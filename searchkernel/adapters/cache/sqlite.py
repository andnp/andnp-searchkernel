"""SQLite-backed cache store with epoch-based invalidation."""

import json
import logging
import os
import sqlite3
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from searchkernel.storage.db import SQLiteTuning

logger = logging.getLogger(__name__)
_ResultT = TypeVar("_ResultT")


class SQLiteCacheStore:
    """SQLite-backed cache store implementing the CacheStore port.

    Persists key-value pairs to a SQLite database. Entries are tagged with
    an epoch for invalidation support. Designed for durability across restarts.
    """

    def __init__(
        self,
        db_path: Path | str,
        *,
        tuning: SQLiteTuning | None = None,
    ):
        """Initialize SQLite cache store.

        Args:
            db_path: Path to SQLite database file.
        """
        self.db_path = Path(db_path)
        self._tuning = tuning or SQLiteTuning()
        self._local = threading.local()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @property
    def tuning(self) -> SQLiteTuning:
        return self._tuning

    def _init_schema(self) -> None:
        """Create the cache table if it doesn't exist."""
        conn = self._get_connection()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cache_store (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                epoch INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_cache_epoch
            ON cache_store (epoch);
        """)
        conn.commit()

    def _open_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self.db_path,
            timeout=self._tuning.busy_timeout_ms / 1_000,
        )
        self._tuning.apply(conn)
        return conn

    def _get_connection(self) -> sqlite3.Connection:
        """Return this thread's connection, replacing connections inherited by fork."""
        conn = getattr(self._local, "connection", None)
        if conn is None or getattr(self._local, "pid", None) != os.getpid():
            if conn is not None:
                self._close_connection(conn)
            conn = self._open_connection()
            self._local.connection = conn
            self._local.pid = os.getpid()
        return conn

    def _close_connection(self, conn: sqlite3.Connection) -> None:
        try:
            conn.close()
        except sqlite3.Error:
            pass

    def _discard_connection(self, conn: sqlite3.Connection) -> None:
        if getattr(self._local, "connection", None) is conn:
            self._local.connection = None
            self._local.pid = None
        self._close_connection(conn)

    def _run(self, operation: Callable[[sqlite3.Connection], _ResultT]) -> _ResultT:
        """Run an operation, reopening once when SQLite rejects a cached connection."""
        for attempt in range(2):
            conn = self._get_connection()
            try:
                return operation(conn)
            except sqlite3.DatabaseError:
                try:
                    conn.rollback()
                except sqlite3.Error:
                    pass
                self._discard_connection(conn)
                if attempt == 1:
                    raise
        raise AssertionError("unreachable")

    def get(self, key: str) -> Any | None:
        """Retrieve a cached value.

        Args:
            key: Cache key.

        Returns:
            Cached value, or None if not found.
        """
        try:
            def read(conn: sqlite3.Connection) -> Any | None:
                cursor = conn.execute(
                    "SELECT value FROM cache_store WHERE key = ?;",
                    (key,),
                )
                row = cursor.fetchone()
                if row:
                    return json.loads(row[0])
                return None

            return self._run(read)
        except (sqlite3.DatabaseError, json.JSONDecodeError) as e:
            logger.warning(f"Error retrieving cache key {key}: {e}", exc_info=True)
            return None

    def set(self, key: str, value: Any, epoch: int) -> None:
        """Store a value with an associated epoch.

        Args:
            key: Cache key.
            value: Value to cache.
            epoch: Index epoch at cache time. Used for invalidation.
        """
        try:
            value_json = json.dumps(value)
            def write(conn: sqlite3.Connection) -> None:
                conn.execute(
                    """
                    INSERT INTO cache_store (key, value, epoch)
                    VALUES (?, ?, ?)
                    ON CONFLICT (key) DO UPDATE SET
                        value = EXCLUDED.value,
                        epoch = EXCLUDED.epoch;
                    """,
                    (key, value_json, epoch),
                )
                conn.commit()

            self._run(write)
        except (sqlite3.DatabaseError, TypeError):
            logger.exception(f"Error storing cache key {key}")

    def invalidate_epoch(self, epoch: int) -> None:
        """Invalidate all entries from an epoch or earlier.

        Args:
            epoch: Entries with epoch <= this are discarded.
        """
        try:
            def delete(conn: sqlite3.Connection) -> None:
                cursor = conn.execute(
                    "DELETE FROM cache_store WHERE epoch <= ?;",
                    (epoch,),
                )
                conn.commit()
                logger.debug(
                    f"Invalidated cache entries for epochs <= {epoch} "
                    f"({cursor.rowcount} entries deleted)"
                )

            self._run(delete)
        except sqlite3.DatabaseError:
            logger.exception(f"Error invalidating cache epoch {epoch}")
