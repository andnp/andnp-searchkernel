"""Indexed persistence for chunk content hashes."""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Iterable
from pathlib import Path

from searchkernel.domain import Chunk

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 1


class ChunkHashStore:
    """Store chunk hashes in SQLite while retaining the legacy JSON API.

    The JSON path remains a compatibility snapshot written by ``persist``.
    SQLite is the source of truth and is queried directly, so startup memory
    does not scale with the number of chunks.
    """

    def __init__(self, storage_path: Path):
        self._storage_path = Path(storage_path)
        self._db_path = self._storage_path.with_suffix(".sqlite3")
        self._dirty: set[str] = set()
        self._initialize_database()

    @property
    def db_path(self) -> Path:
        return self._db_path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _initialize_database(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._initialize_database_once()
        except sqlite3.DatabaseError:
            corrupt_path = self._db_path.with_suffix(self._db_path.suffix + ".corrupt")
            try:
                if self._db_path.exists():
                    self._db_path.replace(corrupt_path)
            except OSError as exc:
                raise RuntimeError(
                    f"cannot recover corrupt hash database {self._db_path}"
                ) from exc
            logger.warning(
                "Recovered corrupt hash database %s as %s",
                self._db_path,
                corrupt_path,
            )
            self._initialize_database_once()

    def _initialize_database_once(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS chunk_hashes (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT,
                    content_hash TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_chunk_hashes_document
                    ON chunk_hashes (document_id, chunk_id);
                CREATE INDEX IF NOT EXISTS idx_chunk_hashes_content
                    ON chunk_hashes (content_hash, chunk_id);
                CREATE TABLE IF NOT EXISTS chunk_hash_store_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                INSERT INTO chunk_hash_store_meta (key, value)
                VALUES ('schema_version', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (str(_SCHEMA_VERSION),),
            )
            conn.commit()
        if self._storage_path.exists():
            self._migrate_json()

    def _migration_status(self, conn: sqlite3.Connection) -> str | None:
        row = conn.execute(
            "SELECT value FROM chunk_hash_store_meta WHERE key = 'json_migration'"
        ).fetchone()
        return str(row[0]) if row else None

    @staticmethod
    def _document_id(chunk_id: str) -> str | None:
        for marker in ("#chunk-", "_chunk_", "#chunk#"):
            if marker in chunk_id:
                return chunk_id.split(marker, 1)[0]
        return None

    @staticmethod
    def _load_json(path: Path) -> dict[str, str]:
        try:
            with path.open() as file:
                payload = json.load(file)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid chunk hash JSON at {path}") from exc
        if not isinstance(payload, dict) or not all(
            isinstance(chunk_id, str) and isinstance(content_hash, str)
            for chunk_id, content_hash in payload.items()
        ):
            raise ValueError("hash store must contain string key/value pairs")
        return payload

    def _set_migration_status(
        self,
        conn: sqlite3.Connection,
        status: str,
    ) -> None:
        conn.execute(
            """
            INSERT INTO chunk_hash_store_meta (key, value)
            VALUES ('json_migration', ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (status,),
        )

    def _migrate_json(self, *, force: bool = False) -> bool:
        with self._connect() as conn:
            status = self._migration_status(conn)
        if status == "complete" and not force:
            return False

        try:
            hashes = self._load_json(self._storage_path)
        except ValueError as exc:
            with self._connect() as conn:
                self._set_migration_status(conn, f"failed:{exc}")
                conn.commit()
            logger.warning("Skipping corrupt chunk hash JSON: %s", exc)
            return False

        try:
            with self._connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                self._set_migration_status(conn, "in_progress")
                conn.executemany(
                    """
                    INSERT INTO chunk_hashes (chunk_id, document_id, content_hash)
                    VALUES (?, ?, ?)
                    ON CONFLICT(chunk_id) DO UPDATE SET
                        document_id = excluded.document_id,
                        content_hash = excluded.content_hash
                    """,
                    [
                        (chunk_id, self._document_id(chunk_id), content_hash)
                        for chunk_id, content_hash in hashes.items()
                    ],
                )
                self._set_migration_status(conn, "complete")
                conn.commit()
        except Exception:
            with self._connect() as conn:
                conn.rollback()
            raise
        logger.info("Migrated %d chunk hashes from %s", len(hashes), self._storage_path)
        return True

    def migrate_json(self, *, force: bool = False) -> bool:
        """Explicitly retry the one-time JSON migration."""
        if not self._storage_path.exists():
            return False
        return self._migrate_json(force=force)

    def _mark_dirty(self, chunk_ids: Iterable[str]) -> None:
        self._dirty.update(chunk_ids)

    def persist(self) -> None:
        """Write a deterministic legacy JSON compatibility snapshot."""
        if not self._dirty:
            return
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT chunk_id, content_hash FROM chunk_hashes ORDER BY rowid"
                ).fetchall()
            payload = {str(row[0]): str(row[1]) for row in rows}
            self._storage_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = self._storage_path.with_suffix(
                self._storage_path.suffix + ".tmp"
            )
            temporary_path.write_text(
                json.dumps(payload, ensure_ascii=False, sort_keys=False)
            )
            temporary_path.replace(self._storage_path)
            self._dirty.clear()
        except (OSError, sqlite3.DatabaseError):
            logger.exception("Failed to persist hash store")

    def get_hash(self, chunk_id: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT content_hash FROM chunk_hashes WHERE chunk_id = ?",
                (chunk_id,),
            ).fetchone()
        return str(row[0]) if row else None

    def set_hash(self, chunk_id: str, content_hash: str) -> None:
        self.set_hashes([(chunk_id, content_hash)])

    def set_hashes(self, hashes: Iterable[tuple[str, str]]) -> None:
        rows = list(hashes)
        if not rows:
            return
        if any(
            not isinstance(chunk_id, str)
            or not isinstance(content_hash, str)
            or not chunk_id
            for chunk_id, content_hash in rows
        ):
            raise ValueError("chunk hashes require non-empty string IDs and hashes")
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO chunk_hashes (chunk_id, document_id, content_hash)
                VALUES (?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    document_id = excluded.document_id,
                    content_hash = excluded.content_hash
                """,
                [
                    (chunk_id, self._document_id(chunk_id), content_hash)
                    for chunk_id, content_hash in rows
                ],
            )
            conn.commit()
        self._mark_dirty(chunk_id for chunk_id, _ in rows)

    def remove_document(self, doc_id: str) -> None:
        self.remove_documents([doc_id])

    def remove_documents(self, doc_ids: Iterable[str]) -> None:
        document_ids = list(dict.fromkeys(doc_ids))
        if not document_ids:
            return
        placeholders = ",".join("?" for _ in document_ids)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT chunk_id FROM chunk_hashes WHERE document_id IN ({placeholders})",
                document_ids,
            ).fetchall()
            conn.execute(
                f"DELETE FROM chunk_hashes WHERE document_id IN ({placeholders})",
                document_ids,
            )
            conn.commit()
        self._mark_dirty(row[0] for row in rows)

    def remove_chunk(self, chunk_id: str) -> None:
        self.delete_chunks([chunk_id])

    def delete_chunks(self, chunk_ids: Iterable[str]) -> None:
        ids = list(dict.fromkeys(chunk_ids))
        if not ids:
            return
        placeholders = ",".join("?" for _ in ids)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT chunk_id FROM chunk_hashes WHERE chunk_id IN ({placeholders})",
                ids,
            ).fetchall()
            conn.execute(
                f"DELETE FROM chunk_hashes WHERE chunk_id IN ({placeholders})",
                ids,
            )
            conn.commit()
        self._mark_dirty(row[0] for row in rows)

    def clear(self) -> None:
        with self._connect() as conn:
            rows = conn.execute("SELECT chunk_id FROM chunk_hashes").fetchall()
            conn.execute("DELETE FROM chunk_hashes")
            conn.commit()
        self._mark_dirty(row[0] for row in rows)
        self._dirty.add("__cleared__")

    def has_changed(self, chunk: Chunk) -> bool:
        """Return whether the stored hash differs from the current chunk."""
        return self.get_hash(chunk.chunk_id) != chunk.content_hash

    def get_chunk_id_by_hash(self, content_hash: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT chunk_id
                FROM chunk_hashes
                WHERE content_hash = ?
                ORDER BY rowid
                LIMIT 1
                """,
                (content_hash,),
            ).fetchone()
        return str(row[0]) if row else None

    def get_chunks_by_document(self, doc_id: str) -> list[tuple[str, str]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT chunk_id, content_hash
                FROM chunk_hashes
                WHERE document_id = ?
                ORDER BY rowid
                """,
                (doc_id,),
            ).fetchall()
        return [(str(row[0]), str(row[1])) for row in rows]
