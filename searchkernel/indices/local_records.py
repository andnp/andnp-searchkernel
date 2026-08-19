"""Record and chunk persistence for the local search backend."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterable, Sequence
from types import TracebackType
from typing import Any, Protocol

from searchkernel.domain import Record
from searchkernel.utils.ordered_key_chunks import (
    DEFAULT_KEY_CHUNK_LIMIT,
    iter_ordered_key_chunks,
)

_LOCAL_FTS_TABLE = "local_records_fts"


class _Lock(Protocol):
    def __enter__(self) -> object: ...

    def __exit__(
        self,
        t: type[BaseException] | None,
        v: BaseException | None,
        tb: TracebackType | None,
    ) -> None: ...


class _SQLiteAccess(Protocol):
    @property
    def lock(self) -> _Lock: ...

    def connection(self) -> sqlite3.Connection: ...


class _EpochBump(Protocol):
    def __call__(
        self,
        conn: sqlite3.Connection,
        *,
        keyword: bool = False,
        vector: bool = False,
    ) -> None: ...


class _RecordWriter:
    """Own record and chunk persistence, including FTS row synchronization."""

    def __init__(
        self,
        access: _SQLiteAccess,
        *,
        metadata_keyword_text: Callable[[dict[str, Any]], str],
        metadata_uri: Callable[[dict[str, Any]], str],
        fts5_available: Callable[[], bool],
        epoch_bump: _EpochBump,
    ) -> None:
        self._access = access
        self._metadata_keyword_text = metadata_keyword_text
        self._metadata_uri = metadata_uri
        self._fts5_available = fts5_available
        self._epoch_bump = epoch_bump

    def _record_uri(self, record: Record) -> str:
        return record.uri or self._metadata_uri(record.metadata)

    def _record_values(self, record: Record) -> tuple[Any, ...]:
        return (
            record.storage_key,
            record.workspace_id,
            record.source_kind,
            record.source_id,
            record.title,
            record.body,
            record.indexed_text,
            record.created_at.isoformat(),
            record.updated_at.isoformat(),
            json.dumps(record.metadata, sort_keys=True),
            self._record_uri(record),
            self._metadata_keyword_text(record.metadata),
            record.status.value,
        )

    @staticmethod
    def _chunk_state_values(record: Record) -> tuple[Any, ...] | None:
        if not record.metadata.get("_searchkernel_chunk"):
            return None
        parent_key = record.metadata.get("_chunk_parent_storage_key")
        chunk_id = record.metadata.get("_chunk_id")
        chunk_index = record.metadata.get("_chunk_index")
        chunk_metadata = record.metadata.get("_chunk_metadata", {})
        if not isinstance(parent_key, str) or not isinstance(chunk_id, str):
            raise TypeError("chunk records require parent and chunk identities")
        if not isinstance(chunk_index, int) or not isinstance(chunk_metadata, dict):
            raise TypeError("chunk records require valid retrieval metadata")
        return (
            record.storage_key,
            parent_key,
            chunk_id,
            chunk_index,
            json.dumps(chunk_metadata, sort_keys=True),
        )

    def _sync_chunk_state(
        self,
        conn: sqlite3.Connection,
        records: Sequence[Record],
    ) -> bool:
        chunk_rows = [
            values
            for record in records
            if (values := self._chunk_state_values(record)) is not None
        ]
        if not chunk_rows:
            return False
        parent_keys = sorted({row[1] for row in chunk_rows})
        incoming_keys = {row[0] for row in chunk_rows}
        placeholders = ",".join("?" for _ in parent_keys)
        stale_rows = conn.execute(
            f"""
            SELECT r.storage_key, r.rowid, r.title, r.body, r.indexed_text, r.uri, r.keywords
            FROM local_records r
            WHERE r.storage_key IN (
                SELECT chunk_storage_key
                FROM local_chunk_state
                WHERE parent_storage_key IN ({placeholders})
            )
            """,
            parent_keys,
        ).fetchall()
        stale_rows = [row for row in stale_rows if row["storage_key"] not in incoming_keys]
        if self._fts5_available() and stale_rows:
            conn.executemany(
                f"""
                INSERT INTO {_LOCAL_FTS_TABLE}
                    ({_LOCAL_FTS_TABLE}, rowid, title, body, uri, keywords)
                VALUES ('delete', ?, ?, ?, ?, ?)
                """,
                [
                    (
                        row["rowid"],
                        row["title"],
                        row["indexed_text"] or row["body"],
                        row["uri"],
                        row["keywords"],
                    )
                    for row in stale_rows
                ],
            )
        stale_keys = [row["storage_key"] for row in stale_rows]
        if stale_keys:
            for key_chunk in iter_ordered_key_chunks(
                stale_keys, limit=DEFAULT_KEY_CHUNK_LIMIT
            ):
                placeholders = ",".join("?" for _ in key_chunk)
                conn.execute(
                    f"DELETE FROM local_records WHERE storage_key IN ({placeholders})",
                    key_chunk,
                )
        conn.executemany(
            """
            INSERT INTO local_chunk_state (
                chunk_storage_key, parent_storage_key, chunk_id, chunk_index, metadata
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(chunk_storage_key) DO UPDATE SET
                parent_storage_key = excluded.parent_storage_key,
                chunk_id = excluded.chunk_id,
                chunk_index = excluded.chunk_index,
                metadata = excluded.metadata
            """,
            chunk_rows,
        )
        return bool(stale_rows)

    def _write_records(
        self,
        conn: sqlite3.Connection,
        rows: list[Record],
    ) -> tuple[bool, bool]:
        keys = list(dict.fromkeys(record.storage_key for record in rows))
        old_rows: dict[str, sqlite3.Row] = {}
        for key_chunk in iter_ordered_key_chunks(
            keys, limit=DEFAULT_KEY_CHUNK_LIMIT
        ):
            placeholders = ",".join("?" for _ in key_chunk)
            old_rows.update(
                {
                    row["storage_key"]: row
                    for row in conn.execute(
                        f"""
                        SELECT storage_key, rowid, workspace_id, source_kind, source_id,
                               title, body, indexed_text, created_at, updated_at,
                               metadata, uri, keywords, status
                        FROM local_records
                        WHERE storage_key IN ({placeholders})
                        """,
                        key_chunk,
                    ).fetchall()
                }
            )

        canonical_values = {
            key: tuple(
                row[column]
                for column in (
                    "workspace_id",
                    "source_kind",
                    "source_id",
                    "title",
                    "body",
                    "indexed_text",
                    "created_at",
                    "updated_at",
                    "metadata",
                    "uri",
                    "keywords",
                    "status",
                )
            )
            for key, row in old_rows.items()
        }
        changed_rows: list[Record] = []
        fts_changed_keys: set[str] = set()
        keyword_changed = False
        for record in rows:
            values = self._record_values(record)
            previous = canonical_values.get(record.storage_key)
            if previous == values[1:]:
                continue
            changed_rows.append(record)
            canonical_values[record.storage_key] = values[1:]
            if previous is None or (
                previous[3] != values[4]
                or (previous[5] or previous[4]) != (values[6] or values[5])
                or previous[9] != values[10]
                or previous[10] != values[11]
            ):
                fts_changed_keys.add(record.storage_key)
            if previous is None or (
                previous[3] != values[4]
                or (previous[5] or previous[4]) != (values[6] or values[5])
                or previous[8] != values[9]
                or previous[9] != values[10]
                or previous[10] != values[11]
                or previous[11] != values[12]
            ):
                keyword_changed = True

        if self._fts5_available():
            conn.executemany(
                f"""
                INSERT INTO {_LOCAL_FTS_TABLE}
                    ({_LOCAL_FTS_TABLE}, rowid, title, body, uri, keywords)
                VALUES ('delete', ?, ?, ?, ?, ?)
                """,
                [
                    (
                        row["rowid"],
                        row["title"],
                        row["indexed_text"] or row["body"],
                        row["uri"],
                        row["keywords"],
                    )
                    for key in keys
                    if key in old_rows and key in fts_changed_keys
                    for row in [old_rows[key]]
                ],
            )
        if changed_rows:
            conn.executemany(
                """
                INSERT INTO local_records (
                    storage_key, workspace_id, source_kind, source_id, title, body,
                    indexed_text, created_at, updated_at, metadata, uri, keywords, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(storage_key) DO UPDATE SET
                    workspace_id = excluded.workspace_id,
                    source_kind = excluded.source_kind,
                    source_id = excluded.source_id,
                    title = excluded.title,
                    body = excluded.body,
                    indexed_text = excluded.indexed_text,
                    created_at = excluded.created_at,
                    updated_at = excluded.updated_at,
                    metadata = excluded.metadata,
                    uri = excluded.uri,
                    keywords = excluded.keywords,
                    status = excluded.status
                """,
                [self._record_values(record) for record in changed_rows],
            )

        if self._fts5_available():
            new_rows: list[sqlite3.Row] = []
            for key_chunk in iter_ordered_key_chunks(
                [key for key in keys if key in fts_changed_keys],
                limit=DEFAULT_KEY_CHUNK_LIMIT,
            ):
                placeholders = ",".join("?" for _ in key_chunk)
                new_rows.extend(
                    conn.execute(
                        f"""
                        SELECT rowid, title, body, indexed_text, uri, keywords
                        FROM local_records
                        WHERE storage_key IN ({placeholders})
                        """,
                        key_chunk,
                    ).fetchall()
                )
            conn.executemany(
                f"""
                INSERT INTO {_LOCAL_FTS_TABLE}
                    (rowid, title, body, uri, keywords)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        row["rowid"],
                        row["title"],
                        row["indexed_text"] or row["body"],
                        row["uri"],
                        row["keywords"],
                    )
                    for row in new_rows
                ],
            )
        stale_rows_removed = self._sync_chunk_state(conn, rows)
        return (
            bool(changed_rows) or stale_rows_removed,
            keyword_changed or stale_rows_removed,
        )

    @staticmethod
    def _records_have_vectors(
        conn: sqlite3.Connection, records: Sequence[Record]
    ) -> bool:
        keys = list(dict.fromkeys(record.storage_key for record in records))
        for key_chunk in iter_ordered_key_chunks(
            keys, limit=DEFAULT_KEY_CHUNK_LIMIT
        ):
            placeholders = ",".join("?" for _ in key_chunk)
            row = conn.execute(
                f"SELECT 1 FROM local_vectors_v2 WHERE storage_key IN ({placeholders}) LIMIT 1",
                key_chunk,
            ).fetchone()
            if row is not None:
                return True
        return False

    def _upsert_records(self, records: Iterable[Record]) -> None:
        rows = list(records)
        if not rows:
            return
        with self._access.lock:
            conn = self._access.connection()
            try:
                vector_affected = self._records_have_vectors(conn, rows)
                record_changed, keyword_changed = self._write_records(conn, rows)
                if keyword_changed or (vector_affected and record_changed):
                    self._epoch_bump(
                        conn,
                        keyword=keyword_changed,
                        vector=vector_affected and record_changed,
                    )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
