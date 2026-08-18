"""Persistent, encoder-namespaced storage for embedding vectors."""

from __future__ import annotations

import math
import os
import sqlite3
import threading
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np

from searchkernel.utils.ordered_key_chunks import (
    DEFAULT_KEY_CHUNK_LIMIT,
    iter_ordered_key_chunks,
)

_SQLITE_LOOKUP_BATCH_SIZE = DEFAULT_KEY_CHUNK_LIMIT


@dataclass(frozen=True, slots=True)
class EmbeddingCacheMetrics:
    """Counters collected by one cache instance."""

    hits: int = 0
    misses: int = 0
    writes: int = 0
    invalidations: int = 0


class SQLiteEmbeddingCache:
    """A small SQLite cache whose files are independent of vector indices."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        encoder_namespace: str,
        dimension: int = 0,
        *,
        validate: bool = False,
    ) -> None:
        if dimension < 0:
            raise ValueError("embedding dimension must not be negative")
        self.path = Path(path)
        self.encoder_namespace = encoder_namespace
        self.dimension = dimension
        self._lock = threading.RLock()
        self._metrics = EmbeddingCacheMetrics()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = self._open_or_recover(validate=validate)

    @property
    def metrics(self) -> EmbeddingCacheMetrics:
        return self._metrics

    def get_many(self, content_hashes: Sequence[str]) -> Mapping[str, Sequence[float]]:
        """Return valid vectors for this encoder namespace."""
        batches = tuple(
            iter_ordered_key_chunks(
                content_hashes,
                limit=_SQLITE_LOOKUP_BATCH_SIZE,
            )
        )
        if not batches:
            return {}
        hash_count = sum(len(batch) for batch in batches)
        with self._lock:
            try:
                rows = []
                for batch in batches:
                    placeholders = ",".join("?" for _ in batch)
                    rows.extend(
                        self._connection.execute(
                            f"SELECT content_hash, vector FROM embeddings "
                            f"WHERE namespace = ? AND content_hash IN ({placeholders})",
                            (self.encoder_namespace, *batch),
                        ).fetchall()
                    )
            except sqlite3.DatabaseError as error:
                if not self._is_malformed(error):
                    raise
                self._recover()
                self._add_metrics(misses=hash_count, invalidations=1)
                return {}

            result: dict[str, Sequence[float]] = {}
            invalid: list[str] = []
            for content_hash, encoded in rows:
                try:
                    vector = self._validated_vector(self._decode_vector(encoded))
                except (ValueError, TypeError):
                    invalid.append(content_hash)
                    continue
                result[content_hash] = list(vector)

            if invalid:
                self._connection.executemany(
                    "DELETE FROM embeddings WHERE namespace = ? AND content_hash = ?",
                    ((self.encoder_namespace, key) for key in invalid),
                )
                self._connection.commit()
            self._add_metrics(
                hits=len(result),
                misses=hash_count - len(result),
                invalidations=len(invalid),
            )
            return result

    def put_many(self, vectors: Mapping[str, Sequence[float]]) -> None:
        """Atomically persist a batch after validating every vector."""
        if not vectors:
            return
        prepared = []
        for content_hash, vector in vectors.items():
            normalized = self._validated_vector(vector)
            with np.errstate(over="ignore"):
                packed = np.asarray(normalized, dtype="<f4")
            if not np.isfinite(packed).all():
                raise ValueError(
                    "embedding vector cannot be represented as finite float32 values"
                )
            prepared.append(
                (
                    self.encoder_namespace,
                    content_hash,
                    self.dimension,
                    # float32 matches faiss's internal representation and halves
                    # storage versus JSON text; embeddings round-trip at this
                    # precision without meaningful loss.
                    packed.tobytes(),
                )
            )
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                self._connection.executemany(
                    "INSERT OR REPLACE INTO embeddings "
                    "(namespace, content_hash, dimension, vector) VALUES (?, ?, ?, ?)",
                    prepared,
                )
                self._connection.commit()
            except sqlite3.DatabaseError as error:
                self._connection.rollback()
                if not self._is_malformed(error):
                    raise
                self._recover()
                self._add_metrics(invalidations=1)
                return
            self._add_metrics(writes=len(prepared))

    def prune_to(self, content_hashes: Iterable[str]) -> int:
        """Remove cached vectors outside the supplied live content hashes."""
        keep = frozenset(content_hashes)
        with self._lock:
            rows = self._connection.execute(
                "SELECT content_hash FROM embeddings WHERE namespace = ?",
                (self.encoder_namespace,),
            ).fetchall()
            stale = [str(row[0]) for row in rows if row[0] not in keep]
            if not stale:
                return 0
            with self._connection:
                self._connection.executemany(
                    "DELETE FROM embeddings WHERE namespace = ? AND content_hash = ?",
                    ((self.encoder_namespace, content_hash) for content_hash in stale),
                )
            return len(stale)

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except (AttributeError, sqlite3.Error):
            return

    def _open_or_recover(self, *, validate: bool = False) -> sqlite3.Connection:
        connection: sqlite3.Connection | None = None
        try:
            connection = self._open()
            if validate:
                self._validate_integrity(connection)
            return connection
        except sqlite3.DatabaseError as error:
            if connection is not None:
                connection.close()
            if "locked" in str(error).lower():
                raise
            self._discard_files()
            self._add_metrics(invalidations=1)
            return self._open()

    @staticmethod
    def _validate_integrity(connection: sqlite3.Connection) -> None:
        result = connection.execute("PRAGMA integrity_check").fetchone()
        if result != ("ok",):
            raise sqlite3.DatabaseError("database integrity check failed")

    def _open(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=5, check_same_thread=False)
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS embeddings ("
                "namespace TEXT NOT NULL, content_hash TEXT NOT NULL, "
                "dimension INTEGER NOT NULL, vector TEXT NOT NULL, "
                "PRIMARY KEY (namespace, content_hash))"
            )
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(embeddings)")
            }
            if columns != {"namespace", "content_hash", "dimension", "vector"}:
                raise sqlite3.DatabaseError("embedding cache schema is malformed")
            connection.commit()
            return connection
        except sqlite3.DatabaseError:
            connection.close()
            raise

    def _recover(self) -> None:
        self._connection.close()
        self._discard_files()
        self._connection = self._open()

    def _discard_files(self) -> None:
        for suffix in ("", "-wal", "-shm"):
            try:
                self.path.with_name(self.path.name + suffix).unlink()
            except FileNotFoundError:
                pass

    def _decode_vector(self, encoded: object) -> tuple[float, ...]:
        if not isinstance(encoded, bytes):
            raise TypeError("embedding vector must be stored as bytes")
        if self.dimension and len(encoded) != self.dimension * 4:
            raise ValueError("embedding vector has unexpected byte length")
        return tuple(float(value) for value in np.frombuffer(encoded, dtype="<f4"))

    def _validated_vector(self, vector: Sequence[float]) -> tuple[float, ...]:
        try:
            normalized = tuple(float(value) for value in vector)
        except (TypeError, ValueError) as error:
            raise ValueError("embedding vector must contain finite values") from error
        if not normalized or any(not math.isfinite(value) for value in normalized):
            raise ValueError("embedding vector must contain finite values")
        if self.dimension and len(normalized) != self.dimension:
            raise ValueError("embedding vector has unexpected dimension")
        return normalized

    def _add_metrics(self, **increments: int) -> None:
        values = {
            field: getattr(self._metrics, field)
            for field in self._metrics.__dataclass_fields__
        }
        for field, amount in increments.items():
            values[field] += amount
        self._metrics = EmbeddingCacheMetrics(**values)

    @staticmethod
    def _is_malformed(error: sqlite3.DatabaseError) -> bool:
        message = str(error).lower()
        return (
            "malformed" in message
            or "not a database" in message
            or "integrity" in message
        )


PersistentEmbeddingCache = SQLiteEmbeddingCache
