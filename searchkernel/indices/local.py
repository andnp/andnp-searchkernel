"""SQLite-backed record stores for the local search backend.

The local backend deliberately speaks the same record-oriented ports as the
pgvector adapter.  Chunk-oriented FAISS/SQLite indices remain available for
the ingestion surface, but query execution uses canonical record identities.
"""

from __future__ import annotations

import asyncio
import difflib
import heapq
import json
import logging
import math
import re
import sqlite3
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Self

import numpy as np

from searchkernel.adapters.keyword_scoring.filesystem import FilesystemArtifactScorer
from searchkernel.domain import (
    GraphEdge,
    GraphNeighbor,
    ModelDimensionMismatchError,
    Record,
    RecordHit,
    RecordIdentity,
    RecordStatus,
    SearchFilters,
    Vector,
)
from searchkernel.domain.vector_filters import (
    candidate_storage_keys,
    compile_source_scoped_filters,
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
from searchkernel.indices.vector_revision import record_embedding_revision
from searchkernel.ports.keyword_scoring import KeywordArtifactScorer
from searchkernel.storage.db import (
    DatabaseManager,
    InMemorySQLiteDatabase,
    SQLiteDatabase,
    SQLiteTuning,
)
from searchkernel.utils.ordered_key_chunks import (
    DEFAULT_KEY_CHUNK_LIMIT,
    iter_ordered_key_chunks,
)

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_LOCAL_KEYWORD_SCHEMA = "local_records_fts"
_LOCAL_KEYWORD_SCHEMA_VERSION = 4
_RECORD_IDENTITY_VERSION_KEY = "record_identity_version"
_LOCAL_RECORD_IDENTITY_VERSION = 2
_LOCAL_FTS_TABLE = "local_records_fts"
_LOCAL_FTS_COLUMNS = ("title", "body", "uri", "keywords")
_FALLBACK_SCAN_MAX_ROWS = 10_000
_FILTERED_KEYWORD_OVERFETCH = 4
_KEYWORD_SQL_FILTERS = frozenset(
    {
        "candidate_ids",
        "candidate_storage_keys",
        "excluded_project_ids",
        "excluded_projects",
        "project_filter",
        "project_id",
        "project_ids",
        "source_filter",
        "source_kind",
        "source_kinds",
        "statuses",
        "workspace_id",
    }
)
_FALLBACK_SCAN_BATCH_SIZE = 1_000
_FUZZY_QUERY_MAX_TERMS = 4
_FUZZY_TERM_RATIO = 0.82
_METADATA_FIELD_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_VECTOR_EMBEDDING_BYTES = np.dtype("<f4").itemsize
_DEFAULT_VECTOR_SNAPSHOT_MAX_BYTES = 64 * 1024 * 1024
logger = logging.getLogger(__name__)


def _fuzzy_term_score(query_term: str, tokens: Sequence[str]) -> float:
    candidates = (
        token
        for token in tokens
        if token[:1] == query_term[:1]
        and abs(len(token) - len(query_term)) <= 2
    )
    return max(
        (
            difflib.SequenceMatcher(None, query_term, token).ratio()
            for token in candidates
        ),
        default=0.0,
    )


class _FallbackHeapItem:
    __slots__ = ("hit",)

    def __init__(self, hit: RecordHit) -> None:
        self.hit = hit

    def __lt__(self, other: Self) -> bool:
        if self.hit.score != other.hit.score:
            return self.hit.score < other.hit.score
        return self.hit.storage_key > other.hit.storage_key


class _VectorSnapshotCache(dict[tuple[str, int], VectorSnapshot]):
    """Clear metadata alongside snapshots when storage is externally reset."""

    def __init__(self, storage_stats: dict[tuple[str, int], tuple[int, int, int]]):
        super().__init__()
        self._storage_stats = storage_stats

    def clear(self) -> None:
        super().clear()
        self._storage_stats.clear()


class _LocalEpochLane:
    """Own epoch keys and mutations shared by the local storage lanes."""

    _RECORD_KEY = "local_record_epoch"
    _LANE_KEYS: ClassVar[dict[str, str]] = {
        "keyword": "local_keyword_epoch",
        "vector": "local_vector_epoch",
        "graph": "local_graph_epoch",
    }

    @classmethod
    def bump(
        cls,
        conn: sqlite3.Connection,
        *,
        keyword: bool = False,
        vector: bool = False,
        graph: bool = False,
    ) -> None:
        lanes = {
            cls._RECORD_KEY: True,
            cls._LANE_KEYS["keyword"]: keyword,
            cls._LANE_KEYS["vector"]: vector,
            cls._LANE_KEYS["graph"]: graph,
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

    @classmethod
    def read(cls, conn: sqlite3.Connection, key: str) -> int:
        row = conn.execute(
            "SELECT value FROM system_state WHERE key = ?",
            (key,),
        ).fetchone()
        return int(row[0]) if row else 0

    @classmethod
    def read_lanes(cls, conn: sqlite3.Connection) -> dict[str, int]:
        return {
            lane: cls.read(conn, key) for lane, key in cls._LANE_KEYS.items()
        }


@dataclass(frozen=True, slots=True)
class _SQLiteAccess:
    """Serialised access to one SQLite database, shared by every engine."""

    db: SQLiteDatabase
    lock: threading.RLock

    def connection(self) -> sqlite3.Connection:
        return self.db.get_connection()


class _GraphEngine:
    """Own graph edge schema and mutation for the local backend."""

    def __init__(self, access: _SQLiteAccess) -> None:
        self._access = access

    @staticmethod
    def _canonical_graph_storage_key(storage_key: str) -> str:
        if not isinstance(storage_key, str):
            raise TypeError("graph endpoints must be canonical storage keys")
        try:
            identity = RecordIdentity.from_storage_key(storage_key)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid canonical storage key for graph: {storage_key!r}"
            ) from exc
        if not identity.source_kind or not identity.source_id:
            raise ValueError(
                f"invalid canonical storage key for graph: {storage_key!r}"
            )
        if identity.storage_key != storage_key:
            raise ValueError(f"non-canonical graph storage key: {storage_key!r}")
        return storage_key

    @staticmethod
    def initialize_schema(conn: sqlite3.Connection) -> None:
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

    def graph_integrity_errors(self) -> list[str]:
        with self._access.lock:
            conn = self._access.connection()
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

    def _identity_from_storage_key(self, storage_key: str) -> RecordIdentity:
        self._canonical_graph_storage_key(storage_key)
        return RecordIdentity.from_storage_key(storage_key)

    def neighbors(
        self,
        record_id: RecordIdentity | str,
        edge_types: list[str] | None = None,
        depth: int = 1,
        max_neighbors: int | None = None,
    ) -> list[GraphNeighbor]:
        return self._neighbors_direction(
            record_id,
            edge_types,
            depth,
            max_neighbors,
            incoming=False,
        )

    def incoming_neighbors(
        self,
        record_id: RecordIdentity | str,
        edge_types: list[str] | None = None,
        depth: int = 1,
        max_neighbors: int | None = None,
    ) -> list[GraphNeighbor]:
        return self._neighbors_direction(
            record_id,
            edge_types,
            depth,
            max_neighbors,
            incoming=True,
        )

    def _neighbors_direction(
        self,
        record_id: RecordIdentity | str,
        edge_types: list[str] | None,
        depth: int,
        max_neighbors: int | None,
        *,
        incoming: bool,
    ) -> list[GraphNeighbor]:
        if depth < 1:
            raise ValueError("depth must be positive")
        _validate_neighbor_limit(max_neighbors)
        allowed = set(edge_types) if edge_types else None
        identity_key = (
            record_id.storage_key if isinstance(record_id, RecordIdentity) else record_id
        )
        identity_key = self._canonical_graph_storage_key(identity_key)
        frontier = {identity_key}
        best: dict[str, tuple[str, float]] = {}
        source_column = "target_id" if incoming else "source_id"
        target_column = "source_id" if incoming else "target_id"
        with self._access.lock:
            conn = self._access.connection()
            for hop in range(depth):
                if not frontier:
                    break
                placeholders = ",".join("?" for _ in frontier)
                rows = conn.execute(
                    f"""
                    SELECT e.source_id, e.target_id, e.edge_type, e.weight
                    FROM local_graph_edges e
                    JOIN local_records target_record
                        ON target_record.storage_key = e.{target_column}
                    WHERE e.{source_column} IN ({placeholders})
                    """,
                    tuple(frontier),
                ).fetchall()
                next_frontier: set[str] = set()
                for row in rows:
                    if allowed is not None and row["edge_type"] not in allowed:
                        continue
                    try:
                        target_id = self._canonical_graph_storage_key(
                            row[target_column]
                        )
                    except ValueError:
                        continue
                    if row[target_column] == identity_key:
                        continue
                    weight = float(row["weight"])
                    if hop:
                        parent_weight = best.get(
                            row[source_column],
                            ("", 1.0),
                        )[1]
                        weight *= parent_weight
                    current = best.get(target_id)
                    if current is None or weight > current[1]:
                        best[target_id] = (row["edge_type"], weight)
                    next_frontier.add(target_id)
                frontier = next_frontier
        return _sort_graph_neighbors(
            (
                GraphNeighbor(
                    self._identity_from_storage_key(target_id),
                    edge_type,
                    weight,
                )
                for target_id, (edge_type, weight) in best.items()
            ),
            max_neighbors,
        )

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
        """Retrieve neighbors for multiple seeds with one query per hop."""
        if depth < 1:
            raise ValueError("depth must be positive")
        _validate_neighbor_limit(max_neighbors)
        seed_keys = list(dict.fromkeys(identity.storage_key for identity in identities))
        frontiers = {seed_key: {seed_key} for seed_key in seed_keys}
        best_by_seed: dict[str, dict[str, tuple[str, float]]] = {
            seed_key: {} for seed_key in seed_keys
        }
        source_column = "target_id" if incoming else "source_id"
        target_column = "source_id" if incoming else "target_id"
        with self._access.lock:
            conn = self._access.connection()
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
                        ON target_record.storage_key = e.{target_column}
                    WHERE e.{source_column} IN ({placeholders})
                    """,
                    tuple(owners),
                ).fetchall()
                next_frontiers = {seed_key: set() for seed_key in seed_keys}
                for row in rows:
                    try:
                        target_id = self._canonical_graph_storage_key(
                            row[target_column]
                        )
                    except ValueError:
                        continue
                    for seed_key in owners.get(row[source_column], ()):
                        if target_id == seed_key:
                            continue
                        if hop:
                            parent_weight = best_by_seed[seed_key].get(
                                row[source_column], ("", 1.0)
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
            seed_key: _sort_graph_neighbors(
                (
                    GraphNeighbor(
                        self._identity_from_storage_key(target_id),
                        edge_type,
                        weight,
                    )
                    for target_id, (edge_type, weight) in best.items()
                ),
                max_neighbors,
            )
            for seed_key, best in best_by_seed.items()
        }

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
            if not isinstance(row[2], str) or not row[2] or not math.isfinite(row[3]):
                raise ValueError("graph edges require a finite weight and edge type")
            rows.append((source_id, target_id, row[2], row[3]))
        if not rows:
            return
        with self._access.lock:
            conn = self._access.connection()
            try:
                endpoint_keys = sorted(
                    {source_id for source_id, _, _, _ in rows}
                    | {target_id for _, target_id, _, _ in rows}
                )
                existing_keys: set[str] = set()
                for key_chunk in LocalRecordBackend._key_chunks(endpoint_keys):
                    placeholders = ",".join("?" for _ in key_chunk)
                    existing_keys.update(
                        row[0]
                        for row in conn.execute(
                            f"""
                            SELECT storage_key
                            FROM local_records
                            WHERE storage_key IN ({placeholders})
                            """,
                            key_chunk,
                        ).fetchall()
                    )
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
                    _LocalEpochLane.bump(conn, graph=True)
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
            if not isinstance(row[2], str) or not row[2]:
                raise ValueError("graph edges require a non-empty edge type")
            rows.append(
                (
                    self._canonical_graph_storage_key(row[0]),
                    self._canonical_graph_storage_key(row[1]),
                    row[2],
                )
            )
        if not rows:
            return
        with self._access.lock:
            conn = self._access.connection()
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
                    _LocalEpochLane.bump(conn, graph=True)
                conn.commit()
            except Exception:
                conn.rollback()
                raise


class _VectorEngine:
    """Own vector snapshot construction and storage statistics."""

    def __init__(
        self,
        access: _SQLiteAccess,
        *,
        snapshot_lock: threading.RLock,
        vector_snapshots: dict[tuple[str, int], VectorSnapshot],
        vector_storage_stats: dict[tuple[str, int], tuple[int, int, int]],
        vector_snapshot_max_rows: int,
        vector_snapshot_max_bytes: int,
        matches: Callable[[sqlite3.Row, SearchFilters | None], bool],
        status_values: Callable[[SearchFilters | None], set[str]],
        filter_values: Callable[[Any], list[Any]],
    ) -> None:
        self._access = access
        self._snapshot_lock = snapshot_lock
        self._vector_snapshots = vector_snapshots
        self._vector_storage_stats = vector_storage_stats
        self._vector_snapshot_max_rows = vector_snapshot_max_rows
        self._vector_snapshot_max_bytes = vector_snapshot_max_bytes
        self._matches = matches
        self._status_values = status_values
        self._filter_values = filter_values

    def vector_count(self, model_name: str, dim: int) -> int:
        row_count, _ = self.vector_storage_stats(model_name, dim)
        return row_count

    def vector_storage_stats(self, model_name: str, dim: int) -> tuple[int, int]:
        key = (model_name, dim)
        with self._access.lock:
            conn = self._access.connection()
            vector_epoch = _LocalEpochLane.read(
                conn, _LocalEpochLane._LANE_KEYS["vector"]
            )
            cached = self._vector_storage_stats.get(key)
            if cached is not None and cached[0] == vector_epoch:
                return cached[1], cached[2]
            row = conn.execute(
                """
                SELECT COUNT(*), COALESCE(SUM(length(embedding)), 0)
                FROM local_vectors_v2
                WHERE encoder_namespace = ? AND dim = ?
                """,
                (model_name, dim),
            ).fetchone()
            stats = (int(row[0]), int(row[1])) if row else (0, 0)
            self._vector_storage_stats[key] = (vector_epoch, *stats)
            return stats

    def _vector_batch_limit(self, dim: int) -> int:
        bytes_per_vector = dim * _VECTOR_EMBEDDING_BYTES
        if bytes_per_vector < 1:
            raise ValueError("vector dimension must be positive")
        return max(
            1,
            min(
                self._vector_snapshot_max_rows,
                self._vector_snapshot_max_bytes // bytes_per_vector,
            ),
        )

    def _get_vector_snapshot(self, model_name: str, dim: int) -> VectorSnapshot:
        key = (model_name, dim)
        with self._snapshot_lock:
            with self._access.lock:
                conn = self._access.connection()
                current_epoch = _LocalEpochLane.read(
                    conn, _LocalEpochLane._LANE_KEYS["vector"]
                )
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
                snapshot_epoch = _LocalEpochLane.read(
                    conn, _LocalEpochLane._LANE_KEYS["vector"]
                )
            snapshot = VectorSnapshot.from_rows(
                rows,
                encoder_namespace=model_name,
                dim=dim,
                epoch=snapshot_epoch,
            )
            self._vector_snapshots[key] = snapshot
            self._vector_storage_stats[key] = (
                snapshot_epoch,
                len(rows),
                sum(len(row["embedding"]) for row in rows),
            )
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

    def search_vector(
        self,
        query_vector: Vector,
        k: int,
        *,
        model_name: str,
        dim: int,
        filters: SearchFilters | None = None,
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
        row_count, byte_count = self.vector_storage_stats(model_name, dim)
        if (
            row_count > self._vector_snapshot_max_rows
            or byte_count > self._vector_snapshot_max_bytes
        ):
            return self._search_vector_blocks(
                query,
                k,
                model_name=model_name,
                dim=dim,
                filters=filters,
            )
        snapshot = self._get_vector_snapshot(model_name, dim)
        eligible = snapshot.filter_mask(
            dict(filters) if filters is not None else None,
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
        filters: SearchFilters | None,
    ) -> list[RecordHit]:
        best_keys: list[str] = []
        best_scores: list[float] = []
        last_storage_key: str | None = None
        batch_limit = self._vector_batch_limit(dim)
        while True:
            with self._access.lock:
                conn = self._access.connection()
                if last_storage_key is None:
                    rows = conn.execute(
                        """
                        SELECT r.storage_key, r.workspace_id, r.source_kind, r.source_id,
                               r.status, r.metadata, r.uri, v.embedding, v.format_version,
                               v.normalization_policy
                        FROM local_records r
                        JOIN local_vectors_v2 v ON v.storage_key = r.storage_key
                        WHERE v.encoder_namespace = ? AND v.dim = ?
                        ORDER BY r.storage_key
                        LIMIT ?
                        """,
                        (model_name, dim, batch_limit),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """
                        SELECT r.storage_key, r.workspace_id, r.source_kind, r.source_id,
                               r.status, r.metadata, r.uri, v.embedding, v.format_version,
                               v.normalization_policy
                        FROM local_records r
                        JOIN local_vectors_v2 v ON v.storage_key = r.storage_key
                        WHERE v.encoder_namespace = ? AND v.dim = ?
                          AND r.storage_key > ?
                        ORDER BY r.storage_key
                        LIMIT ?
                        """,
                        (model_name, dim, last_storage_key, batch_limit),
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
            last_storage_key = rows[-1]["storage_key"]
        ordered = sorted(
            zip(best_keys, best_scores, strict=True),
            key=lambda item: (-item[1], item[0]),
        )
        return [
            RecordHit(RecordIdentity.from_storage_key(storage_key), score)
            for storage_key, score in ordered[:k]
        ]

    def _iter_vector_batches(
        self,
        model_name: str,
        dim: int,
    ) -> Iterable[list[sqlite3.Row]]:
        """Yield bounded vector rows for streamed optional-index builds."""
        last_storage_key: str | None = None
        batch_limit = self._vector_batch_limit(dim)
        while True:
            with self._access.lock:
                conn = self._access.connection()
                if last_storage_key is None:
                    rows = conn.execute(
                        """
                        SELECT r.storage_key, r.workspace_id, r.source_kind, r.source_id,
                               r.status, r.metadata, r.uri, v.embedding, v.format_version,
                               v.normalization_policy
                        FROM local_records r
                        JOIN local_vectors_v2 v ON v.storage_key = r.storage_key
                        WHERE v.encoder_namespace = ? AND v.dim = ?
                        ORDER BY r.storage_key
                        LIMIT ?
                        """,
                        (model_name, dim, batch_limit),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """
                        SELECT r.storage_key, r.workspace_id, r.source_kind, r.source_id,
                               r.status, r.metadata, r.uri, v.embedding, v.format_version,
                               v.normalization_policy
                        FROM local_records r
                        JOIN local_vectors_v2 v ON v.storage_key = r.storage_key
                        WHERE v.encoder_namespace = ? AND v.dim = ?
                          AND r.storage_key > ?
                        ORDER BY r.storage_key
                        LIMIT ?
                        """,
                        (model_name, dim, last_storage_key, batch_limit),
                    ).fetchall()
            if not rows:
                return
            yield rows
            last_storage_key = rows[-1]["storage_key"]


class _KeywordEngine:
    """Own keyword filter-SQL construction for the local backend."""

    def __init__(
        self,
        access: _SQLiteAccess,
        *,
        status_values: Callable[[SearchFilters | None], set[str]],
        filter_values: Callable[[Any], list[Any]],
        matches: Callable[[sqlite3.Row, SearchFilters | None], bool],
        metadata_keyword_text: Callable[[dict[str, Any]], str],
        metadata_uri: Callable[[dict[str, Any]], str],
        keyword_overfetch_multiplier: float,
        artifact_scorer: KeywordArtifactScorer,
    ) -> None:
        self._access = access
        self._status_values = status_values
        self._filter_values = filter_values
        self._matches = matches
        self._metadata_keyword_text = metadata_keyword_text
        self._metadata_uri = metadata_uri
        self._keyword_overfetch_multiplier = keyword_overfetch_multiplier
        self._artifact_scorer = artifact_scorer
        self._fts5_available = False
        self._keyword_search_diagnostic = (
            "FTS5 indexed lexical search has not been initialized"
        )
        self._last_keyword_search_diagnostics: Mapping[str, int | bool] = (
            MappingProxyType(
                {
                    "scanned": 0,
                    "requested_k": 0,
                    "returned": 0,
                    "scan_complete": True,
                    "fallback": False,
                }
            )
        )

    def _natural_language_match_query(
        self,
        query: str,
        match_query: str,
    ) -> str | None:
        if '"' in query or self._artifact_scorer.looks_like_identifier_query(query):
            return None
        terms = match_query.split()
        if len(terms) < 2:
            return None
        return " OR ".join(terms)

    def _keyword_filter_sql(
        self,
        filters: SearchFilters | None,
    ) -> tuple[list[str], list[Any]]:
        filters = filters or {}
        statuses = sorted(self._status_values(filters))
        clauses: list[str] = []
        parameters: list[Any] = []
        self._append_keyword_in_filter(
            clauses, parameters, "r.status", statuses
        )

        workspace_id = filters.get("workspace_id")
        if workspace_id is not None:
            clauses.append("r.workspace_id = ?")
            parameters.append(workspace_id)

        source_kinds = filters.get("source_kinds")
        if source_kinds is None and filters.get("source_kind") is not None:
            source_kinds = [filters["source_kind"]]
        if source_kinds is None and filters.get("source_filter") is not None:
            source_kinds = filters["source_filter"]
        if source_kinds is not None:
            source_kinds = self._filter_values(source_kinds)
            if not source_kinds:
                return ["0"], []
            self._append_keyword_in_filter(
                clauses, parameters, "r.source_kind", source_kinds
            )

        for source_filter in compile_source_scoped_filters(filters):
            if source_filter.workspace_ids is not None:
                if not source_filter.workspace_ids:
                    clauses.append("r.source_kind <> ?")
                    parameters.append(source_filter.source_kind)
                else:
                    placeholders = ", ".join("?" for _ in source_filter.workspace_ids)
                    clauses.append(
                        f"(r.source_kind <> ? OR r.workspace_id IN ({placeholders}))"
                    )
                    parameters.extend(
                        [source_filter.source_kind, *source_filter.workspace_ids]
                    )
            for field in source_filter.metadata_non_empty:
                path = f"$.{field}"
                clauses.append(
                    "(r.source_kind <> ? OR ("
                    "json_type(json_extract(r.metadata, ?)) = 'array' AND "
                    "json_array_length(json_extract(r.metadata, ?)) > 0))"
                )
                parameters.extend([source_filter.source_kind, path, path])
            for field, allowed_values in source_filter.metadata_contains_any:
                path = f"$.{field}"
                if not allowed_values:
                    clauses.append("r.source_kind <> ?")
                    parameters.append(source_filter.source_kind)
                    continue
                placeholders = ", ".join("?" for _ in allowed_values)
                clauses.append(
                    "(r.source_kind <> ? OR ("
                    "json_type(json_extract(r.metadata, ?)) = 'array' AND "
                    "EXISTS (SELECT 1 FROM json_each(r.metadata, ?) AS item "
                    f"WHERE item.type = 'text' AND item.value IN ({placeholders}))" "))"
                )
                parameters.extend(
                    [source_filter.source_kind, path, path, *allowed_values]
                )

        project_values = filters.get("project_ids")
        if project_values is None:
            project_values = filters.get("project_id")
        if project_values is None:
            project_values = filters.get("project_filter")
        if project_values is not None:
            project_values = self._filter_values(project_values)
            if not project_values and (
                "project_ids" in filters or "project_id" in filters
            ):
                return ["0"], []
            if project_values:
                self._append_keyword_in_filter(
                    clauses,
                    parameters,
                    "CAST(json_extract(r.metadata, '$.project_id') AS TEXT)",
                    [str(value) for value in project_values],
                )

        excluded_projects = filters.get("excluded_projects")
        if excluded_projects is None:
            excluded_projects = filters.get("excluded_project_ids")
        if excluded_projects:
            excluded_projects = self._filter_values(excluded_projects)
            clauses.append(
                "(json_extract(r.metadata, '$.project_id') IS NULL OR "
                "CAST(json_extract(r.metadata, '$.project_id') AS TEXT) "
                "NOT IN ({}))".format(", ".join("?" for _ in excluded_projects))
            )
            parameters.extend(str(value) for value in excluded_projects)

        candidate_keys = filters.get("candidate_ids")
        if candidate_keys is None:
            candidate_keys = filters.get("candidate_storage_keys")
        if candidate_keys is not None:
            candidate_keys = sorted(candidate_storage_keys(candidate_keys))
            if not candidate_keys:
                return ["0"], []
            self._append_keyword_in_filter(
                clauses, parameters, "r.storage_key", candidate_keys
            )

        path_expression = (
            "COALESCE(NULLIF(json_extract(r.metadata, '$.file_path'), ''), "
            "NULLIF(json_extract(r.metadata, '$.path'), ''), "
            "NULLIF(json_extract(r.metadata, '$.source_file'), ''), "
            "NULLIF(r.uri, ''))"
        )
        included_paths = filters.get("paths")
        if included_paths is None:
            included_paths = filters.get("file_paths")
        if included_paths is None:
            included_paths = filters.get("source_files")
        if included_paths is None:
            included_paths = filters.get("path")
        if included_paths is None:
            included_paths = filters.get("file_path")
        if included_paths is None:
            included_paths = filters.get("source_file")
        self._append_keyword_path_filter(
            clauses,
            parameters,
            path_expression,
            included_paths,
            exclude=False,
        )

        excluded_paths = filters.get("excluded_files")
        if excluded_paths is None:
            excluded_paths = filters.get("excluded_paths")
        if excluded_paths is None:
            excluded_paths = filters.get("excluded_file_paths")
        if excluded_paths is None:
            excluded_paths = filters.get("excluded_source_files")
        self._append_keyword_path_filter(
            clauses,
            parameters,
            path_expression,
            excluded_paths,
            exclude=True,
        )

        document_expression = (
            "COALESCE(NULLIF(json_extract(r.metadata, '$.doc_id'), ''), "
            "r.source_id)"
        )
        document_values = filters.get("document_ids")
        if document_values is None:
            document_values = filters.get("document_id")
        if document_values is None:
            document_values = filters.get("doc_ids")
        if document_values is None:
            document_values = filters.get("doc_id")
        if document_values is not None:
            values = self._filter_values(document_values)
            if not values:
                return ["0"], []
            self._append_keyword_in_filter(
                clauses, parameters, document_expression, [str(value) for value in values]
            )

        excluded_documents = filters.get("excluded_documents")
        if excluded_documents is None:
            excluded_documents = filters.get("excluded_document_ids")
        if excluded_documents is None:
            excluded_documents = filters.get("excluded_doc_ids")
        if excluded_documents:
            values = self._filter_values(excluded_documents)
            placeholders = ", ".join("?" for _ in values)
            clauses.append(
                f"({document_expression} NOT IN ({placeholders}) "
                f"AND r.source_id NOT IN ({placeholders}))"
            )
            parameters.extend(str(value) for value in values)
            parameters.extend(str(value) for value in values)

        metadata_equals = filters.get("metadata_equals")
        if metadata_equals is not None:
            for field, value in metadata_equals.items():
                if value is None:
                    continue
                if not _METADATA_FIELD_RE.fullmatch(field):
                    raise ValueError(f"metadata_equals field {field!r} is invalid")
                clauses.append("json_extract(r.metadata, ?) = ?")
                parameters.extend((f"$.{field}", str(value)))
        return clauses, parameters

    @staticmethod
    def _append_keyword_in_filter(
        clauses: list[str],
        parameters: list[Any],
        column: str,
        values: Sequence[Any],
    ) -> None:
        clauses.append(f"{column} IN ({', '.join('?' for _ in values)})")
        parameters.extend(values)

    def _append_keyword_path_filter(
        self,
        clauses: list[str],
        parameters: list[Any],
        expression: str,
        values: Any,
        *,
        exclude: bool,
    ) -> None:
        if values is None:
            return
        normalized = {
            str(value).replace("\\", "/") for value in self._filter_values(values)
        }
        variants = sorted(
            {
                variant
                for value in normalized
                for variant in self._keyword_path_variants(value)
            }
        )
        if not variants:
            if not exclude:
                clauses.append("0")
            return
        comparisons: list[str] = []
        for value in variants:
            comparisons.append(f"{expression} = ?")
            parameters.append(value)
            if "/" not in value:
                escaped = value.replace("\\", "\\\\").replace("%", "\\%").replace(
                    "_", "\\_"
                )
                comparisons.append(f"{expression} LIKE ? ESCAPE '\\'")
                parameters.append(f"%/{escaped}")
        predicate = " OR ".join(comparisons)
        clauses.append(f"NOT ({predicate})" if exclude else f"({predicate})")

    @staticmethod
    def _keyword_path_variants(value: str) -> set[str]:
        variants = {value}
        leaf = value.rsplit("/", 1)[-1]
        variants.add(leaf)
        if "." in leaf:
            variants.add(value.rsplit(".", 1)[0])
            variants.add(leaf.rsplit(".", 1)[0])
        return variants

    def search_keyword(
        self,
        query: str,
        k: int,
        filters: SearchFilters | None = None,
    ) -> list[RecordHit]:
        if k < 1 or not query.strip():
            self._last_keyword_search_diagnostics = MappingProxyType(
                {
                    "scanned": 0,
                    "requested_k": k,
                    "returned": 0,
                    "scan_complete": True,
                    "fallback": False,
                }
            )
            return []
        if self._fts5_available:
            hits = self._search_keyword_fts(query, k, filters)
            self._last_keyword_search_diagnostics = MappingProxyType(
                {
                    "scanned": 0,
                    "requested_k": k,
                    "returned": len(hits),
                    "scan_complete": True,
                    "fallback": False,
                }
            )
            return hits
        return self._search_keyword_fallback(query, k, filters)

    def _search_keyword_fts(
        self,
        query: str,
        k: int,
        filters: SearchFilters | None,
    ) -> list[RecordHit]:
        match_query = _keyword_scoring.sanitize_fts_query(query)
        if match_query == '""':
            return []
        needs_artifact_rerank = self._artifact_scorer.looks_like_identifier_query(query)
        artifact_tokens = self._artifact_scorer.identifier_tokens(query)
        if needs_artifact_rerank and artifact_tokens:
            artifact_queries = [
                _keyword_scoring.sanitize_fts_query(token)
                for token in artifact_tokens
            ]
            match_query = " OR ".join(
                f"({artifact_query})"
                for artifact_query in artifact_queries
                if artifact_query != '""'
            )
        if needs_artifact_rerank and len(query.strip().split()) == 1:
            match_query = '"' + match_query.replace('"', "") + '"'
        limit = k
        if needs_artifact_rerank:
            limit = max(k, math.ceil(k * self._keyword_overfetch_multiplier))
        if filters and not set(filters).issubset(_KEYWORD_SQL_FILTERS):
            limit = min(limit * _FILTERED_KEYWORD_OVERFETCH, _FALLBACK_SCAN_MAX_ROWS)
        def fetch_rows(current_query: str) -> list[sqlite3.Row]:
            clauses, parameters = self._keyword_filter_sql(filters)
            clauses.insert(0, f"{_LOCAL_FTS_TABLE} MATCH ?")
            parameters.insert(0, current_query)
            with self._access.lock:
                conn = self._access.connection()
                return conn.execute(
                    f"""
                    SELECT
                        r.storage_key,
                        r.workspace_id,
                        r.source_kind,
                        r.source_id,
                        r.status,
                        r.title,
                        r.body,
                        r.indexed_text,
                        r.uri,
                        r.keywords,
                        r.metadata,
                        -bm25({_LOCAL_FTS_TABLE}, 5.0, 1.0, 4.0, 2.0) AS score
                    FROM {_LOCAL_FTS_TABLE}
                    JOIN local_records r ON r.rowid = {_LOCAL_FTS_TABLE}.rowid
                    WHERE {" AND ".join(clauses)}
                    ORDER BY score DESC, r.storage_key ASC
                    LIMIT ?
                    """,
                    (*parameters, limit),
                ).fetchall()

        def build_hits(rows: Sequence[sqlite3.Row]) -> list[RecordHit]:
            hits: list[RecordHit] = []
            for row in rows:
                if filters and not self._matches(row, filters):
                    continue
                score = float(row["score"])
                if needs_artifact_rerank:
                    score += self._artifact_scorer.score(
                        query,
                        title=row["title"],
                        body=row["body"],
                        indexed_text=row["indexed_text"],
                        headers=row["keywords"],
                        uri=row["uri"] or "",
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
            return hits

        hits = build_hits(fetch_rows(match_query))
        fallback_query = self._natural_language_match_query(query, match_query)
        if not hits and fallback_query is not None:
            hits = build_hits(fetch_rows(fallback_query))
        if not hits:
            hits = self._search_keyword_fuzzy(query, k, filters)
        return hits[:k]

    def _search_keyword_fuzzy(
        self,
        query: str,
        k: int,
        filters: SearchFilters | None,
    ) -> list[RecordHit]:
        if (
            '"' in query
            or self._artifact_scorer.looks_like_identifier_query(query)
        ):
            return []
        terms = [term.casefold() for term in _TOKEN_RE.findall(query)]
        if (
            not 2 <= len(terms) <= _FUZZY_QUERY_MAX_TERMS
            or not all(len(term) >= 4 for term in terms)
        ):
            return []
        prefix_query = " OR ".join(
            f"{term[:3]}*" for term in terms
        )
        clauses, parameters = self._keyword_filter_sql(filters)
        clauses.insert(0, f"{_LOCAL_FTS_TABLE} MATCH ?")
        parameters.insert(0, prefix_query)
        with self._access.lock:
            conn = self._access.connection()
            rows = conn.execute(
                f"""
                SELECT r.storage_key, r.workspace_id, r.source_kind, r.source_id,
                       r.status, r.title, r.body, r.indexed_text, r.uri,
                       r.keywords, r.metadata
                FROM {_LOCAL_FTS_TABLE}
                JOIN local_records r ON r.rowid = {_LOCAL_FTS_TABLE}.rowid
                WHERE {" AND ".join(clauses)}
                ORDER BY bm25({_LOCAL_FTS_TABLE}, 5.0, 1.0, 4.0, 2.0),
                         r.storage_key ASC
                LIMIT ?
                """,
                (*parameters, _FALLBACK_SCAN_MAX_ROWS),
            ).fetchall()
        hits: list[RecordHit] = []
        for row in rows:
            if filters and not self._matches(row, filters):
                continue
            text = " ".join(
                (
                    row["title"] or "",
                    row["indexed_text"] or row["body"] or "",
                    row["uri"] or "",
                    row["keywords"] or "",
                )
            )
            tokens = list(dict.fromkeys(
                _TOKEN_RE.findall(text.casefold())
            ))
            scores = [_fuzzy_term_score(term, tokens) for term in terms]
            if all(score >= _FUZZY_TERM_RATIO for score in scores):
                hits.append(
                    RecordHit(
                        RecordIdentity(
                            row["workspace_id"],
                            row["source_kind"],
                            row["source_id"],
                        ),
                        sum(scores),
                    )
                )
        hits.sort(key=lambda item: (-item.score, item.storage_key))
        return hits[:k]

    def _search_keyword_fallback(
        self,
        query: str,
        k: int,
        filters: SearchFilters | None,
    ) -> list[RecordHit]:
        terms = [term.casefold() for term in _TOKEN_RE.findall(query)]
        if not terms:
            self._last_keyword_search_diagnostics = MappingProxyType(
                {
                    "scanned": 0,
                    "requested_k": k,
                    "returned": 0,
                    "scan_complete": True,
                    "fallback": True,
                }
            )
            return []
        clauses, parameters = self._keyword_filter_sql(filters)
        heap: list[_FallbackHeapItem] = []
        last_rowid = 0
        scanned_rows = 0
        scan_complete = False
        while True:
            with self._access.lock:
                conn = self._access.connection()
                rows = conn.execute(
                    f"""
                    SELECT r.rowid, storage_key, workspace_id, source_kind,
                           source_id, status, title, body, indexed_text, uri,
                           keywords, metadata
                    FROM local_records r
                    WHERE r.rowid > ?
                      AND {" AND ".join(clause.replace("r.", "") for clause in clauses)}
                    ORDER BY r.rowid ASC
                    LIMIT ?
                    """,
                    (last_rowid, *parameters, _FALLBACK_SCAN_BATCH_SIZE),
                ).fetchall()
            if not rows:
                scan_complete = True
                break
            scanned_rows += len(rows)
            last_rowid = int(rows[-1]["rowid"])
            for row in rows:
                if filters and not self._matches(row, filters):
                    continue
                indexed_text = row["indexed_text"] or row["body"]
                haystack = " ".join(
                    (
                        row["title"] or "",
                        indexed_text or "",
                        row["uri"] or "",
                        row["keywords"] or "",
                    )
                ).lower()
                score = sum(haystack.count(term) for term in terms)
                if not score:
                    continue
                hit = RecordHit(
                    RecordIdentity(
                        row["workspace_id"],
                        row["source_kind"],
                        row["source_id"],
                    ),
                    float(score),
                )
                candidate = _FallbackHeapItem(hit)
                if len(heap) < k:
                    heapq.heappush(heap, candidate)
                elif heap[0] < candidate:
                    heapq.heapreplace(heap, candidate)
            if len(rows) < _FALLBACK_SCAN_BATCH_SIZE:
                scan_complete = True
                break
        hits = sorted(
            (candidate.hit for candidate in heap),
            key=lambda item: (-item.score, item.storage_key),
        )
        self._last_keyword_search_diagnostics = MappingProxyType(
            {
                "scanned": scanned_rows,
                "requested_k": k,
                "returned": len(hits),
                "scan_complete": scan_complete,
                "fallback": True,
            }
        )
        return hits

    def _migrate_keyword_columns(self, conn: sqlite3.Connection) -> None:
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
            keywords = self._metadata_keyword_text(metadata)
            uri = row["uri"] or self._metadata_uri(metadata)
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

    def initialize_schema(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS local_keyword_schema (
                name TEXT PRIMARY KEY,
                version INTEGER NOT NULL
            )
            """
        )
        version_row = conn.execute(
            "SELECT version FROM local_keyword_schema WHERE name = ?",
            (_LOCAL_KEYWORD_SCHEMA,),
        ).fetchone()
        needs_keyword_migration = (
            version_row is None
            or version_row[0] != _LOCAL_KEYWORD_SCHEMA_VERSION
        )
        if needs_keyword_migration:
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

        if self._fts5_available and (
            needs_rebuild
            or version_row is None
            or version_row[0] != _LOCAL_KEYWORD_SCHEMA_VERSION
        ):
            try:
                conn.execute(f"DELETE FROM {_LOCAL_FTS_TABLE}")
            except sqlite3.DatabaseError:
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
            rows = conn.execute(
                """
                SELECT rowid, title, body, indexed_text, uri, keywords
                FROM local_records
                """
            ).fetchall()
            conn.executemany(
                f"""
                INSERT INTO {_LOCAL_FTS_TABLE}
                    (rowid, title, body, uri, keywords)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    (
                        row["rowid"],
                        row["title"],
                        row["indexed_text"] or row["body"],
                        row["uri"],
                        row["keywords"],
                    )
                    for row in rows
                ),
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
        with self._access.lock:
            conn = self._access.connection()
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
        """Rebuild the keyword index from effective indexed text."""
        with self._access.lock:
            conn = self._access.connection()
            if not self._fts5_available:
                self._keyword_search_diagnostic = (
                    "FTS5 indexed lexical search is unavailable; "
                    "the keyword index cannot be rebuilt"
                )
                return
            try:
                conn.execute(f"DELETE FROM {_LOCAL_FTS_TABLE}")
                rows = conn.execute(
                    """
                    SELECT rowid, title, body, indexed_text, uri, keywords
                    FROM local_records
                    """
                ).fetchall()
                conn.executemany(
                    f"""
                    INSERT INTO {_LOCAL_FTS_TABLE}
                        (rowid, title, body, uri, keywords)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        (
                            row["rowid"],
                            row["title"],
                            row["indexed_text"] or row["body"],
                            row["uri"],
                            row["keywords"],
                        )
                        for row in rows
                    ),
                )
                _LocalEpochLane.bump(conn, keyword=True)
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
                rows = conn.execute(
                    """
                    SELECT rowid, title, body, indexed_text, uri, keywords
                    FROM local_records
                    """
                ).fetchall()
                conn.executemany(
                    f"""
                    INSERT INTO {_LOCAL_FTS_TABLE}
                        (rowid, title, body, uri, keywords)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        (
                            row["rowid"],
                            row["title"],
                            row["indexed_text"] or row["body"],
                            row["uri"],
                            row["keywords"],
                        )
                        for row in rows
                    ),
                )
                _LocalEpochLane.bump(conn, keyword=True)
                conn.commit()


class _RecordWriter:
    """Own record and chunk persistence, including FTS row synchronization."""

    def __init__(
        self,
        access: _SQLiteAccess,
        *,
        metadata_keyword_text: Callable[[dict[str, Any]], str],
        metadata_uri: Callable[[dict[str, Any]], str],
        fts5_available: Callable[[], bool],
    ) -> None:
        self._access = access
        self._metadata_keyword_text = metadata_keyword_text
        self._metadata_uri = metadata_uri
        self._fts5_available = fts5_available

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
    ) -> None:
        chunk_rows = [
            values
            for record in records
            if (values := self._chunk_state_values(record)) is not None
        ]
        if not chunk_rows:
            return
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
            for key_chunk in LocalRecordBackend._key_chunks(stale_keys):
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

    def _write_records(
        self,
        conn: sqlite3.Connection,
        rows: list[Record],
    ) -> None:
        keys = list(dict.fromkeys(record.storage_key for record in rows))
        old_rows: dict[str, sqlite3.Row] = {}
        for key_chunk in LocalRecordBackend._key_chunks(keys):
            placeholders = ",".join("?" for _ in key_chunk)
            old_rows.update(
                {
                    row["storage_key"]: row
                    for row in conn.execute(
                        f"""
                        SELECT storage_key, rowid, title, body, indexed_text, uri, keywords
                        FROM local_records
                        WHERE storage_key IN ({placeholders})
                        """,
                        key_chunk,
                    ).fetchall()
                }
            )

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
                    for row in old_rows.values()
                ],
            )
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
            [self._record_values(record) for record in rows],
        )

        if self._fts5_available():
            new_rows: list[sqlite3.Row] = []
            for key_chunk in LocalRecordBackend._key_chunks(keys):
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
        self._sync_chunk_state(conn, rows)

    @staticmethod
    def _records_have_vectors(
        conn: sqlite3.Connection, records: Sequence[Record]
    ) -> bool:
        keys = list(dict.fromkeys(record.storage_key for record in records))
        for key_chunk in LocalRecordBackend._key_chunks(keys):
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
                self._write_records(conn, rows)
                _LocalEpochLane.bump(conn, keyword=True, vector=vector_affected)
                conn.commit()
            except Exception:
                conn.rollback()
                raise


class _SchemaManager:
    """Own table creation, column migration, and identity versioning."""

    def __init__(
        self,
        access: _SQLiteAccess,
        *,
        initialize_keyword_schema: Callable[[sqlite3.Connection], None],
        initialize_graph_schema: Callable[[sqlite3.Connection], None],
    ) -> None:
        self._access = access
        self._initialize_keyword_schema = initialize_keyword_schema
        self._initialize_graph_schema = initialize_graph_schema
        self._record_identity_stale = False

    def initialize_schema(self) -> None:
        conn = self._access.connection()
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
                indexed_text TEXT,
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
            CREATE TABLE IF NOT EXISTS local_chunk_state (
                chunk_storage_key TEXT PRIMARY KEY,
                parent_storage_key TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                metadata TEXT NOT NULL,
                FOREIGN KEY (chunk_storage_key) REFERENCES local_records(storage_key)
                    ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_local_chunk_parent
                ON local_chunk_state (parent_storage_key, chunk_index, chunk_id);
            CREATE TABLE IF NOT EXISTS local_vectors_v2 (
                storage_key TEXT NOT NULL,
                encoder_namespace TEXT NOT NULL,
                dim INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                revision TEXT,
                format_version INTEGER NOT NULL,
                normalization_policy TEXT NOT NULL,
                PRIMARY KEY (storage_key, encoder_namespace, dim),
                FOREIGN KEY (storage_key) REFERENCES local_records(storage_key)
                    ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_local_vectors_v2_namespace
                ON local_vectors_v2 (encoder_namespace, dim);
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
        self._ensure_local_vector_column(conn, "revision", "TEXT")
        self._ensure_local_record_column(conn, "keywords", "TEXT NOT NULL DEFAULT ''")
        self._ensure_local_record_column(conn, "indexed_text", "TEXT")
        self._initialize_keyword_schema(conn)
        self._initialize_graph_schema(conn)
        self._record_identity_stale = self._resolve_record_identity_staleness(conn)
        conn.commit()

    def _resolve_record_identity_staleness(self, conn: sqlite3.Connection) -> bool:
        version_row = conn.execute(
            "SELECT value FROM system_state WHERE key = ?",
            (_RECORD_IDENTITY_VERSION_KEY,),
        ).fetchone()
        if version_row is not None:
            # An unreadable marker cannot prove the store is current, and
            # refusing to open it would be worse than reporting it stale.
            try:
                return int(version_row[0]) != _LOCAL_RECORD_IDENTITY_VERSION
            except (TypeError, ValueError):
                return True
        has_records = (
            conn.execute("SELECT 1 FROM local_records LIMIT 1").fetchone() is not None
        )
        if has_records:
            return True
        self._write_record_identity_version(conn)
        return False

    @staticmethod
    def _write_record_identity_version(conn: sqlite3.Connection) -> None:
        conn.execute(
            "INSERT INTO system_state (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (_RECORD_IDENTITY_VERSION_KEY, str(_LOCAL_RECORD_IDENTITY_VERSION)),
        )

    @property
    def record_identity_stale(self) -> bool:
        """Whether stored records were written under an older identity scheme."""
        return self._record_identity_stale

    def mark_record_identity_current(self) -> None:
        """Record that the store has been rebuilt under the current identity scheme."""
        with self._access.lock:
            conn = self._access.connection()
            self._write_record_identity_version(conn)
            conn.commit()
        self._record_identity_stale = False

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
    def _ensure_local_vector_column(
        conn: sqlite3.Connection,
        column: str,
        definition: str,
    ) -> None:
        columns = {
            row[1] for row in conn.execute("PRAGMA table_info(local_vectors_v2)")
        }
        if column not in columns:
            conn.execute(f"ALTER TABLE local_vectors_v2 ADD COLUMN {column} {definition}")


class _HydrationEngine:
    """Own record hydration and chunk-parent expansion for the local backend."""

    def __init__(self, access: _SQLiteAccess) -> None:
        self._access = access

    def _record_rows(self) -> list[sqlite3.Row]:
        conn = self._access.connection()
        conn.row_factory = sqlite3.Row
        return conn.execute("SELECT * FROM local_records").fetchall()

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
        with self._access.lock:
            conn = self._access.connection()
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
                indexed_text=row["indexed_text"],
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
        """Hydrate canonical identities with bounded record queries."""
        keys = list(dict.fromkeys(identity.storage_key for identity in identities))
        if not keys:
            return {}
        with self._access.lock:
            conn = self._access.connection()
            conn.row_factory = sqlite3.Row
            rows: list[sqlite3.Row] = []
            for key_chunk in LocalRecordBackend._key_chunks(keys):
                placeholders = ",".join("?" for _ in key_chunk)
                rows.extend(
                    conn.execute(
                        f"SELECT * FROM local_records WHERE storage_key IN ({placeholders})",
                        key_chunk,
                    ).fetchall()
                )
        records = {
            row["storage_key"]: Record(
                workspace_id=row["workspace_id"],
                source_kind=row["source_kind"],
                source_id=row["source_id"],
                title=row["title"],
                body=row["body"],
                indexed_text=row["indexed_text"],
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
                metadata=json.loads(row["metadata"]),
                uri=row["uri"],
                status=RecordStatus(row["status"]),
            )
            for row in rows
        }
        return {key: records.get(key) for key in keys}

    def chunk_parent(self, record_id: RecordIdentity | str) -> RecordIdentity | None:
        """Return the canonical parent identity for a persisted chunk."""
        storage_key = (
            record_id.storage_key if isinstance(record_id, RecordIdentity) else record_id
        )
        with self._access.lock:
            conn = self._access.connection()
            row = conn.execute(
                """
                SELECT parent_storage_key
                FROM local_chunk_state
                WHERE chunk_storage_key = ?
                """,
                (storage_key,),
            ).fetchone()
        if row is None:
            return None
        return RecordIdentity.from_storage_key(row[0])

    def chunk_records(
        self,
        parent_id: RecordIdentity | str,
    ) -> dict[str, Record]:
        """Hydrate persisted chunks for one canonical parent."""
        parent_key = (
            parent_id.storage_key if isinstance(parent_id, RecordIdentity) else parent_id
        )
        with self._access.lock:
            conn = self._access.connection()
            rows = conn.execute(
                """
                SELECT chunk_storage_key
                FROM local_chunk_state
                WHERE parent_storage_key = ?
                ORDER BY chunk_index, chunk_id
                """,
                (parent_key,),
            ).fetchall()
        identities = [RecordIdentity.from_storage_key(row[0]) for row in rows]
        hydrated = self.hydrate_records(identities)
        return {
            key: record
            for key, record in hydrated.items()
            if record is not None
        }


class LocalRecordBackend:
    """Shared durable state for the local vector, keyword, and graph stores."""

    def __init__(
        self,
        db_path: Path | None = None,
        *,
        db_manager: SQLiteDatabase | None = None,
        sqlite_tuning: SQLiteTuning | None = None,
        keyword_overfetch_multiplier: float = 4.0,
        keyword_artifact_scorer: KeywordArtifactScorer | None = None,
        vector_engine: str = "exact",
        faiss_threshold: int = 50_000,
        vector_snapshot_max_rows: int = 100_000,
        vector_snapshot_max_bytes: int = _DEFAULT_VECTOR_SNAPSHOT_MAX_BYTES,
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
        if vector_snapshot_max_bytes < 1:
            raise ValueError("vector_snapshot_max_bytes must be positive")
        # An injected manager belongs to the caller; only managers created by
        # this backend may be closed by the backend.
        self._owns_database = db_manager is None
        self._db = db_manager or (
            DatabaseManager(db_path, tuning=sqlite_tuning)
            if db_path is not None
            else InMemorySQLiteDatabase(sqlite_tuning)
        )
        self._lock = threading.RLock()
        self._access = _SQLiteAccess(db=self._db, lock=self._lock)
        self._graph_engine = _GraphEngine(self._access)
        self._snapshot_lock = threading.RLock()
        self._vector_storage_stats: dict[tuple[str, int], tuple[int, int, int]] = {}
        self._vector_snapshots: dict[tuple[str, int], VectorSnapshot] = (
            _VectorSnapshotCache(self._vector_storage_stats)
        )
        self._epoch_lane = _LocalEpochLane()
        self._keyword_overfetch_multiplier = keyword_overfetch_multiplier
        self._keyword_artifact_scorer = keyword_artifact_scorer or FilesystemArtifactScorer()
        self._vector_engine = vector_engine
        self._faiss_threshold = faiss_threshold
        self._vector_snapshot_max_rows = vector_snapshot_max_rows
        self._vector_snapshot_max_bytes = vector_snapshot_max_bytes
        self._vector_snapshot_engine = _VectorEngine(
            self._access,
            snapshot_lock=self._snapshot_lock,
            vector_snapshots=self._vector_snapshots,
            vector_storage_stats=self._vector_storage_stats,
            vector_snapshot_max_rows=self._vector_snapshot_max_rows,
            vector_snapshot_max_bytes=self._vector_snapshot_max_bytes,
            matches=self._matches,
            status_values=self._status_values,
            filter_values=self._filter_values,
        )
        self._keyword_engine = _KeywordEngine(
            self._access,
            status_values=self._status_values,
            filter_values=self._filter_values,
            matches=self._matches,
            metadata_keyword_text=self._metadata_keyword_text,
            metadata_uri=self._metadata_uri,
            keyword_overfetch_multiplier=self._keyword_overfetch_multiplier,
            artifact_scorer=self._keyword_artifact_scorer,
        )
        self._record_writer = _RecordWriter(
            self._access,
            metadata_keyword_text=self._metadata_keyword_text,
            metadata_uri=self._metadata_uri,
            fts5_available=lambda: self._keyword_engine._fts5_available,
        )
        self._schema_manager = _SchemaManager(
            self._access,
            initialize_keyword_schema=self._keyword_engine.initialize_schema,
            initialize_graph_schema=self._graph_engine.initialize_schema,
        )
        self._hydration_engine = _HydrationEngine(self._access)
        self._schema_manager.initialize_schema()

    @property
    def db_manager(self) -> SQLiteDatabase:
        return self._db

    def close(self) -> None:
        """Close the database created by this backend, if it owns one."""
        if self._owns_database:
            self._db.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    @property
    def record_identity_stale(self) -> bool:
        """Whether stored records were written under an older identity scheme."""
        return self._schema_manager.record_identity_stale

    def mark_record_identity_current(self) -> None:
        """Record that the store has been rebuilt under the current identity scheme."""
        self._schema_manager.mark_record_identity_current()

    @staticmethod
    def _metadata_keyword_text(metadata: dict[str, Any]) -> str:
        values: list[str] = []
        for key in (
            "tags",
            "keywords",
            "source_keywords",
            "aliases",
            "header_path",
            "headers",
            "file_path",
            "source_file",
            "path",
            "filename",
            "file_name",
            "paths",
            "filenames",
            "files_changed",
            "symbols",
            "symbol",
            "symbol_path",
            "symbol_paths",
            "file_tokens",
            "commit_file_tokens",
            "path_tokens",
            "symbol_tokens",
            "tokens",
            "exact_tokens",
        ):
            value = metadata.get(key)
            if value is None:
                continue
            values.extend(LocalRecordBackend._metadata_keyword_values(value))
        return " ".join(" ".join(value.strip().lower().split()) for value in values if value)

    @staticmethod
    def _metadata_keyword_values(value: Any) -> list[str]:
        if isinstance(value, Mapping):
            return [
                item
                for key in sorted(value, key=str)
                for item in LocalRecordBackend._metadata_keyword_values(value[key])
            ]
        if isinstance(value, (list, tuple, set, frozenset)):
            items = sorted(value, key=str) if isinstance(value, (set, frozenset)) else value
            return [
                item
                for nested in items
                for item in LocalRecordBackend._metadata_keyword_values(nested)
            ]
        return [str(value)]

    @staticmethod
    def _metadata_uri(metadata: dict[str, Any]) -> str:
        for key in (
            "uri",
            "source_file",
            "file_path",
            "path",
            "filename",
            "file_name",
        ):
            value = metadata.get(key)
            if value:
                return str(value)
        return ""

    @staticmethod
    def _key_chunks(keys: Sequence[str]) -> Iterable[list[str]]:
        for chunk in iter_ordered_key_chunks(keys, limit=DEFAULT_KEY_CHUNK_LIMIT):
            yield list(chunk)

    def index(self, records: list[Record]) -> None:
        """Index records for keyword retrieval."""
        self._record_writer._upsert_records(records)

    def upsert(self, records: list[Record], model_name: str, dim: int) -> None:
        """Persist records and their optional model-specific embeddings."""
        if dim < 1:
            raise ValueError("dim must be positive")
        rows = list(records)
        if not rows:
            return
        for record in rows:
            if record.embedding is None:
                raise ValueError(
                    "vector upsert requires an embedding for every record; "
                    f"missing embedding for {record.storage_key!r}"
                )
        with self._lock:
            conn = self._db.get_connection()
            try:
                existing_dims = {
                    int(row[0])
                    for row in conn.execute(
                        """
                        SELECT DISTINCT dim
                        FROM local_vectors_v2
                        WHERE encoder_namespace = ?
                        """,
                        (model_name,),
                    ).fetchall()
                }
                if existing_dims and existing_dims != {dim}:
                    existing_dim = next(iter(existing_dims))
                    raise ModelDimensionMismatchError(
                        f"Dimension mismatch for model {model_name!r}: "
                        f"expected {existing_dim}, got {dim}"
                    )
                packed_vectors: list[tuple[str, bytes, str]] = []
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
                                record_embedding_revision(record, model_name, dim),
                            )
                        )
                vector_affected = bool(packed_vectors)
                self._record_writer._write_records(conn, rows)
                conn.executemany(
                    """
                    INSERT INTO local_vectors_v2 (
                        storage_key, encoder_namespace, dim, embedding,
                        revision, format_version, normalization_policy
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(storage_key, encoder_namespace, dim) DO UPDATE SET
                        embedding = excluded.embedding,
                        revision = excluded.revision,
                        format_version = excluded.format_version,
                        normalization_policy = excluded.normalization_policy
                    """,
                    [
                        (
                            storage_key,
                            model_name,
                            dim,
                            embedding,
                            revision,
                            VECTOR_FORMAT_VERSION,
                            NORMALIZATION_POLICY,
                        )
                        for storage_key, embedding, revision in packed_vectors
                    ],
                )
                self._epoch_lane.bump(
                    conn,
                    keyword=True,
                    vector=vector_affected,
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    @staticmethod
    def _status_values(filters: SearchFilters | None) -> set[str]:
        if filters and filters.get("include_inactive"):
            return {"active", "stale", "archived"}
        if filters and filters.get("status") is not None:
            value = filters["status"]
            return {value.value if isinstance(value, RecordStatus) else str(value)}
        if filters and filters.get("lifecycle_status") is not None:
            value = filters["lifecycle_status"]
            return {value.value if isinstance(value, RecordStatus) else str(value)}
        values = filters.get("statuses") if filters else None
        if values is None and filters:
            values = filters.get("lifecycle_statuses")
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
        filters: SearchFilters | None,
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

    def search_keyword(
        self,
        query: str,
        k: int,
        filters: SearchFilters | None = None,
    ) -> list[RecordHit]:
        return self._keyword_engine.search_keyword(query, k, filters)

    def search_vector(
        self,
        query_vector: Vector,
        k: int,
        *,
        model_name: str,
        dim: int,
        filters: SearchFilters | None = None,
    ) -> list[RecordHit]:
        return self._vector_snapshot_engine.search_vector(
            query_vector,
            k,
            model_name=model_name,
            dim=dim,
            filters=filters,
        )

    @property
    def vector_engine(self) -> str:
        return self._vector_engine

    @property
    def faiss_threshold(self) -> int:
        return self._faiss_threshold

    def vector_count(self, model_name: str, dim: int) -> int:
        return self._vector_snapshot_engine.vector_count(model_name, dim)

    def vector_storage_stats(self, model_name: str, dim: int) -> tuple[int, int]:
        return self._vector_snapshot_engine.vector_storage_stats(model_name, dim)

    def _iter_vector_batches(
        self,
        model_name: str,
        dim: int,
    ) -> Iterable[list[sqlite3.Row]]:
        """Yield bounded vector rows for streamed optional-index builds."""
        return self._vector_snapshot_engine._iter_vector_batches(model_name, dim)

    def hydrate_record(
        self,
        record_id: RecordIdentity | str,
        *,
        source_kind: str | None = None,
        workspace_id: str | None = None,
    ) -> Record | None:
        return self._hydration_engine.hydrate_record(
            record_id,
            source_kind=source_kind,
            workspace_id=workspace_id,
        )

    def hydrate_records(
        self,
        identities: Sequence[RecordIdentity],
    ) -> dict[str, Record | None]:
        """Hydrate canonical identities with bounded record queries."""
        return self._hydration_engine.hydrate_records(identities)

    def chunk_parent(self, record_id: RecordIdentity | str) -> RecordIdentity | None:
        """Return the canonical parent identity for a persisted chunk."""
        return self._hydration_engine.chunk_parent(record_id)

    def chunk_records(
        self,
        parent_id: RecordIdentity | str,
    ) -> dict[str, Record]:
        """Hydrate persisted chunks for one canonical parent."""
        return self._hydration_engine.chunk_records(parent_id)

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
                existing_rows: list[sqlite3.Row] = []
                for key_chunk in self._key_chunks(record_ids):
                    placeholders = ",".join("?" for _ in key_chunk)
                    existing_rows.extend(
                        conn.execute(
                            f"""
                            SELECT storage_key, rowid, title, body, indexed_text, uri, keywords
                            FROM local_records
                            WHERE storage_key IN ({placeholders})
                            """,
                            key_chunk,
                        ).fetchall()
                    )
                    existing_vectors += int(
                        conn.execute(
                            f"""
                            SELECT COUNT(*) FROM local_vectors_v2
                            WHERE storage_key IN ({placeholders})
                            """,
                            key_chunk,
                        ).fetchone()[0]
                    )
                existing_records = len(existing_rows)
                if self._fts5_available:
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
                            for row in existing_rows
                        ],
                    )
                for key_chunk in self._key_chunks(record_ids):
                    placeholders = ",".join("?" for _ in key_chunk)
                    deleted_graph_edges += conn.execute(
                        f"""
                        DELETE FROM local_graph_edges
                        WHERE source_id IN ({placeholders})
                           OR target_id IN ({placeholders})
                        """,
                        (*key_chunk, *key_chunk),
                    ).rowcount
                    conn.execute(
                        f"DELETE FROM local_vectors_v2 WHERE storage_key IN ({placeholders})",
                        key_chunk,
                    )
                    conn.execute(
                        f"DELETE FROM local_records WHERE storage_key IN ({placeholders})",
                        key_chunk,
                    )
                self._epoch_lane.bump(
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
    def _fts5_available(self) -> bool:
        return self._keyword_engine._fts5_available

    @_fts5_available.setter
    def _fts5_available(self, value: bool) -> None:
        self._keyword_engine._fts5_available = value

    @property
    def keyword_index_available(self) -> bool:
        return self._keyword_engine.keyword_index_available

    @property
    def keyword_search_diagnostic(self) -> str:
        return self._keyword_engine.keyword_search_diagnostic

    @property
    def last_keyword_search_diagnostics(self) -> Mapping[str, int | bool]:
        return self._keyword_engine._last_keyword_search_diagnostics

    def check_keyword_index(self) -> bool:
        """Return whether the external-content keyword index matches records."""
        return self._keyword_engine.check_keyword_index()

    def rebuild_keyword_index(self) -> None:
        """Rebuild the keyword index from effective indexed text."""
        return self._keyword_engine.rebuild_keyword_index()

    def epoch(self) -> int:
        with self._lock:
            conn = self._db.get_connection()
            return self._epoch_lane.read(conn, _LocalEpochLane._RECORD_KEY)

    def keyword_epoch(self) -> int:
        return self._lane_epoch(_LocalEpochLane._LANE_KEYS["keyword"])

    def vector_epoch(self) -> int:
        return self._lane_epoch(_LocalEpochLane._LANE_KEYS["vector"])

    def graph_epoch(self) -> int:
        return self._lane_epoch(_LocalEpochLane._LANE_KEYS["graph"])

    def epochs(self) -> dict[str, int]:
        with self._lock:
            conn = self._db.get_connection()
            return self._epoch_lane.read_lanes(conn)

    def _lane_epoch(self, key: str) -> int:
        with self._lock:
            conn = self._db.get_connection()
            return self._epoch_lane.read(conn, key)

    def upsert_edges(
        self,
        edges: Sequence[GraphEdge | tuple[str, str, str, float]],
    ) -> None:
        self._graph_engine.upsert_edges(edges)

    def delete_edges(
        self,
        edges: Sequence[GraphEdge | tuple[str, str, str, float]],
    ) -> None:
        self._graph_engine.delete_edges(edges)

    def graph_integrity_errors(self) -> list[str]:
        return self._graph_engine.graph_integrity_errors()

    def check_graph_integrity(self) -> bool:
        return self._graph_engine.check_graph_integrity()

    def neighbors(
        self,
        record_id: RecordIdentity | str,
        edge_types: list[str] | None = None,
        depth: int = 1,
        max_neighbors: int | None = None,
    ) -> list[GraphNeighbor]:
        return self._graph_engine.neighbors(
            record_id,
            edge_types,
            depth,
            max_neighbors,
        )

    def incoming_neighbors(
        self,
        record_id: RecordIdentity | str,
        edge_types: list[str] | None = None,
        depth: int = 1,
        max_neighbors: int | None = None,
    ) -> list[GraphNeighbor]:
        return self._graph_engine.incoming_neighbors(
            record_id,
            edge_types,
            depth,
            max_neighbors,
        )

    def neighbors_many(
        self,
        identities: Sequence[RecordIdentity],
        *,
        depth: int,
        max_neighbors: int | None = None,
    ) -> dict[str, list[GraphNeighbor]]:
        return self._graph_engine.neighbors_many(
            identities,
            depth=depth,
            max_neighbors=max_neighbors,
        )

    def incoming_neighbors_many(
        self,
        identities: Sequence[RecordIdentity],
        *,
        depth: int,
        max_neighbors: int | None = None,
    ) -> dict[str, list[GraphNeighbor]]:
        return self._graph_engine.incoming_neighbors_many(
            identities,
            depth=depth,
            max_neighbors=max_neighbors,
        )

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
        self._selection_key: tuple[str, int] | None = None
        self._selection_epoch: int | None = None
        self._selection_use_faiss: bool | None = None

    @property
    def engine_name(self) -> str:
        return self._last_engine_name

    def _selected_store(
        self,
        model_name: str,
        dim: int,
    ) -> Any | None:
        use_faiss = self._engine == "faiss"
        if self._engine == "auto":
            key = (model_name, dim)
            selection_epoch = self._backend.vector_epoch()
            if (
                self._selection_key == key
                and self._selection_epoch == selection_epoch
                and self._selection_use_faiss is not None
            ):
                use_faiss = self._selection_use_faiss
            else:
                use_faiss = (
                    self._backend.vector_count(model_name, dim)
                    >= self._backend.faiss_threshold
                )
                self._selection_key = key
                self._selection_epoch = selection_epoch
                self._selection_use_faiss = use_faiss
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
        filters: SearchFilters | None = None,
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
        filters: SearchFilters | None = None,
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
        self, query: str, k: int, filters: SearchFilters | None = None
    ) -> list[RecordHit]:
        return self._backend.search_keyword(query, k, filters)

    @property
    def keyword_index_available(self) -> bool:
        return self._backend.keyword_index_available

    @property
    def keyword_search_diagnostic(self) -> str:
        return self._backend.keyword_search_diagnostic

    @property
    def last_keyword_search_diagnostics(self) -> Mapping[str, int | bool]:
        return self._backend.last_keyword_search_diagnostics

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
        max_neighbors: int | None = None,
    ) -> list[GraphNeighbor]:
        return self._backend.neighbors(
            record_id,
            edge_types,
            depth,
            max_neighbors,
        )

    def incoming_neighbors(
        self,
        record_id: RecordIdentity | str,
        edge_types: list[str] | None = None,
        depth: int = 1,
        max_neighbors: int | None = None,
    ) -> list[GraphNeighbor]:
        return self._backend.incoming_neighbors(
            record_id,
            edge_types,
            depth,
            max_neighbors,
        )

    def neighbors_many(
        self,
        identities: Sequence[RecordIdentity],
        *,
        depth: int,
        max_neighbors: int | None = None,
    ) -> dict[str, list[GraphNeighbor]]:
        return self._backend.neighbors_many(
            identities,
            depth=depth,
            max_neighbors=max_neighbors,
        )

    def incoming_neighbors_many(
        self,
        identities: Sequence[RecordIdentity],
        *,
        depth: int,
        max_neighbors: int | None = None,
    ) -> dict[str, list[GraphNeighbor]]:
        return self._backend.incoming_neighbors_many(
            identities,
            depth=depth,
            max_neighbors=max_neighbors,
        )


def _validate_neighbor_limit(max_neighbors: int | None) -> None:
    if max_neighbors is not None and max_neighbors <= 0:
        raise ValueError("max_neighbors must be positive")


def _sort_graph_neighbors(
    neighbors: Iterable[GraphNeighbor],
    max_neighbors: int | None,
) -> list[GraphNeighbor]:
    key = lambda item: (-item.weight, item.identity.storage_key, item.edge_type)
    if max_neighbors is None:
        return sorted(neighbors, key=key)
    return heapq.nsmallest(max_neighbors, neighbors, key=key)


SQLiteKeywordStore = LocalKeywordStore
SQLiteGraphStore = LocalGraphStore

__all__ = [
    "FAISSLocalVectorStore",
    "LocalGraphStore",
    "LocalKeywordStore",
    "LocalRecordBackend",
    "LocalVectorStore",
    "SQLiteExactVectorStore",
    "SQLiteGraphStore",
    "SQLiteKeywordStore",
]
