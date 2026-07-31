"""Optional FAISS adapter for the local record vector store."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from searchkernel.domain import Record, RecordHit, RecordIdentity, Vector
from searchkernel.indices.local_vectors import (
    NORMALIZATION_POLICY,
    VECTOR_FORMAT_VERSION,
    PackedVectorCodec,
    VectorSnapshot,
)
from searchkernel.utils.atomic_io import atomic_write_binary, atomic_write_json

if TYPE_CHECKING:
    from searchkernel.indices.local import LocalRecordBackend


@dataclass(frozen=True, slots=True)
class _FAISSState:
    index: Any
    epoch: int
    ids: tuple[int, ...]
    storage_keys: tuple[str, ...]
    id_to_storage_key: dict[int, str]


class FAISSLocalVectorStore:
    """FAISS-backed local vector store with exact-search fallback."""

    engine_name = "faiss"

    def __init__(
        self,
        backend: LocalRecordBackend,
        *,
        index_path: Path | None = None,
        overfetch_multiplier: float = 4.0,
        max_scan_rounds: int = 4,
    ) -> None:
        if overfetch_multiplier < 1.0:
            raise ValueError("overfetch_multiplier must be at least 1")
        if max_scan_rounds < 1:
            raise ValueError("max_scan_rounds must be positive")
        self._backend = backend
        self._index_path = index_path
        self._overfetch_multiplier = overfetch_multiplier
        self._max_scan_rounds = max_scan_rounds
        self._states: dict[tuple[str, int], _FAISSState] = {}

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
        if k < 1:
            return []
        query = PackedVectorCodec.normalize(
            query_vector, dim, context="query vector"
        )
        snapshot = self._backend._get_vector_snapshot(model_name, dim)
        eligible = snapshot.filter_mask(
            filters,
            status_values=self._backend._status_values(filters),
            filter_values=self._backend._filter_values,
        )
        if not np.any(eligible):
            return []
        try:
            state = self._get_state(snapshot)
            return self._search_state(state, snapshot, query, eligible, k)
        except Exception:  # noqa: BLE001 - broken optional indexes use exact fallback
            return self._backend.search_vector(
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

    def verify_recall(
        self,
        query_vector: Vector,
        k: int,
        *,
        model_name: str,
        dim: int,
        filters: dict[str, Any] | None = None,
    ) -> float:
        exact = self._backend.search_vector(
            query_vector,
            k,
            model_name=model_name,
            dim=dim,
            filters=filters,
        )
        approximate = self.search(
            query_vector,
            k,
            model_name=model_name,
            dim=dim,
            filters=filters,
        )
        if not exact:
            return 1.0
        return len({hit.storage_key for hit in exact} & {
            hit.storage_key for hit in approximate
        }) / len(exact)

    def _get_state(self, snapshot: VectorSnapshot) -> _FAISSState:
        key = (snapshot.encoder_namespace, snapshot.dim)
        cached = self._states.get(key)
        if cached is not None and cached.epoch == snapshot.epoch:
            return cached
        loaded = self._load_state(snapshot)
        if loaded is not None:
            self._states[key] = loaded
            return loaded
        state = self._build_state(snapshot)
        self._states[key] = state
        self._persist_state(snapshot, state)
        return state

    @staticmethod
    def _stable_id(storage_key: str) -> int:
        value = int.from_bytes(
            hashlib.blake2b(storage_key.encode("utf-8"), digest_size=8).digest(),
            "little",
        ) & ((1 << 63) - 1)
        return value or 1

    def _build_state(self, snapshot: VectorSnapshot) -> _FAISSState:
        import faiss

        ids = tuple(self._stable_id(key) for key in snapshot.storage_keys)
        if len(set(ids)) != len(ids):
            raise ValueError("FAISS stable ID collision")
        index = faiss.IndexIDMap2(faiss.IndexFlatIP(snapshot.dim))
        if ids:
            index.add_with_ids(
                np.asarray(snapshot.matrix, dtype=np.float32),
                np.asarray(ids, dtype=np.int64),
            )
        return _FAISSState(
            index=index,
            epoch=snapshot.epoch,
            ids=ids,
            storage_keys=snapshot.storage_keys,
            id_to_storage_key=dict(zip(ids, snapshot.storage_keys, strict=True)),
        )

    def _search_state(
        self,
        state: _FAISSState,
        snapshot: VectorSnapshot,
        query: np.ndarray,
        eligible: np.ndarray,
        k: int,
    ) -> list[RecordHit]:
        eligible_keys = {
            snapshot.storage_keys[position]
            for position in np.flatnonzero(eligible)
        }
        total = len(state.storage_keys)
        scan = min(total, max(k, int(np.ceil(k * self._overfetch_multiplier))))
        hits: dict[str, RecordHit] = {}
        for _ in range(self._max_scan_rounds):
            scores, ids = state.index.search(
                np.asarray(query[None, :], dtype=np.float32),
                scan,
            )
            for score, faiss_id in zip(scores[0], ids[0], strict=True):
                storage_key = state.id_to_storage_key.get(int(faiss_id))
                if storage_key is None or storage_key not in eligible_keys:
                    continue
                if storage_key in hits:
                    continue
                hits[storage_key] = RecordHit(
                    RecordIdentity.from_storage_key(storage_key),
                    float(score),
                )
                if len(hits) >= k:
                    break
            if len(hits) >= k or scan >= total:
                break
            scan = min(total, max(scan + 1, scan * 2))
        return sorted(
            hits.values(),
            key=lambda hit: (-hit.score, hit.storage_key),
        )[:k]

    def _paths(self, snapshot: VectorSnapshot) -> tuple[Path, Path]:
        if self._index_path is None:
            raise FileNotFoundError("FAISS persistence is disabled")
        if self._index_path.suffix:
            index_path = self._index_path
        else:
            digest = hashlib.sha256(
                f"{snapshot.encoder_namespace}:{snapshot.dim}".encode()
            ).hexdigest()[:16]
            index_path = self._index_path / f"{digest}.faiss"
        return index_path, index_path.with_suffix(".json")

    def _persist_state(self, snapshot: VectorSnapshot, state: _FAISSState) -> None:
        try:
            import faiss

            index_path, metadata_path = self._paths(snapshot)
            atomic_write_binary(index_path, bytes(faiss.serialize_index(state.index)))
            atomic_write_json(
                metadata_path,
                {
                    "format_version": VECTOR_FORMAT_VERSION,
                    "normalization_policy": NORMALIZATION_POLICY,
                    "encoder_namespace": snapshot.encoder_namespace,
                    "dim": snapshot.dim,
                    "epoch": snapshot.epoch,
                    "ids": list(state.ids),
                    "storage_keys": list(state.storage_keys),
                    "tombstones": [],
                },
            )
        except Exception:  # noqa: BLE001 - corrupted optional indexes use exact fallback
            return

    def _load_state(self, snapshot: VectorSnapshot) -> _FAISSState | None:
        try:
            import faiss

            index_path, metadata_path = self._paths(snapshot)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (
                metadata["format_version"] != VECTOR_FORMAT_VERSION
                or metadata["normalization_policy"] != NORMALIZATION_POLICY
                or metadata["encoder_namespace"] != snapshot.encoder_namespace
                or metadata["dim"] != snapshot.dim
                or metadata["epoch"] != snapshot.epoch
                or tuple(metadata["storage_keys"]) != snapshot.storage_keys
            ):
                return None
            ids = tuple(int(value) for value in metadata["ids"])
            if len(ids) != len(snapshot.storage_keys):
                return None
            index = faiss.deserialize_index(np.frombuffer(index_path.read_bytes(), dtype=np.uint8))
            if index.d != snapshot.dim or index.ntotal != len(ids):
                return None
            return _FAISSState(
                index=index,
                epoch=snapshot.epoch,
                ids=ids,
                storage_keys=snapshot.storage_keys,
                id_to_storage_key=dict(zip(ids, snapshot.storage_keys, strict=True)),
            )
        except Exception:  # noqa: BLE001 - stale or corrupt indexes use exact fallback
            return None
