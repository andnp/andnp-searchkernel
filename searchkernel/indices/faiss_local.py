"""Optional FAISS adapter for the local record vector store."""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from searchkernel.domain import Record, RecordHit, RecordIdentity, Vector
from searchkernel.domain.vector_filters import (
    metadata_mapping,
    record_matches_vector_filters,
)
from searchkernel.indices.local_vectors import (
    NORMALIZATION_POLICY,
    VECTOR_FORMAT_VERSION,
    PackedVectorCodec,
)
from searchkernel.utils.atomic_io import atomic_write_binary, atomic_write_json

if TYPE_CHECKING:
    from searchkernel.indices.local import LocalRecordBackend


@dataclass(frozen=True, slots=True)
class _CandidateMetadata:
    source_id: str
    workspace_id: str | None
    source_kind: str
    status: str
    metadata: dict[str, Any]
    uri: str | None


@dataclass(frozen=True, slots=True)
class _FAISSState:
    index: Any
    encoder_namespace: str
    dim: int
    epoch: int
    ids: tuple[int, ...]
    storage_keys: tuple[str, ...]
    id_to_storage_key: dict[int, str]
    candidate_metadata: dict[str, _CandidateMetadata]


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
        self._state_lock = threading.RLock()
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
        if self._backend.vector_count(model_name, dim) == 0:
            return []
        try:
            state = self._get_state(model_name, dim)
            if state.epoch != self._backend.vector_epoch():
                state = self._get_state(model_name, dim)
            return self._search_state(
                state,
                query,
                k,
                filters=filters,
            )
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

    def _get_state(self, model_name: str, dim: int) -> _FAISSState:
        key = (model_name, dim)
        vector_epoch = self._backend.vector_epoch()
        with self._state_lock:
            cached = self._states.get(key)
            if cached is not None and cached.epoch == vector_epoch:
                return cached
            loaded = self._load_state(model_name, dim, vector_epoch)
            if loaded is not None:
                self._states[key] = loaded
                return loaded
            state = self._build_state(model_name, dim, vector_epoch)
            if self._backend.vector_epoch() != vector_epoch:
                raise RuntimeError("vector index changed while FAISS state was built")
            self._states[key] = state
            self._persist_state(state)
            return state

    @staticmethod
    def _stable_id(storage_key: str) -> int:
        value = int.from_bytes(
            hashlib.blake2b(storage_key.encode("utf-8"), digest_size=8).digest(),
            "little",
        ) & ((1 << 63) - 1)
        return value or 1

    def _build_state(
        self,
        model_name: str,
        dim: int,
        epoch: int,
    ) -> _FAISSState:
        import faiss

        index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
        ids: list[int] = []
        storage_keys: list[str] = []
        id_to_storage_key: dict[int, str] = {}
        candidate_metadata: dict[str, _CandidateMetadata] = {}
        for rows in self._backend._iter_vector_batches(model_name, dim):
            vectors = [
                PackedVectorCodec.decode(
                    row["embedding"],
                    dim,
                    context=f"stored embedding for {row['storage_key']}",
                )
                for row in rows
            ]
            batch_keys = [row["storage_key"] for row in rows]
            batch_ids = [self._stable_id(key) for key in batch_keys]
            if any(faiss_id in id_to_storage_key for faiss_id in batch_ids):
                raise ValueError("FAISS stable ID collision")
            index.add_with_ids(
                np.asarray(vectors, dtype=np.float32),
                np.asarray(batch_ids, dtype=np.int64),
            )
            ids.extend(batch_ids)
            storage_keys.extend(batch_keys)
            id_to_storage_key.update(zip(batch_ids, batch_keys, strict=True))
            candidate_metadata.update(
                {
                    row["storage_key"]: _CandidateMetadata(
                        source_id=row["source_id"],
                        workspace_id=row["workspace_id"],
                        source_kind=row["source_kind"],
                        status=row["status"],
                        metadata=dict(metadata_mapping(row["metadata"])),
                        uri=row["uri"],
                    )
                    for row in rows
                }
            )
        return _FAISSState(
            index=index,
            encoder_namespace=model_name,
            dim=dim,
            epoch=epoch,
            ids=tuple(ids),
            storage_keys=tuple(storage_keys),
            id_to_storage_key=id_to_storage_key,
            candidate_metadata=candidate_metadata,
        )

    def _search_state(
        self,
        state: _FAISSState,
        query: np.ndarray,
        k: int,
        *,
        filters: dict[str, Any] | None,
    ) -> list[RecordHit]:
        total = len(state.storage_keys)
        if total == 0:
            return []
        scan = min(total, max(k, int(np.ceil(k * self._overfetch_multiplier))))
        hits: dict[str, RecordHit] = {}
        for _ in range(self._max_scan_rounds):
            with self._state_lock:
                scores, ids = state.index.search(
                    np.asarray(query[None, :], dtype=np.float32),
                    scan,
                )
            for score, faiss_id in zip(scores[0], ids[0], strict=True):
                storage_key = state.id_to_storage_key.get(int(faiss_id))
                metadata = (
                    state.candidate_metadata.get(storage_key)
                    if storage_key is not None
                    else None
                )
                if (
                    storage_key is None
                    or metadata is None
                    or not record_matches_vector_filters(
                        storage_key=storage_key,
                        source_id=metadata.source_id,
                        workspace_id=metadata.workspace_id,
                        source_kind=metadata.source_kind,
                        status=metadata.status,
                        metadata=metadata.metadata,
                        uri=metadata.uri,
                        filters=filters,
                    )
                ):
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

    def _paths(self, model_name: str, dim: int) -> tuple[Path, Path]:
        if self._index_path is None:
            raise FileNotFoundError("FAISS persistence is disabled")
        if self._index_path.suffix:
            index_path = self._index_path
        else:
            digest = hashlib.sha256(
                f"{model_name}:{dim}".encode()
            ).hexdigest()[:16]
            index_path = self._index_path / f"{digest}.faiss"
        return index_path, index_path.with_suffix(".json")

    def _persist_state(self, state: _FAISSState) -> None:
        try:
            import faiss

            index_path, metadata_path = self._paths(
                state.encoder_namespace,
                state.dim,
            )
            atomic_write_binary(index_path, bytes(faiss.serialize_index(state.index)))
            atomic_write_json(
                metadata_path,
                {
                    "format_version": VECTOR_FORMAT_VERSION,
                    "normalization_policy": NORMALIZATION_POLICY,
                    "encoder_namespace": state.encoder_namespace,
                    "dim": state.dim,
                    "epoch": state.epoch,
                    "ids": list(state.ids),
                    "storage_keys": list(state.storage_keys),
                    "candidate_metadata": {
                        storage_key: {
                            "source_id": metadata.source_id,
                            "workspace_id": metadata.workspace_id,
                            "source_kind": metadata.source_kind,
                            "status": metadata.status,
                            "metadata": metadata.metadata,
                            "uri": metadata.uri,
                        }
                        for storage_key, metadata in state.candidate_metadata.items()
                    },
                    "tombstones": [],
                },
            )
        except Exception:  # noqa: BLE001 - corrupted optional indexes use exact fallback
            return

    def _load_state(
        self,
        model_name: str,
        dim: int,
        epoch: int,
    ) -> _FAISSState | None:
        try:
            import faiss

            index_path, metadata_path = self._paths(model_name, dim)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (
                metadata["format_version"] != VECTOR_FORMAT_VERSION
                or metadata["normalization_policy"] != NORMALIZATION_POLICY
                or metadata["encoder_namespace"] != model_name
                or metadata["dim"] != dim
                or metadata["epoch"] != epoch
            ):
                return None
            ids = tuple(int(value) for value in metadata["ids"])
            storage_keys = tuple(metadata["storage_keys"])
            if len(ids) != len(storage_keys) or len(set(ids)) != len(ids):
                return None
            candidate_metadata = {
                storage_key: _CandidateMetadata(
                    source_id=value["source_id"],
                    workspace_id=value["workspace_id"],
                    source_kind=value["source_kind"],
                    status=value["status"],
                    metadata=dict(metadata_mapping(value["metadata"])),
                    uri=value["uri"],
                )
                for storage_key, value in metadata["candidate_metadata"].items()
            }
            if set(candidate_metadata) != set(storage_keys):
                return None
            index = faiss.deserialize_index(np.frombuffer(index_path.read_bytes(), dtype=np.uint8))
            if index.d != dim or index.ntotal != len(ids):
                return None
            return _FAISSState(
                index=index,
                encoder_namespace=model_name,
                dim=dim,
                epoch=epoch,
                ids=ids,
                storage_keys=storage_keys,
                id_to_storage_key=dict(zip(ids, storage_keys, strict=True)),
                candidate_metadata=candidate_metadata,
            )
        except Exception:  # noqa: BLE001 - stale or corrupt indexes use exact fallback
            return None
