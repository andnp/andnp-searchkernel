"""Optional FAISS adapter for the local record vector store."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, Literal

import numpy as np

from searchkernel.domain import (
    Record,
    RecordHit,
    RecordIdentity,
    RecordStatus,
    SearchFilters,
    Vector,
)
from searchkernel.domain.vector_filters import (
    CompiledVectorFilter,
    compile_vector_filters,
    metadata_mapping,
)
from searchkernel.indices.local_vectors import (
    NORMALIZATION_POLICY,
    VECTOR_FORMAT_VERSION,
    PackedVectorCodec,
)
from searchkernel.utils.atomic_io import (
    atomic_write_binary,
    atomic_write_json,
    atomic_write_stream,
)

if TYPE_CHECKING:
    from searchkernel.indices.local import LocalRecordBackend


FAISSSearchStrategy = Literal["exact", "approximate"]


@dataclass(frozen=True, slots=True)
class FAISSConfiguration:
    """Validated construction-time settings for one FAISS vector store."""

    search_strategy: FAISSSearchStrategy = "exact"
    hnsw_m: int = 32
    hnsw_ef_construction: int = 40
    hnsw_ef_search: int = 16
    overfetch_multiplier: float = 4.0
    max_scan_rounds: int = 4
    max_scan_candidates: int = 100_000

    def __post_init__(self) -> None:
        if self.search_strategy not in {"exact", "approximate"}:
            raise ValueError("search_strategy must be exact or approximate")
        for name in (
            "hnsw_m",
            "hnsw_ef_construction",
            "hnsw_ef_search",
            "max_scan_rounds",
            "max_scan_candidates",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if not isinstance(self.overfetch_multiplier, (int, float)) or isinstance(
            self.overfetch_multiplier, bool
        ):
            raise TypeError("overfetch_multiplier must be numeric")
        if not math.isfinite(self.overfetch_multiplier) or self.overfetch_multiplier < 1.0:
            raise ValueError("overfetch_multiplier must be finite and at least 1")

    def as_dict(self) -> dict[str, Any]:
        return {
            "search_strategy": self.search_strategy,
            "hnsw_m": self.hnsw_m,
            "hnsw_ef_construction": self.hnsw_ef_construction,
            "hnsw_ef_search": self.hnsw_ef_search,
            "overfetch_multiplier": float(self.overfetch_multiplier),
            "max_scan_rounds": self.max_scan_rounds,
            "max_scan_candidates": self.max_scan_candidates,
        }

    @property
    def build_fingerprint(self) -> str:
        return self._fingerprint(
            {
                "search_strategy": self.search_strategy,
                "hnsw_m": self.hnsw_m,
                "hnsw_ef_construction": self.hnsw_ef_construction,
            }
        )

    @property
    def query_policy_fingerprint(self) -> str:
        return self._fingerprint(
            {
                "hnsw_ef_search": self.hnsw_ef_search,
                "overfetch_multiplier": float(self.overfetch_multiplier),
                "max_scan_rounds": self.max_scan_rounds,
                "max_scan_candidates": self.max_scan_candidates,
            }
        )

    @staticmethod
    def _fingerprint(values: dict[str, Any]) -> str:
        payload = json.dumps(
            values, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.as_dict(), ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class _CandidateMetadata:
    source_id: str
    workspace_id: str | None
    source_kind: str
    status: str
    metadata: dict[str, Any]
    uri: str | None


@dataclass(frozen=True, slots=True)
class _CandidateMetadataSidecar:
    path: Path
    offsets: dict[str, tuple[int, int]]


class _SidecarMetadataReader:
    def __init__(self, sidecar: _CandidateMetadataSidecar, handle: BinaryIO) -> None:
        self._sidecar = sidecar
        self._handle = handle

    def get(self, storage_key: str) -> _CandidateMetadata | None:
        location = self._sidecar.offsets.get(storage_key)
        if location is None:
            return None
        offset, length = location
        self._handle.seek(offset)
        raw = self._handle.read(length)
        if len(raw) != length or not raw.endswith(b"\n"):
            raise ValueError("FAISS metadata sidecar record is truncated")
        value = json.loads(raw)
        if value.get("storage_key") != storage_key:
            raise ValueError("FAISS metadata sidecar key mismatch")
        return _CandidateMetadata(
            source_id=value["source_id"],
            workspace_id=value["workspace_id"],
            source_kind=value["source_kind"],
            status=value["status"],
            metadata=dict(metadata_mapping(value["metadata"])),
            uri=value["uri"],
        )


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
    metadata_sidecar: _CandidateMetadataSidecar | None = None
    active_ids: frozenset[int] | None = None
    eligibility_masks: dict[tuple[str, str], frozenset[int]] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )


class FAISSLocalVectorStore:
    """FAISS-backed local vector store with exact-search fallback."""

    engine_name = "faiss"

    def __init__(
        self,
        backend: LocalRecordBackend,
        *,
        index_path: Path | None = None,
        search_strategy: FAISSSearchStrategy = "exact",
        overfetch_multiplier: float = 4.0,
        max_scan_rounds: int = 4,
        hnsw_m: int = 32,
        hnsw_ef_construction: int = 40,
        hnsw_ef_search: int = 16,
        max_scan_candidates: int = 100_000,
        configuration: FAISSConfiguration | None = None,
    ) -> None:
        self._backend = backend
        self._index_path = index_path
        self._configuration = configuration or FAISSConfiguration(
            search_strategy=search_strategy,
            hnsw_m=hnsw_m,
            hnsw_ef_construction=hnsw_ef_construction,
            hnsw_ef_search=hnsw_ef_search,
            overfetch_multiplier=overfetch_multiplier,
            max_scan_rounds=max_scan_rounds,
            max_scan_candidates=max_scan_candidates,
        )
        self._state_lock = threading.RLock()
        self._states: dict[tuple[str, int], _FAISSState] = {}
        self._last_search_diagnostics: dict[str, Any] = {
            "strategy": self._configuration.search_strategy,
            "configuration_fingerprint": self._configuration.fingerprint,
            "build_fingerprint": self._configuration.build_fingerprint,
            "query_policy_fingerprint": (
                self._configuration.query_policy_fingerprint
            ),
            "fallback": False,
        }

    @property
    def search_strategy(self) -> FAISSSearchStrategy:
        return self._configuration.search_strategy

    @property
    def configuration(self) -> FAISSConfiguration:
        return self._configuration

    @property
    def last_search_diagnostics(self) -> dict[str, Any]:
        return dict(self._last_search_diagnostics)

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
        compiled_filter: CompiledVectorFilter | None = None,
    ) -> list[RecordHit]:
        if k < 1:
            return []
        query = PackedVectorCodec.normalize(
            query_vector, dim, context="query vector"
        )
        predicate = compiled_filter or compile_vector_filters(filters)
        if predicate.source_scoped_filters:
            self._last_search_diagnostics = {
                "strategy": "exact_filtered",
                "requested_k": k,
                "fallback": False,
            }
            hits = self._backend.search_vector(
                query_vector,
                k,
                model_name=model_name,
                dim=dim,
                filters=filters,
                compiled_filter=predicate,
            )
            self._last_search_diagnostics.update(
                {"returned": len(hits), "under_returned": len(hits) < k}
            )
            return hits
        self._last_search_diagnostics = {
            "strategy": self.search_strategy,
            "configuration_fingerprint": self.configuration.fingerprint,
            "build_fingerprint": self.configuration.build_fingerprint,
            "query_policy_fingerprint": self.configuration.query_policy_fingerprint,
            "requested_k": k,
            "fallback": False,
        }
        try:
            state = self._get_state(model_name, dim)
            hits = self._search_state(
                state,
                query,
                k,
                filters=filters,
                compiled_filter=predicate,
            )
            self._last_search_diagnostics["returned"] = len(hits)
            self._last_search_diagnostics["under_returned"] = len(hits) < k
            return hits
        except Exception as exc:  # noqa: BLE001 - optional indexes use exact fallback
            self._last_search_diagnostics.update(
                {
                    "fallback": True,
                    "fallback_reason": f"{type(exc).__name__}: {exc}",
                }
            )
            hits = self._backend.search_vector(
                query_vector,
                k,
                model_name=model_name,
                dim=dim,
                filters=filters,
                compiled_filter=predicate,
            )
            self._last_search_diagnostics["returned"] = len(hits)
            self._last_search_diagnostics["under_returned"] = len(hits) < k
            return hits

    def delete(self, record_ids: list[str]) -> None:
        self._backend.delete(record_ids)

    def epoch(self) -> int:
        return self._backend.epoch()

    def migrate_legacy_persistence(self, model_name: str, dim: int) -> bool:
        """Explicitly migrate a valid legacy artifact to split persistence.

        Return ``True`` when a split generation is already valid or is
        published and verified. Return ``False`` when legacy validation,
        publication, or post-publication verification fails. The migration
        status and failure reason are recorded in ``last_search_diagnostics``;
        legacy files are never deleted or overwritten.
        """
        key = (model_name, dim)
        with self._state_lock:
            epoch = self._backend.vector_epoch()
            try:
                manifest_path = self._manifest_path(model_name, dim)
                if manifest_path.is_file():
                    split_state = self._load_split_state(
                        model_name, dim, epoch, manifest_path
                    )
                    if split_state is not None:
                        self._states[key] = split_state
                        self._last_search_diagnostics.update(
                            {
                                "migration": "already_split",
                                "persistence": "loaded",
                            }
                        )
                        return True
                legacy_state = self._load_legacy_state(model_name, dim, epoch)
                if legacy_state is None:
                    self._last_search_diagnostics.update(
                        {
                            "migration": "failed",
                            "migration_reason": "valid legacy artifact not found",
                        }
                    )
                    return False
                if self._backend.vector_epoch() != epoch:
                    raise RuntimeError("vector index changed during migration")
                if not self._persist_state(legacy_state):
                    self._last_search_diagnostics.update(
                        {
                            "migration": "failed",
                            "migration_reason": "split publication failed",
                        }
                    )
                    return False
                verified = self._load_split_state(
                    model_name, dim, epoch, manifest_path
                )
                if verified is None:
                    self._last_search_diagnostics.update(
                        {
                            "migration": "failed",
                            "migration_reason": "published split verification failed",
                        }
                    )
                    return False
                self._states[key] = verified
                self._last_search_diagnostics.update(
                    {"migration": "migrated", "persistence": "loaded"}
                )
                return True
            except Exception as exc:  # noqa: BLE001 - explicit migration reports failure
                self._last_search_diagnostics.update(
                    {
                        "migration": "failed",
                        "migration_reason": f"{type(exc).__name__}: {exc}",
                    }
                )
                return False

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

    def verify_recall(
        self,
        query_vector: Vector,
        k: int,
        *,
        model_name: str,
        dim: int,
        filters: SearchFilters | None = None,
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
                self._last_search_diagnostics["persistence"] = "loaded"
                self._states[key] = loaded
                return loaded
            state = self._build_state(model_name, dim, vector_epoch)
            if self._backend.vector_epoch() != vector_epoch:
                raise RuntimeError("vector index changed while FAISS state was built")
            self._states[key] = state
            self._persist_state(state)
            self._last_search_diagnostics["persistence"] = "rebuilt"
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

        if self.search_strategy == "exact":
            base_index = faiss.IndexFlatIP(dim)
        else:
            base_index = faiss.IndexHNSWFlat(
                dim,
                self.configuration.hnsw_m,
                faiss.METRIC_INNER_PRODUCT,
            )
            self._set_hnsw_settings(base_index)
        index = faiss.IndexIDMap2(base_index)
        ids: list[int] = []
        storage_keys: list[str] = []
        id_to_storage_key: dict[int, str] = {}
        candidate_metadata: dict[str, _CandidateMetadata] = {}
        for rows in self._backend.iter_vector_batches(model_name, dim):
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
            active_ids=frozenset(
                faiss_id
                for faiss_id, storage_key in zip(
                    ids, storage_keys, strict=True
                )
                if candidate_metadata[storage_key].status == RecordStatus.ACTIVE.value
            ),
        )

    def _search_state(
        self,
        state: _FAISSState,
        query: np.ndarray,
        k: int,
        *,
        filters: SearchFilters | None,
        compiled_filter: CompiledVectorFilter | None = None,
    ) -> list[RecordHit]:
        total = len(state.storage_keys)
        if total == 0:
            return []
        exact = self.search_strategy == "exact"
        # Tombstoned vectors stay resident in the index and are discarded
        # after the search, so every budget counts residency, not live keys.
        resident = int(state.index.ntotal)
        tombstones = max(0, resident - total)
        candidate_budget = (
            resident
            if exact
            else min(resident, self.configuration.max_scan_candidates)
        )
        scan = (
            min(k + tombstones, candidate_budget)
            if exact and not filters
            else candidate_budget
            if exact
            else min(
                candidate_budget,
                max(k, int(np.ceil(k * self.configuration.overfetch_multiplier)))
                + tombstones,
            )
        )
        self._last_search_diagnostics.update(
            {
                "candidate_budget": candidate_budget,
                "scan_limit": scan,
                "candidate_budget_hit": False,
                "scan_rounds": 0,
            }
        )
        predicate = compiled_filter or compile_vector_filters(filters)
        id_only, eligible_ids = self._id_only_filter(state, predicate)
        eligibility_ids = (
            self._eligibility_mask(state, predicate)
            if exact and state.metadata_sidecar is None and not id_only
            else eligible_ids
        )
        hits: dict[str, RecordHit] = {}
        with self._metadata_reader(state, required=not id_only) as reader:
            for scan_round in range(self.configuration.max_scan_rounds):
                self._last_search_diagnostics["scan_rounds"] = scan_round + 1
                with self._state_lock:
                    scores, ids = state.index.search(
                        np.asarray(query[None, :], dtype=np.float32),
                        scan,
                    )
                valid_storage_keys = self._validated_storage_keys(
                    state,
                    (int(faiss_id) for faiss_id in ids[0]),
                    predicate,
                    reader=reader,
                    eligible_ids=eligibility_ids,
                )
                returned_count = int(np.count_nonzero(ids[0] != -1))
                for score, faiss_id in zip(scores[0], ids[0], strict=True):
                    storage_key = valid_storage_keys.get(int(faiss_id))
                    if storage_key is None:
                        continue
                    if storage_key in hits:
                        continue
                    hits[storage_key] = RecordHit(
                        RecordIdentity.from_storage_key(storage_key),
                        float(score),
                    )
                    if len(hits) >= k:
                        break
                if len(hits) >= k or scan >= candidate_budget:
                    self._last_search_diagnostics["candidate_budget_hit"] = (
                        not exact and scan >= candidate_budget and len(hits) < k
                    )
                    break
                if returned_count < scan:
                    break
                scan = min(candidate_budget, max(scan + 1, scan * 2))
                self._last_search_diagnostics["scan_limit"] = scan
        return sorted(
            hits.values(),
            key=lambda hit: (-hit.score, hit.storage_key),
        )[:k]

    @staticmethod
    def _id_only_filter(
        state: _FAISSState,
        predicate: CompiledVectorFilter,
    ) -> tuple[bool, frozenset[int] | None]:
        if (
            predicate.workspace_id is not None
            or predicate.source_kinds is not None
            or predicate.candidate_keys is not None
            or predicate.excluded_storage_keys is not None
            or predicate.requires_metadata
        ):
            return False, None
        if predicate.statuses == frozenset({RecordStatus.ACTIVE.value}):
            if state.active_ids is None:
                return False, None
            return True, state.active_ids
        if predicate.statuses == frozenset(status.value for status in RecordStatus):
            return True, None
        return False, None

    @staticmethod
    def _canonical_scalar_filter_key(
        predicate: CompiledVectorFilter,
    ) -> tuple[str, str] | None:
        if (
            not isinstance(predicate.workspace_id, str)
            or predicate.source_kinds is None
            or len(predicate.source_kinds) != 1
            or predicate.statuses != frozenset({"active"})
            or predicate.candidate_keys is not None
            or predicate.excluded_storage_keys is not None
            or predicate.requires_metadata
        ):
            return None
        return predicate.workspace_id, next(iter(predicate.source_kinds))

    def _eligibility_mask(
        self,
        state: _FAISSState,
        predicate: CompiledVectorFilter,
    ) -> frozenset[int] | None:
        key = self._canonical_scalar_filter_key(predicate)
        if key is None:
            return None
        with self._state_lock:
            cached = state.eligibility_masks.get(key)
            if cached is not None:
                return cached
            workspace_id, source_kind = key
            eligible = frozenset(
                faiss_id
                for faiss_id, storage_key in zip(
                    state.ids, state.storage_keys, strict=True
                )
                if (
                    state.candidate_metadata[storage_key].workspace_id == workspace_id
                    and state.candidate_metadata[storage_key].source_kind
                    == source_kind
                    and state.candidate_metadata[storage_key].status == "active"
                )
            )
            state.eligibility_masks[key] = eligible
            return eligible

    @contextmanager
    def _metadata_reader(
        self,
        state: _FAISSState,
        *,
        required: bool,
    ) -> Iterator[
        Mapping[str, _CandidateMetadata] | _SidecarMetadataReader | None
    ]:
        if not required:
            yield None
            return
        if state.metadata_sidecar is None:
            yield state.candidate_metadata
            return
        with state.metadata_sidecar.path.open("rb") as handle:
            yield _SidecarMetadataReader(state.metadata_sidecar, handle)

    @staticmethod
    def _validated_storage_keys(
        state: _FAISSState,
        faiss_ids: Any,
        predicate: CompiledVectorFilter,
        *,
        reader: Mapping[str, _CandidateMetadata] | _SidecarMetadataReader | None = None,
        eligible_ids: frozenset[int] | None = None,
    ) -> dict[int, str]:
        valid: dict[int, str] = {}
        for faiss_id in faiss_ids:
            storage_key = state.id_to_storage_key.get(faiss_id)
            if eligible_ids is not None:
                if storage_key is not None and faiss_id in eligible_ids:
                    valid[faiss_id] = storage_key
                continue
            if storage_key is None:
                continue
            if reader is None:
                if eligible_ids is None or faiss_id in eligible_ids:
                    valid[faiss_id] = storage_key
                continue
            metadata = reader.get(storage_key) if reader is not None else None
            if (
                storage_key is not None
                and metadata is not None
                and predicate.matches(
                    storage_key=storage_key,
                    source_id=metadata.source_id,
                    workspace_id=metadata.workspace_id,
                    source_kind=metadata.source_kind,
                    status=metadata.status,
                    metadata=metadata.metadata,
                    uri=metadata.uri,
                )
            ):
                valid[faiss_id] = storage_key
        return valid

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

    def _manifest_path(self, model_name: str, dim: int) -> Path:
        index_path, _ = self._paths(model_name, dim)
        return index_path.with_suffix(".manifest.json")

    @staticmethod
    def _metadata_value(metadata: _CandidateMetadata) -> dict[str, Any]:
        return {
            "source_id": metadata.source_id,
            "workspace_id": metadata.workspace_id,
            "source_kind": metadata.source_kind,
            "status": metadata.status,
            "metadata": metadata.metadata,
            "uri": metadata.uri,
        }

    @staticmethod
    def _common_persistence_metadata(state: _FAISSState) -> dict[str, Any]:
        return {
            "format_version": VECTOR_FORMAT_VERSION,
            "normalization_policy": NORMALIZATION_POLICY,
            "encoder_namespace": state.encoder_namespace,
            "dim": state.dim,
            "search_strategy": "exact",
            "epoch": state.epoch,
        }

    def _persistence_metadata(self, state: _FAISSState) -> dict[str, Any]:
        return {
            **self._common_persistence_metadata(state),
            "search_strategy": self.search_strategy,
            "configuration": self.configuration.as_dict(),
            "configuration_fingerprint": self.configuration.fingerprint,
            "build_fingerprint": self.configuration.build_fingerprint,
            "query_policy_fingerprint": (
                self.configuration.query_policy_fingerprint
            ),
            "ids": list(state.ids),
            "storage_keys": list(state.storage_keys),
            "active_ids": sorted(state.active_ids or ()),
            "tombstones": [],
        }

    def _persist_state(self, state: _FAISSState) -> bool:
        try:
            import faiss

            index_path, _ = self._paths(
                state.encoder_namespace,
                state.dim,
            )
            index_bytes = bytes(faiss.serialize_index(state.index))
            generation = (
                f"{state.epoch}-{self.configuration.build_fingerprint[:16]}-"
                f"{time.time_ns()}"
            )
            generation_index = index_path.with_name(
                f"{index_path.stem}.{generation}.faiss"
            )
            generation_metadata = index_path.with_name(
                f"{index_path.stem}.{generation}.jsonl"
            )
            offsets: dict[str, dict[str, int]] = {}

            def write_metadata(handle: BinaryIO) -> None:
                for storage_key in state.storage_keys:
                    line = json.dumps(
                        {
                            "storage_key": storage_key,
                            **self._metadata_value(
                                state.candidate_metadata[storage_key]
                            ),
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode("utf-8") + b"\n"
                    offset = handle.tell()
                    handle.write(line)
                    offsets[storage_key] = {
                        "offset": offset,
                        "length": len(line),
                    }

            atomic_write_binary(generation_index, index_bytes)
            metadata_size = atomic_write_stream(generation_metadata, write_metadata)
            manifest = {
                **self._persistence_metadata(state),
                "persistence_format": "split",
                "generation": generation,
                "index_file": generation_index.name,
                "metadata_file": generation_metadata.name,
                "index_size": len(index_bytes),
                "metadata_size": metadata_size,
                "metadata_offsets": offsets,
            }
            atomic_write_json(self._manifest_path(state.encoder_namespace, state.dim), manifest)
            return True
        except Exception as exc:  # noqa: BLE001 - optional persistence is best effort
            self._last_search_diagnostics["persistence_error"] = (
                f"{type(exc).__name__}: {exc}"
            )
            return False

    def _load_state(
        self,
        model_name: str,
        dim: int,
        epoch: int,
    ) -> _FAISSState | None:
        try:
            manifest_path = self._manifest_path(model_name, dim)
            if manifest_path.is_file():
                split_state = self._load_split_state(
                    model_name, dim, epoch, manifest_path
                )
                if split_state is not None:
                    return split_state
            return self._load_legacy_state(model_name, dim, epoch)
        except Exception as exc:  # noqa: BLE001 - stale or corrupt indexes rebuild
            self._last_search_diagnostics["persistence_reason"] = (
                f"{type(exc).__name__}: {exc}"
            )
            return None

    def _load_split_state(
        self,
        model_name: str,
        dim: int,
        epoch: int,
        manifest_path: Path,
    ) -> _FAISSState | None:
        import faiss

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        index_path, _ = self._paths(model_name, dim)
        generation = manifest["generation"]
        expected_prefix = f"{epoch}-{self.configuration.build_fingerprint[:16]}-"
        if (
            not isinstance(generation, str)
            or not generation.startswith(expected_prefix)
            or not generation[len(expected_prefix) :].isdigit()
        ):
            return None
        index_name = f"{index_path.stem}.{generation}.faiss"
        metadata_name = f"{index_path.stem}.{generation}.jsonl"
        if (
            manifest.get("persistence_format") != "split"
            or manifest.get("index_file") != index_name
            or manifest.get("metadata_file") != metadata_name
            or Path(index_name).name != index_name
            or Path(metadata_name).name != metadata_name
        ):
            return None
        build_fingerprint = manifest.get("build_fingerprint")
        if (
            manifest.get("format_version") != VECTOR_FORMAT_VERSION
            or manifest.get("normalization_policy") != NORMALIZATION_POLICY
            or manifest.get("encoder_namespace") != model_name
            or manifest.get("dim") != dim
            or build_fingerprint != self.configuration.build_fingerprint
            or manifest.get("epoch") != epoch
        ):
            return None
        ids, storage_keys = self._load_persisted_keys(manifest)
        active_ids = self._load_active_ids(manifest, ids)
        offsets = self._load_metadata_offsets(manifest, storage_keys)
        generation_index = index_path.with_name(index_name)
        generation_metadata = index_path.with_name(metadata_name)
        if (
            not generation_index.is_file()
            or not generation_metadata.is_file()
            or generation_index.stat().st_size != manifest["index_size"]
            or generation_metadata.stat().st_size != manifest["metadata_size"]
        ):
            return None
        index = faiss.deserialize_index(
            np.frombuffer(generation_index.read_bytes(), dtype=np.uint8)
        )
        if index.d != dim or index.ntotal != len(ids):
            return None
        self._restore_hnsw_settings(index)
        return _FAISSState(
            index=index,
            encoder_namespace=model_name,
            dim=dim,
            epoch=epoch,
            ids=ids,
            storage_keys=storage_keys,
            id_to_storage_key=dict(zip(ids, storage_keys, strict=True)),
            candidate_metadata={},
            metadata_sidecar=_CandidateMetadataSidecar(
                generation_metadata, offsets
            ),
            active_ids=active_ids,
        )

    @staticmethod
    def _load_persisted_keys(
        metadata: Mapping[str, Any],
    ) -> tuple[tuple[int, ...], tuple[str, ...]]:
        ids = tuple(int(value) for value in metadata["ids"])
        storage_keys = tuple(str(value) for value in metadata["storage_keys"])
        if (
            len(ids) != len(storage_keys)
            or len(set(ids)) != len(ids)
            or len(set(storage_keys)) != len(storage_keys)
        ):
            raise ValueError("FAISS persisted keys are not unique")
        return ids, storage_keys

    @staticmethod
    def _load_active_ids(
        metadata: Mapping[str, Any],
        ids: tuple[int, ...],
    ) -> frozenset[int]:
        raw_active_ids = metadata["active_ids"]
        if not isinstance(raw_active_ids, list):
            raise TypeError("FAISS active IDs must be an explicit list")
        active_ids = tuple(int(value) for value in raw_active_ids)
        if len(set(active_ids)) != len(active_ids) or not set(active_ids).issubset(
            ids
        ):
            raise ValueError("FAISS active IDs are not a unique persisted subset")
        return frozenset(active_ids)

    @staticmethod
    def _load_metadata_offsets(
        metadata: Mapping[str, Any],
        storage_keys: tuple[str, ...],
    ) -> dict[str, tuple[int, int]]:
        raw_offsets = metadata["metadata_offsets"]
        if not isinstance(raw_offsets, dict) or set(raw_offsets) != set(storage_keys):
            raise ValueError("FAISS metadata offsets do not match storage keys")
        metadata_size = int(metadata["metadata_size"])
        offsets: dict[str, tuple[int, int]] = {}
        previous_end = 0
        for storage_key in storage_keys:
            raw_location = raw_offsets[storage_key]
            offset = int(raw_location["offset"])
            length = int(raw_location["length"])
            if offset < previous_end or length < 1 or offset + length > metadata_size:
                raise ValueError("FAISS metadata offsets are invalid")
            offsets[storage_key] = (offset, length)
            previous_end = offset + length
        return offsets

    def _load_legacy_state(
        self,
        model_name: str,
        dim: int,
        epoch: int,
    ) -> _FAISSState | None:
        import faiss

        index_path, metadata_path = self._paths(model_name, dim)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        build_fingerprint = metadata.get("build_fingerprint")
        configuration_matches = (
            build_fingerprint == self.configuration.build_fingerprint
            if build_fingerprint is not None
            else metadata.get("configuration_fingerprint")
            == self.configuration.fingerprint
        )
        if (
            metadata["format_version"] != VECTOR_FORMAT_VERSION
            or metadata["normalization_policy"] != NORMALIZATION_POLICY
            or metadata["encoder_namespace"] != model_name
            or metadata["dim"] != dim
            or not configuration_matches
            or metadata["epoch"] != epoch
        ):
            return None
        ids, storage_keys = self._load_persisted_keys(metadata)
        candidate_metadata_values = metadata["candidate_metadata"]
        if not isinstance(candidate_metadata_values, list) or len(
            candidate_metadata_values
        ) != len(storage_keys):
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
            for storage_key, value in zip(
                storage_keys, candidate_metadata_values, strict=True
            )
        }
        index = faiss.deserialize_index(
            np.frombuffer(index_path.read_bytes(), dtype=np.uint8)
        )
        if index.d != dim or index.ntotal != len(ids):
            return None
        self._restore_hnsw_settings(index)
        return _FAISSState(
            index=index,
            encoder_namespace=model_name,
            dim=dim,
            epoch=epoch,
            ids=ids,
            storage_keys=storage_keys,
            id_to_storage_key=dict(zip(ids, storage_keys, strict=True)),
            candidate_metadata=candidate_metadata,
            active_ids=frozenset(
                faiss_id
                for faiss_id, storage_key in zip(ids, storage_keys, strict=True)
                if candidate_metadata[storage_key].status == RecordStatus.ACTIVE.value
            ),
        )

    def _restore_hnsw_settings(self, index: Any) -> None:
        if self.search_strategy != "approximate":
            return
        import faiss

        base_index = faiss.downcast_index(index.index)
        self._set_hnsw_settings(base_index)

    def _set_hnsw_settings(self, index: object) -> None:
        import faiss

        if not isinstance(index, faiss.IndexHNSW):
            raise TypeError("FAISS approximate index must expose HNSW settings")
        index.hnsw.efConstruction = self.configuration.hnsw_ef_construction
        index.hnsw.efSearch = self.configuration.hnsw_ef_search
