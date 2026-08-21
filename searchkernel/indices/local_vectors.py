"""Compact vector storage primitives for the local record backend."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from searchkernel.domain.vector_filters import (
    CompiledVectorFilter,
    compile_vector_filters,
    metadata_mapping,
)

VECTOR_FORMAT_VERSION = 2
NORMALIZATION_POLICY = "l2"
_SCALAR_FILTER_NAMES = frozenset(
    {
        "candidate_ids",
        "candidate_storage_keys",
        "include_inactive",
        "lifecycle_status",
        "lifecycle_statuses",
        "source_filter",
        "source_kind",
        "source_kinds",
        "status",
        "statuses",
        "workspace_id",
    }
)


class PackedVectorCodec:
    """Encode and decode normalized little-endian float32 vectors."""

    @staticmethod
    def _validated_array(
        values: Sequence[float] | np.ndarray,
        dim: int,
        *,
        context: str,
    ) -> np.ndarray:
        if dim < 1:
            raise ValueError(f"{context} dimension must be positive")
        try:
            raw = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{context} must be a one-dimensional numeric vector") from exc
        if raw.ndim != 1 or raw.shape[0] != dim:
            actual = raw.shape[0] if raw.ndim == 1 else "non-1d"
            raise ValueError(
                f"{context} dimension mismatch: expected {dim}, got {actual}"
            )
        if not np.isfinite(raw).all():
            raise ValueError(f"{context} must contain only finite values")
        vector = np.asarray(raw, dtype="<f4")
        if not np.isfinite(vector).all():
            raise ValueError(f"{context} cannot be represented as finite float32 values")
        norm = float(np.linalg.norm(vector.astype(np.float64)))
        if not np.isfinite(norm) or norm == 0.0:
            raise ValueError(f"{context} must have a non-zero finite norm")
        normalized = np.asarray(vector / norm, dtype="<f4")
        if not np.isfinite(normalized).all():
            raise ValueError(f"{context} normalization produced non-finite values")
        return np.ascontiguousarray(normalized)

    @classmethod
    def encode(
        cls,
        values: Sequence[float] | np.ndarray,
        dim: int,
        *,
        context: str = "embedding",
    ) -> bytes:
        return cls._validated_array(values, dim, context=context).tobytes()

    @classmethod
    def normalize(
        cls,
        values: Sequence[float] | np.ndarray,
        dim: int,
        *,
        context: str = "vector",
    ) -> np.ndarray:
        return cls._validated_array(values, dim, context=context)

    @staticmethod
    def decode(
        payload: bytes | bytearray | memoryview,
        dim: int,
        *,
        context: str = "stored embedding",
    ) -> np.ndarray:
        if dim < 1:
            raise ValueError(f"{context} dimension must be positive")
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise ValueError(f"{context} must be packed bytes")  # noqa: TRY004
        expected_size = dim * np.dtype("<f4").itemsize
        if len(payload) != expected_size:
            raise ValueError(
                f"{context} byte length mismatch: expected {expected_size}, got {len(payload)}"
            )
        vector = np.frombuffer(payload, dtype="<f4").copy()
        if not np.isfinite(vector).all():
            raise ValueError(f"{context} must contain only finite values")
        norm = float(np.linalg.norm(vector.astype(np.float64)))
        if not np.isfinite(norm) or norm == 0.0:
            raise ValueError(f"{context} must have a non-zero finite norm")
        return np.ascontiguousarray(vector)

    @classmethod
    def migrate_json(
        cls,
        payload: Any,
        dim: int,
        *,
        context: str = "embedding",
    ) -> bytes:
        if not isinstance(payload, str):
            raise ValueError(f"{context} must be a JSON array")  # noqa: TRY004
        try:
            values = json.loads(payload)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"{context} contains malformed JSON") from exc
        if not isinstance(values, list):
            raise ValueError(f"{context} must be a JSON array")  # noqa: TRY004
        return cls.encode(values, dim, context=context)


@dataclass(frozen=True, slots=True)
class VectorSnapshot:
    """Immutable exact-search state for one encoder namespace."""

    encoder_namespace: str
    dim: int
    epoch: int
    matrix: np.ndarray
    storage_keys: tuple[str, ...]
    source_ids: np.ndarray
    workspace_ids: np.ndarray
    source_kinds: np.ndarray
    statuses: np.ndarray
    metadata: tuple[dict[str, Any], ...]
    uris: tuple[str | None, ...]
    _position_by_key: dict[str, int] | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def _position_index(self) -> dict[str, int]:
        """Build (and cache) a storage_key -> row position lookup."""
        cached = self._position_by_key
        if cached is None:
            cached = {key: index for index, key in enumerate(self.storage_keys)}
            object.__setattr__(self, "_position_by_key", cached)
        return cached

    @classmethod
    def from_rows(
        cls,
        rows: Sequence[Any],
        *,
        encoder_namespace: str,
        dim: int,
        epoch: int,
        materialize_metadata: bool = False,
    ) -> VectorSnapshot:
        storage_keys: list[str] = []
        source_ids: list[str] = []
        workspace_ids: list[str | None] = []
        source_kinds: list[str] = []
        statuses: list[str] = []
        metadata: list[dict[str, Any]] = []
        uris: list[str | None] = []
        payloads: list[bytes] = []
        expected_size = dim * np.dtype("<f4").itemsize
        for row in rows:
            storage_key = row["storage_key"]
            if (
                row["format_version"] != VECTOR_FORMAT_VERSION
                or row["normalization_policy"] != NORMALIZATION_POLICY
            ):
                raise ValueError(f"unsupported vector format for {storage_key}")
            payload = row["embedding"]
            if not isinstance(payload, (bytes, bytearray, memoryview)):
                raise ValueError(  # noqa: TRY004
                    f"stored embedding for {storage_key} must be packed bytes"
                )
            if len(payload) != expected_size:
                raise ValueError(
                    f"stored embedding for {storage_key} byte length mismatch: "
                    f"expected {expected_size}, got {len(payload)}"
                )
            payloads.append(bytes(payload))
            storage_keys.append(storage_key)
            source_ids.append(row["source_id"])
            workspace_ids.append(row["workspace_id"])
            source_kinds.append(row["source_kind"])
            statuses.append(row["status"])
            if materialize_metadata:
                metadata.append(dict(metadata_mapping(row["metadata"])))
                uris.append(row["uri"])
        # np.frombuffer over `bytes` already yields a non-writeable view, and
        # that view keeps the joined buffer alive via its `base` reference,
        # so the trailing full-matrix `.copy()` would only duplicate memory
        # (~30MB at 100k x 384) without changing safety or read-only-ness.
        matrix = np.frombuffer(b"".join(payloads), dtype="<f4").reshape(
            len(payloads), dim
        )
        finite_rows = np.isfinite(matrix).all(axis=1)
        if not finite_rows.all():
            bad_row = int(np.flatnonzero(~finite_rows)[0])
            raise ValueError(
                f"stored embedding for {storage_keys[bad_row]} must contain only "
                "finite values"
            )
        norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
        bad_norm = ~np.isfinite(norms) | (norms == 0.0)
        if bad_norm.any():
            bad_row = int(np.flatnonzero(bad_norm)[0])
            raise ValueError(
                f"stored embedding for {storage_keys[bad_row]} must have a "
                "non-zero finite norm"
            )
        arrays = [
            np.asarray(values, dtype=object)
            for values in (source_ids, workspace_ids, source_kinds, statuses)
        ]
        matrix.setflags(write=False)
        for array in arrays:
            array.setflags(write=False)
        return cls(
            encoder_namespace=encoder_namespace,
            dim=dim,
            epoch=epoch,
            matrix=matrix,
            storage_keys=tuple(storage_keys),
            source_ids=arrays[0],
            workspace_ids=arrays[1],
            source_kinds=arrays[2],
            statuses=arrays[3],
            metadata=tuple(metadata),
            uris=tuple(uris),
        )

    def filter_mask(
        self,
        filters: Mapping[str, Any] | None,
        *,
        status_values: set[str],
        filter_values: Any,
    ) -> np.ndarray:
        predicate = compile_vector_filters(filters)
        # Vectorized prefilter on scalar fields always runs first, even when
        # a Python metadata predicate is also needed below: it is cheap and
        # narrows the rows the slow per-row predicate must visit.
        eligible = np.isin(self.statuses, tuple(predicate.statuses))
        if predicate.workspace_id is not None and isinstance(
            predicate.workspace_id, str
        ):
            eligible &= self.workspace_ids == predicate.workspace_id
        if predicate.source_kinds is not None:
            eligible &= np.isin(self.source_kinds, tuple(predicate.source_kinds))
        if predicate.candidate_keys is not None:
            position_index = self._position_index()
            candidate_mask = np.zeros(len(self.storage_keys), dtype=bool)
            for candidate_key in predicate.candidate_keys:
                position = position_index.get(candidate_key)
                if position is not None:
                    candidate_mask[position] = True
            eligible &= candidate_mask
        if self._can_prefilter_scalars(filters, predicate):
            return np.asarray(eligible, dtype=bool)
        if len(self.metadata) != len(self.storage_keys):
            raise ValueError("metadata must be materialized for this filter")
        result = np.zeros(len(self.storage_keys), dtype=bool)
        for position in np.flatnonzero(eligible):
            workspace_id = self.workspace_ids[position]
            result[position] = predicate.matches(
                storage_key=self.storage_keys[position],
                source_id=str(self.source_ids[position]),
                workspace_id=(str(workspace_id) if workspace_id is not None else None),
                source_kind=str(self.source_kinds[position]),
                status=str(self.statuses[position]),
                metadata=self.metadata[position],
                uri=self.uris[position],
            )
        return result

    @staticmethod
    def _can_prefilter_scalars(
        filters: Mapping[str, Any] | None,
        predicate: CompiledVectorFilter,
    ) -> bool:
        filter_names = set(filters or {})
        return (
            filter_names.issubset(_SCALAR_FILTER_NAMES)
            and (
                predicate.workspace_id is None
                or isinstance(predicate.workspace_id, str)
            )
            and predicate.project_values is None
            and predicate.excluded_projects is None
            and predicate.included_paths is None
            and predicate.excluded_paths is None
            and predicate.document_values is None
            and predicate.excluded_documents is None
            and predicate.metadata_equals is None
            and not predicate.source_scoped_filters
        )
