"""Compact vector storage primitives for the local record backend."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

VECTOR_FORMAT_VERSION = 2
NORMALIZATION_POLICY = "l2"


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
        context: str = "legacy embedding",
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

    @classmethod
    def from_rows(
        cls,
        rows: Sequence[Any],
        *,
        encoder_namespace: str,
        dim: int,
        epoch: int,
    ) -> VectorSnapshot:
        vectors: list[np.ndarray] = []
        storage_keys: list[str] = []
        source_ids: list[str] = []
        workspace_ids: list[str | None] = []
        source_kinds: list[str] = []
        statuses: list[str] = []
        for row in rows:
            if (
                row["format_version"] != VECTOR_FORMAT_VERSION
                or row["normalization_policy"] != NORMALIZATION_POLICY
            ):
                raise ValueError(f"unsupported vector format for {row['storage_key']}")
            vectors.append(
                PackedVectorCodec.decode(
                    row["embedding"],
                    dim,
                    context=f"stored embedding for {row['storage_key']}",
                )
            )
            storage_keys.append(row["storage_key"])
            source_ids.append(row["source_id"])
            workspace_ids.append(row["workspace_id"])
            source_kinds.append(row["source_kind"])
            statuses.append(row["status"])
        matrix = (
            np.ascontiguousarray(np.vstack(vectors), dtype="<f4")
            if vectors
            else np.empty((0, dim), dtype="<f4")
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
        )

    def filter_mask(
        self,
        filters: dict[str, Any] | None,
        *,
        status_values: set[str],
        filter_values: Any,
    ) -> np.ndarray:
        filters = filters or {}
        mask = np.isin(self.statuses, tuple(status_values))
        workspace_id = filters.get("workspace_id")
        if workspace_id is not None:
            mask &= self.workspace_ids == workspace_id
        source_kinds = filters.get("source_kinds")
        if source_kinds is None and filters.get("source_kind") is not None:
            source_kinds = [filters["source_kind"]]
        if source_kinds is not None:
            values = filter_values(source_kinds)
            if not values:
                return np.zeros(len(self.storage_keys), dtype=bool)
            mask &= np.isin(self.source_kinds, tuple(str(value) for value in values))
        candidate_ids = filters.get("candidate_ids")
        if candidate_ids is None:
            candidate_ids = filters.get("candidate_storage_keys")
        if candidate_ids is not None:
            values = {str(value) for value in filter_values(candidate_ids)}
            if not values:
                return np.zeros(len(self.storage_keys), dtype=bool)
            mask &= np.asarray(
                [
                    storage_key in values or source_id in values
                    for storage_key, source_id in zip(
                        self.storage_keys, self.source_ids, strict=True
                    )
                ],
                dtype=bool,
            )
        return mask
