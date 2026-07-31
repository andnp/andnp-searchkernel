"""Compact vector storage primitives for the local record backend."""

from __future__ import annotations

import json
from collections.abc import Sequence
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
