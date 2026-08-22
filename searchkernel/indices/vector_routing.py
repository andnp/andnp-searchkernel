"""Adaptive routing state for local vector engines."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

VectorEngineName = Literal["sqlite-exact", "faiss"]


@dataclass(frozen=True, slots=True)
class VectorRouteKey:
    """Identify one machine-local routing calibration scope."""

    model_name: str
    dim: int
    vector_epoch: int
    filter_shape: str


@dataclass(frozen=True, slots=True)
class VectorRouteMeasurement:
    """Record one exact-engine comparison and its selected winner."""

    sqlite_ms: float
    faiss_ms: float
    selected: VectorEngineName


class AdaptiveVectorRouter:
    """Cache exact-engine winners for the lifetime of one vector store."""

    def __init__(self) -> None:
        self._measurements: dict[VectorRouteKey, VectorRouteMeasurement] = {}

    def get(self, key: VectorRouteKey) -> VectorRouteMeasurement | None:
        return self._measurements.get(key)

    def record(
        self,
        key: VectorRouteKey,
        *,
        sqlite_ms: float,
        faiss_ms: float,
    ) -> VectorRouteMeasurement:
        selected: VectorEngineName = (
            "faiss" if faiss_ms < sqlite_ms else "sqlite-exact"
        )
        measurement = VectorRouteMeasurement(sqlite_ms, faiss_ms, selected)
        self._measurements[key] = measurement
        return measurement

    @staticmethod
    def filter_shape(filters: Mapping[str, object] | None) -> str:
        """Classify filters by their expected engine-cost shape."""
        if not filters:
            return "unfiltered"
        if filters.get("candidate_ids") is not None or filters.get(
            "candidate_storage_keys"
        ) is not None:
            return "candidate"
        return "filtered"


__all__ = ["AdaptiveVectorRouter", "VectorRouteKey", "VectorRouteMeasurement"]
