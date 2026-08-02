"""Typed mutation epoch contracts for search adapters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

_EPOCH_LANES = ("keyword", "vector", "graph")


def _validated_epoch(lane: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{lane} epoch must be an integer")
    if value < 0:
        raise ValueError(f"{lane} epoch must not be negative")
    return value


@dataclass(frozen=True, slots=True)
class SearchEpochs:
    """Authoritative mutation versions for the search lanes."""

    keyword: int = 0
    vector: int = 0
    graph: int = 0

    def __post_init__(self) -> None:
        for lane in _EPOCH_LANES:
            _validated_epoch(lane, getattr(self, lane))

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> SearchEpochs:
        """Build a complete snapshot from an adapter-provided mapping."""
        missing = [lane for lane in _EPOCH_LANES if lane not in values]
        if missing:
            raise ValueError(
                f"epoch snapshot is missing {', '.join(missing)} lane(s)"
            )
        return cls(
            keyword=_validated_epoch("keyword", values["keyword"]),
            vector=_validated_epoch("vector", values["vector"]),
            graph=_validated_epoch("graph", values["graph"]),
        )

    def for_lane(self, lane: str) -> int:
        """Return one lane's epoch, rejecting unknown lane names."""
        if lane not in _EPOCH_LANES:
            raise ValueError(f"unknown search epoch lane: {lane}")
        return getattr(self, lane)


@runtime_checkable
class SearchEpochProvider(Protocol):
    """Adapter contract for a complete, authoritative epoch snapshot.

    The legacy ``epochs()`` method is retained as the stable adapter method;
    implementations must return all three lanes in the mapping.
    """

    def epochs(self) -> SearchEpochs | Mapping[str, int]:
        ...
