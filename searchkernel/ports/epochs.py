"""Typed mutation epoch contracts for search adapters."""

from __future__ import annotations

from dataclasses import dataclass

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

    def for_lane(self, lane: str) -> int:
        """Return one lane's epoch, rejecting unknown lane names."""
        if lane not in _EPOCH_LANES:
            raise ValueError(f"unknown search epoch lane: {lane}")
        return getattr(self, lane)
