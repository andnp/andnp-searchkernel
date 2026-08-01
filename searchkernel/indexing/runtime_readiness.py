"""Readiness and availability tracking for progressive semantic indexing."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Literal, cast

type IndexStatus = Literal["uninitialized", "indexing", "partial", "ready", "failed"]

type ReadinessStage = Literal[
    "unavailable", "available", "backfilling", "complete"
]

IndexAvailability = ReadinessStage
SemanticAvailability = ReadinessStage

_READINESS_STAGES = frozenset(
    ("available", "unavailable", "backfilling", "complete")
)
_SERVING_STAGES = frozenset(("available", "backfilling", "complete"))
_READY_STAGES = frozenset(("available", "complete"))


@dataclass(frozen=True, slots=True)
class SearchAvailability:
    """Capabilities currently available to the search layer.

    Lexical and graph capabilities form the minimum serving contract. Semantic
    tiers may be usable while still backfilling; only completed tiers count
    toward full readiness.
    """

    lexical: IndexAvailability
    graph: IndexAvailability
    semantic_coarse: SemanticAvailability
    semantic_fine: SemanticAvailability

    def __post_init__(self) -> None:
        if self.lexical not in _READINESS_STAGES:
            raise ValueError(f"invalid lexical availability: {self.lexical!r}")
        if self.graph not in _READINESS_STAGES:
            raise ValueError(f"invalid graph availability: {self.graph!r}")
        if self.semantic_coarse not in _READINESS_STAGES:
            raise ValueError(
                f"invalid semantic_coarse availability: {self.semantic_coarse!r}"
            )
        if self.semantic_fine not in _READINESS_STAGES:
            raise ValueError(
                f"invalid semantic_fine availability: {self.semantic_fine!r}"
            )

    def can_serve_queries(self) -> bool:
        """Return whether lexical and graph search can serve partial results."""
        return (
            self.lexical in _SERVING_STAGES
            and self.graph in _SERVING_STAGES
        )

    def is_fully_ready(self) -> bool:
        """Return whether all required search stages are ready.

        ``available`` means a stage has no work pending (for example, an empty
        corpus). ``backfilling`` and ``unavailable`` both keep full readiness
        false, while allowing lexical/graph serving when those stages exist.
        """
        return (
            self.lexical in _READY_STAGES
            and self.graph in _READY_STAGES
            and self.semantic_coarse in _READY_STAGES
            and self.semantic_fine in _READY_STAGES
        )

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-compatible representation of the capabilities."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> SearchAvailability:
        """Restore a durable availability snapshot."""
        return cls(
            lexical=_readiness_stage(data, "lexical"),
            graph=_readiness_stage(data, "graph"),
            semantic_coarse=_readiness_stage(data, "semantic_coarse"),
            semantic_fine=_readiness_stage(data, "semantic_fine"),
        )


def _readiness_stage(data: Mapping[str, object], key: str) -> ReadinessStage:
    value = data[key]
    if not isinstance(value, str) or value not in _READINESS_STAGES:
        raise ValueError(f"invalid {key} availability: {value!r}")
    return cast(ReadinessStage, value)


def can_serve_queries(
    *,
    init_error: Exception | None,
    ready_event_set: bool,
    is_virgin_startup: bool,
    indices_queryable: bool,
    availability: SearchAvailability | None = None,
) -> bool:
    if init_error is not None:
        return False
    if not indices_queryable:
        return False
    if availability is not None and not availability.can_serve_queries():
        return False
    if ready_event_set:
        return True
    return not is_virgin_startup


def is_fully_ready(
    *,
    init_error: Exception | None,
    ready_event_set: bool,
    index_status: IndexStatus,
    indices_queryable: bool,
    availability: SearchAvailability | None = None,
) -> bool:
    return (
        ready_event_set
        and init_error is None
        and index_status == "ready"
        and indices_queryable
        and (availability is None or availability.is_fully_ready())
    )


def can_refresh_loaded_indices(
    *,
    ready_event_set: bool,
    init_error: Exception | None,
) -> bool:
    return ready_event_set and init_error is None


def semantic_tier_from_progress(
    indexed_count: int, total_count: int
) -> SemanticAvailability:
    """Derive semantic tier state from indexing progress.

    In main's synchronous architecture, a document is only marked "indexed"
    after the full pipeline including embeddings. So the IndexState counts
    directly reflect semantic work progress.

    Args:
        indexed_count: Number of fully indexed documents (including embeddings)
        total_count: Total number of documents to index

    Returns:
        SemanticAvailability tier state
    """
    if total_count == 0:
        # Nothing to index yet, no work pending
        return "available"
    if indexed_count >= total_count:
        # All documents fully indexed including embeddings
        return "complete"
    # 0 < indexed_count < total_count: still indexing
    return "backfilling"


__all__ = [
    "IndexAvailability",
    "IndexStatus",
    "ReadinessStage",
    "SearchAvailability",
    "SemanticAvailability",
    "can_refresh_loaded_indices",
    "can_serve_queries",
    "is_fully_ready",
    "semantic_tier_from_progress",
]