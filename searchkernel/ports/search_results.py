"""Contracts returned by the canonical record-oriented search path."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol

from searchkernel.domain import ChunkResult, Record, SearchResultProvenance

FailureStage = Literal[
    "keyword",
    "vector",
    "graph",
    "parent_expansion",
    "hydration",
    "rerank",
]


class SearchTrace(Protocol):
    """Minimal trace interface carried by a search outcome."""

    total_duration_ms: float | None

    def close(self) -> None: ...

    def to_dict(self) -> dict[str, object]: ...


@dataclass(frozen=True, slots=True)
class RecordSearchResult:
    """A ranked, hydrated record with reusable kernel provenance."""

    record: Record
    score: float
    provenance: SearchResultProvenance
    chunk_matches: tuple[ChunkResult, ...] = ()

    @property
    def record_id(self) -> str:
        return self.record.source_id

    @property
    def storage_key(self) -> str:
        return self.record.storage_key

    @property
    def excerpts(self) -> tuple[ChunkResult, ...]:
        """Return the best chunk excerpts contributing to this result."""
        return self.chunk_matches


@dataclass(frozen=True, slots=True)
class RecordSearchFailure:
    """A source or hydration failure captured in degraded mode."""

    stage: FailureStage
    message: str
    exception_type: str = "Exception"


@dataclass(frozen=True, slots=True)
class RecordSearchOutcome:
    """Search results plus explicit degradation diagnostics."""

    results: tuple[RecordSearchResult, ...] = ()
    failures: tuple[RecordSearchFailure, ...] = ()
    missing_record_ids: tuple[str, ...] = ()
    cache_diagnostics: tuple[str, ...] = ()
    diagnostics: tuple[str, ...] = ()
    candidate_count: int = 0
    candidate_counts: Mapping[str, int] = field(default_factory=dict)
    stage_timings_ms: Mapping[str, float] = field(default_factory=dict)
    trace: SearchTrace | None = None

    @property
    def degraded(self) -> bool:
        return bool(self.failures or self.missing_record_ids)


__all__ = [
    "FailureStage",
    "RecordSearchFailure",
    "RecordSearchOutcome",
    "RecordSearchResult",
    "SearchTrace",
]
