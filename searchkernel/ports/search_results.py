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
MAX_FAILURE_DETAIL_LENGTH = 256
DiagnosticAvailability = Literal["available", "unavailable"]


class SearchTrace(Protocol):
    """Minimal trace interface carried by a search outcome."""

    @property
    def total_duration_ms(self) -> float | None: ...

    def close(self) -> None: ...

    def to_dict(self) -> dict[str, object]: ...


@dataclass(frozen=True, slots=True)
class RecordSearchResult:
    """A ranked, hydrated record with reusable kernel provenance.

    ``score`` is the raw query score. ``normalized_score`` is query-relative:
    it compares this result with the other returned results for the same
    query and must not be interpreted as a cross-query probability.
    """

    record: Record
    score: float
    provenance: SearchResultProvenance
    normalized_score: float = 0.0
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
    detail: str | None = None

    def __post_init__(self) -> None:
        if self.detail is not None:
            object.__setattr__(
                self,
                "detail",
                self.detail[:MAX_FAILURE_DETAIL_LENGTH],
            )


@dataclass(frozen=True, slots=True)
class SearchDiagnosticSkip:
    """A lane or stage that was intentionally not executed."""

    lane: str
    reason: str


@dataclass(frozen=True, slots=True)
class DiagnosticCapability:
    """Whether a diagnostic capability produced evidence for this search."""

    state: DiagnosticAvailability
    reason: str | None = None
    count: int | None = None

    @property
    def available(self) -> bool:
        return self.state == "available"


@dataclass(frozen=True, slots=True)
class RecordSearchDiagnostics:
    """Stable, provider-neutral evidence about one record search."""

    enabled_lanes: tuple[str, ...] = ()
    lane_budgets: Mapping[str, int] = field(default_factory=dict)
    skipped_lanes: tuple[SearchDiagnosticSkip, ...] = ()
    failures: tuple[RecordSearchFailure, ...] = ()
    missing_record_ids: tuple[str, ...] = ()
    stage_timings_ms: Mapping[str, float] = field(default_factory=dict)
    result_provenance: Mapping[str, tuple[str, ...]] = field(
        default_factory=dict
    )
    final_duplicate_count: int = 0
    raw_pre_fusion_overlap: DiagnosticCapability = field(
        default_factory=lambda: DiagnosticCapability(
            state="unavailable",
            reason="raw pre-fusion overlap is not retained by the pipeline",
        )
    )

    @property
    def degraded(self) -> bool:
        return bool(self.failures or self.missing_record_ids)


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
    diagnostic_evidence: RecordSearchDiagnostics | None = None

    @property
    def degraded(self) -> bool:
        return bool(self.failures or self.missing_record_ids)


__all__ = [
    "MAX_FAILURE_DETAIL_LENGTH",
    "DiagnosticAvailability",
    "DiagnosticCapability",
    "FailureStage",
    "RecordSearchDiagnostics",
    "RecordSearchFailure",
    "RecordSearchOutcome",
    "RecordSearchResult",
    "SearchDiagnosticSkip",
    "SearchTrace",
]
