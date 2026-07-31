"""Canonical record-oriented search APIs."""

from searchkernel.search.diversity import (
    DiversityDiagnostic,
    SourceDiversityPolicy,
    apply_source_diversity,
)
from searchkernel.search.orchestrator import SearchOrchestrator
from searchkernel.search.query_plan import (
    QueryPlan,
    QueryRouter,
    QueryRouterConfig,
    route_query,
)
from searchkernel.search.record_pipeline import RecordSearchPipeline

__all__ = [
    "DiversityDiagnostic",
    "QueryPlan",
    "QueryRouter",
    "QueryRouterConfig",
    "RecordSearchPipeline",
    "SearchOrchestrator",
    "SourceDiversityPolicy",
    "apply_source_diversity",
    "route_query",
]
