"""
Source-agnostic search/indexing kernel.

A domain-agnostic library for building hybrid vector+keyword+graph search systems
with pluggable embedding/LLM/reranker providers.
"""

from searchkernel.domain import (
    Chunk,
    Record,
    ScoredRef,
    SearchResult,
    canonical_storage_key,
)
from searchkernel.kernel import SearchKernel
from searchkernel.ports import (
    ContentSource,
    EmbeddingProvider,
    RecordIngestor,
    SearchableSource,
    SearchAPI,
)
from searchkernel.search.orchestrator import SearchOrchestrator
from searchkernel.search.pipeline import SearchPipelineConfig
from searchkernel.search.query_plan import (
    QueryPlan,
    QueryRouter,
    QueryRouterConfig,
    route_query,
)
from searchkernel.search.record_pipeline import (
    QueryEmbeddingProvider,
    RecordHydrator,
    RecordSearchCandidate,
    RecordSearchConfig,
    RecordSearchError,
    RecordSearchFailure,
    RecordSearchOutcome,
    RecordSearchPipeline,
    RecordSearchPolicy,
    RecordSearchResult,
)
from searchkernel.search.utils import classify_query_type, truncate_content

__version__ = "0.1.0"

__all__ = [
    "Chunk",
    "ContentSource",
    "EmbeddingProvider",
    "QueryEmbeddingProvider",
    "QueryPlan",
    "QueryRouter",
    "QueryRouterConfig",
    "Record",
    "RecordHydrator",
    "RecordIngestor",
    "RecordSearchCandidate",
    "RecordSearchConfig",
    "RecordSearchError",
    "RecordSearchFailure",
    "RecordSearchOutcome",
    "RecordSearchPipeline",
    "RecordSearchPolicy",
    "RecordSearchResult",
    "ScoredRef",
    "SearchAPI",
    "SearchKernel",
    "SearchOrchestrator",
    "SearchPipelineConfig",
    "SearchResult",
    "SearchableSource",
    "canonical_storage_key",
    "classify_query_type",
    "route_query",
    "truncate_content",
]
