"""
Source-agnostic search/indexing kernel.

A domain-agnostic library for building hybrid vector+keyword+graph search systems
with pluggable embedding/LLM/reranker providers.
"""

from searchkernel.domain import Record, ScoredRef, SearchResult
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
    "ContentSource",
    "EmbeddingProvider",
    "QueryEmbeddingProvider",
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
    "classify_query_type",
    "truncate_content",
]
