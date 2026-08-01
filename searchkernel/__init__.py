"""
Source-agnostic search/indexing kernel.

A domain-agnostic library for building hybrid vector+keyword+graph search systems
with pluggable embedding/LLM/reranker providers.

Record-oriented retrieval is the supported search path. The deprecated
chunk-oriented execution path is no longer exported.
"""

from searchkernel.domain import (
    ActiveModelMetadata,
    BackupMetadata,
    Chunk,
    MigrationPhase,
    MigrationState,
    ModelDimensionMismatchError,
    ModelNamespace,
    Record,
    RollbackMetadata,
    ScoredRef,
    SearchResult,
    ValidationResult,
    canonical_storage_key,
)
from searchkernel.indexing.coordinator import (
    CoordinatorProgress,
    CoordinatorReceipt,
    ResumableSemanticCoordinator,
)
from searchkernel.kernel import SearchKernel
from searchkernel.ports import (
    ActiveModelStore,
    BatchContentSource,
    ContentSource,
    EmbeddingProvider,
    ModelBackupStore,
    ModelLifecycleStore,
    ModelNamespaceStore,
    ModelValidationStore,
    RecordIngestor,
    SearchableSource,
    SearchAPI,
    SourceBatch,
)
from searchkernel.search.orchestrator import SearchOrchestrator
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
    "ActiveModelMetadata",
    "ActiveModelStore",
    "BackupMetadata",
    "BatchContentSource",
    "Chunk",
    "ContentSource",
    "CoordinatorProgress",
    "CoordinatorReceipt",
    "EmbeddingProvider",
    "MigrationPhase",
    "MigrationState",
    "ModelBackupStore",
    "ModelDimensionMismatchError",
    "ModelLifecycleStore",
    "ModelNamespace",
    "ModelNamespaceStore",
    "ModelValidationStore",
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
    "ResumableSemanticCoordinator",
    "RollbackMetadata",
    "ScoredRef",
    "SearchAPI",
    "SearchKernel",
    "SearchOrchestrator",
    "SearchResult",
    "SearchableSource",
    "SourceBatch",
    "ValidationResult",
    "canonical_storage_key",
    "classify_query_type",
    "route_query",
    "truncate_content",
]
