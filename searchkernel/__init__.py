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
    RecordHit,
    RecordIdentity,
    RollbackMetadata,
    SearchResultProvenance,
    ValidationResult,
    canonical_storage_key,
)
from searchkernel.indexing.coordinator import (
    CoordinatorProgress,
    CoordinatorReceipt,
    ResumableSemanticCoordinator,
)
from searchkernel.indexing.runtime_readiness import SearchAvailability
from searchkernel.kernel import SearchKernel
from searchkernel.local import LocalRecordKernel, build_local_record_kernel
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
    SearchAPI,
    SourceBatch,
)
from searchkernel.runtime.validated_read_cache import (
    ValidatedCacheEntry,
    ValidatedCacheStore,
    ValidatedCacheValue,
    ValidatedReadThroughCache,
    ValidatedReadThroughCacheMetrics,
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
    RecordSearchQueryContext,
    RecordSearchResult,
)
from searchkernel.search.utils import classify_query_type, truncate_content

__version__ = "0.5.0"

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
    "LocalRecordKernel",
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
    "RecordHit",
    "RecordIdentity",
    "RecordHydrator",
    "RecordIngestor",
    "RecordSearchCandidate",
    "RecordSearchConfig",
    "RecordSearchError",
    "RecordSearchFailure",
    "RecordSearchOutcome",
    "RecordSearchPipeline",
    "RecordSearchPolicy",
    "RecordSearchQueryContext",
    "RecordSearchResult",
    "ResumableSemanticCoordinator",
    "RollbackMetadata",
    "SearchAPI",
    "SearchAvailability",
    "SearchKernel",
    "SearchOrchestrator",
    "SearchResultProvenance",
    "SourceBatch",
    "ValidatedCacheEntry",
    "ValidatedCacheStore",
    "ValidatedCacheValue",
    "ValidatedReadThroughCache",
    "ValidatedReadThroughCacheMetrics",
    "ValidationResult",
    "canonical_storage_key",
    "build_local_record_kernel",
    "classify_query_type",
    "route_query",
    "truncate_content",
]
