"""Port/Protocol interfaces for the search kernel.

Ports define the contracts between the kernel core and the outside world.
They are purely abstract (Protocol or ABC); implementations live in adapters/.

Dependency rule: ports import only from domain/ and stdlib/typing.
"""

from searchkernel.domain import GraphEdge, GraphNeighbor, RecordHit, SearchFilters
from searchkernel.ports.candidate_filter import CandidateFilterSupport
from searchkernel.ports.chunking import RecordChunker
from searchkernel.ports.chunking_config import ChunkTuningConfig
from searchkernel.ports.content_source import (
    AsyncRecordIngestor,
    BatchContentSource,
    CheckpointStore,
    ContentSource,
    IngestionError,
    IngestionFailureMode,
    IngestionReceipt,
    RecordIngestionResult,
    RecordIngestionStatus,
    RecordIngestor,
    SourceBatch,
)
from searchkernel.ports.embedding import (
    AsyncEmbeddingProvider,
    EmbeddingBatchProvider,
    EmbeddingBatchSink,
    EmbeddingProvider,
    EmbeddingSink,
    EmbeddingWrite,
)
from searchkernel.ports.epochs import SearchEpochs
from searchkernel.ports.federation import (
    FEDERATION_CONTRACT_VERSION,
    MAX_QUERY_LENGTH,
    MAX_RERANK_TEXT_LENGTH,
    MAX_SNIPPET_LENGTH,
    MAX_TOP_K,
    CallerAuthorizationContext,
    SearchDiagnostics,
    SearchHit,
    SearchHitProvenance,
    SearchRequest,
    SearchResponse,
    SearchSource,
    SourceIdentity,
)
from searchkernel.ports.federation import (
    SourceCapabilities as FederationSourceCapabilities,
)
from searchkernel.ports.freshness import (
    FreshnessDecision,
    FreshnessPolicy,
    FreshnessProvider,
    FreshnessSnapshot,
    FreshnessStatus,
    VersionSnapshot,
    VersionToken,
    validate_fresh_hit,
)
from searchkernel.ports.graph import BatchGraphStore
from searchkernel.ports.index_manager import IndexManagerPort
from searchkernel.ports.llm import LLMProvider
from searchkernel.ports.orchestrator_config import OrchestratorConfig
from searchkernel.ports.record_search import (
    AsyncRecordHydrator,
    BatchParentRecordExpander,
    BatchRecordHydrator,
    ParentRecordExpander,
)
from searchkernel.ports.reindex import (
    ActiveModelStore,
    ModelBackupStore,
    ModelLifecycleStore,
    ModelNamespaceStore,
    ModelValidationStore,
)
from searchkernel.ports.rerank import Reranker
from searchkernel.ports.retrieval import (
    RetrievalFieldExtractor,
    RetrievalFields,
    SourceCapabilities,
    extract_retrieval_fields,
)
from searchkernel.ports.search import SearchAPI
from searchkernel.ports.search_results import (
    FailureStage,
    RecordSearchFailure,
    RecordSearchOutcome,
    RecordSearchResult,
    SearchTrace,
)
from searchkernel.ports.stores import (
    AsyncGraphStore,
    AsyncKeywordStore,
    AsyncVectorStore,
    CacheStore,
    GraphStore,
    KeywordStore,
    VectorStore,
)

__all__ = [
    "FEDERATION_CONTRACT_VERSION",
    "MAX_QUERY_LENGTH",
    "MAX_RERANK_TEXT_LENGTH",
    "MAX_SNIPPET_LENGTH",
    "MAX_TOP_K",
    "ActiveModelStore",
    "AsyncEmbeddingProvider",
    "AsyncGraphStore",
    "AsyncKeywordStore",
    "AsyncRecordHydrator",
    "AsyncRecordIngestor",
    "AsyncVectorStore",
    "BatchContentSource",
    "BatchGraphStore",
    "BatchParentRecordExpander",
    "BatchRecordHydrator",
    "CacheStore",
    "CallerAuthorizationContext",
    "CandidateFilterSupport",
    "CheckpointStore",
    "ChunkTuningConfig",
    "ContentSource",
    "EmbeddingBatchProvider",
    "EmbeddingBatchSink",
    "EmbeddingProvider",
    "EmbeddingSink",
    "EmbeddingWrite",
    "FailureStage",
    "FederationSourceCapabilities",
    "FreshnessDecision",
    "FreshnessPolicy",
    "FreshnessProvider",
    "FreshnessSnapshot",
    "FreshnessStatus",
    "GraphEdge",
    "GraphNeighbor",
    "GraphStore",
    "IndexManagerPort",
    "IngestionError",
    "IngestionFailureMode",
    "IngestionReceipt",
    "KeywordStore",
    "LLMProvider",
    "ModelBackupStore",
    "ModelLifecycleStore",
    "ModelNamespaceStore",
    "ModelValidationStore",
    "OrchestratorConfig",
    "ParentRecordExpander",
    "RecordChunker",
    "RecordHit",
    "RecordIngestionResult",
    "RecordIngestionStatus",
    "RecordIngestor",
    "RecordSearchFailure",
    "RecordSearchOutcome",
    "RecordSearchResult",
    "Reranker",
    "RetrievalFieldExtractor",
    "RetrievalFields",
    "SearchAPI",
    "SearchDiagnostics",
    "SearchEpochs",
    "SearchFilters",
    "SearchHit",
    "SearchHitProvenance",
    "SearchRequest",
    "SearchResponse",
    "SearchSource",
    "SearchTrace",
    "SourceBatch",
    "SourceCapabilities",
    "SourceIdentity",
    "VectorStore",
    "VersionSnapshot",
    "VersionToken",
    "extract_retrieval_fields",
    "validate_fresh_hit",
]
