"""Port/Protocol interfaces for the search kernel.

Ports define the contracts between the kernel core and the outside world.
They are purely abstract (Protocol or ABC); implementations live in adapters/.

Dependency rule: ports import only from domain/ and stdlib/typing.
"""

from searchkernel.domain import GraphEdge, GraphNeighbor, RecordHit, SearchFilters
from searchkernel.ports.candidate_filter import CandidateFilterSupport
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
    EmbeddingProvider,
    EmbeddingSink,
)
from searchkernel.ports.epochs import SearchEpochs
from searchkernel.ports.index_manager import IndexManagerPort
from searchkernel.ports.llm import LLMProvider
from searchkernel.ports.orchestrator_config import OrchestratorConfig
from searchkernel.ports.record_search import (
    AsyncRecordHydrator,
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
from searchkernel.ports.stores import (
    AsyncGraphStore,
    AsyncKeywordStore,
    AsyncVectorStore,
    BatchGraphStore,
    CacheStore,
    GraphStore,
    KeywordStore,
    VectorStore,
)

__all__ = [
    "ActiveModelStore",
    "AsyncEmbeddingProvider",
    "AsyncGraphStore",
    "AsyncKeywordStore",
    "AsyncRecordHydrator",
    "AsyncRecordIngestor",
    "AsyncVectorStore",
    "BatchContentSource",
    "BatchGraphStore",
    "BatchRecordHydrator",
    "CacheStore",
    "CandidateFilterSupport",
    "CheckpointStore",
    "ChunkTuningConfig",
    "ContentSource",
    "EmbeddingBatchProvider",
    "EmbeddingProvider",
    "EmbeddingSink",
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
    "RecordHit",
    "RecordIngestionResult",
    "RecordIngestionStatus",
    "RecordIngestor",
    "Reranker",
    "RetrievalFieldExtractor",
    "RetrievalFields",
    "SearchAPI",
    "SearchEpochs",
    "SearchFilters",
    "SourceBatch",
    "SourceCapabilities",
    "VectorStore",
    "extract_retrieval_fields",
]
