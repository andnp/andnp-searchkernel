"""Public composition helpers for the canonical local record backend."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from searchkernel.domain import Vector
from searchkernel.indices import (
    LocalGraphStore,
    LocalKeywordStore,
    LocalRecordBackend,
    LocalVectorStore,
)
from searchkernel.kernel import SearchKernel
from searchkernel.ports.embedding import AsyncEmbeddingProvider, EmbeddingProvider
from searchkernel.ports.rerank import Reranker
from searchkernel.search.orchestrator import SearchOrchestrator
from searchkernel.search.record_pipeline import (
    QueryEmbeddingProvider,
    RecordSearchConfig,
    RecordSearchPipeline,
    RecordSearchPolicy,
)

_DEFAULT_VECTOR_SNAPSHOT_MAX_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class LocalRecordKernel:
    """The canonical local stores and the kernel that searches them."""

    backend: LocalRecordBackend
    vector_store: LocalVectorStore
    keyword_store: LocalKeywordStore
    graph_store: LocalGraphStore
    pipeline: RecordSearchPipeline
    kernel: SearchKernel

    def close(self) -> None:
        """Release resources owned by the local composition."""
        self.backend.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()


def build_local_record_kernel(
    db_path: Path | None = None,
    *,
    embedding_provider: (
        EmbeddingProvider
        | AsyncEmbeddingProvider
        | QueryEmbeddingProvider
        | Callable[[str], Vector | Awaitable[Vector]]
    ),
    embedding_model_name: str | None = None,
    embedding_dim: int | None = None,
    vector_engine: str = "exact",
    vector_snapshot_max_rows: int = 100_000,
    vector_snapshot_max_bytes: int = _DEFAULT_VECTOR_SNAPSHOT_MAX_BYTES,
    faiss_path: Path | None = None,
    reranker: Reranker | None = None,
    search_policy: RecordSearchPolicy | None = None,
    search_config: RecordSearchConfig | None = None,
) -> LocalRecordKernel:
    """Build the durable local record stores and their search kernel."""

    backend = LocalRecordBackend(
        db_path,
        vector_engine=vector_engine,
        vector_snapshot_max_rows=vector_snapshot_max_rows,
        vector_snapshot_max_bytes=vector_snapshot_max_bytes,
    )
    vector_store = LocalVectorStore(backend, faiss_path=faiss_path)
    keyword_store = LocalKeywordStore(backend)
    graph_store = LocalGraphStore(backend)
    pipeline = RecordSearchPipeline(
        hydrator=backend,
        keyword_store=keyword_store,
        vector_store=vector_store,
        graph_store=graph_store,
        embedding_provider=embedding_provider,
        embedding_model_name=embedding_model_name,
        embedding_dim=embedding_dim,
        reranker=reranker,
        policy=search_policy,
        config=search_config,
    )
    kernel = SearchKernel.build(
        orchestrator=SearchOrchestrator(pipeline=pipeline),
    )
    return LocalRecordKernel(
        backend=backend,
        vector_store=vector_store,
        keyword_store=keyword_store,
        graph_store=graph_store,
        pipeline=pipeline,
        kernel=kernel,
    )


__all__ = ["LocalRecordKernel", "build_local_record_kernel"]
