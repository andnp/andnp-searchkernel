"""Canonical record-oriented search orchestration."""

import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from searchkernel.domain import Record, RecordIdentity, Vector
from searchkernel.ports import (
    AsyncEmbeddingProvider,
    AsyncGraphStore,
    AsyncKeywordStore,
    AsyncVectorStore,
    EmbeddingProvider,
    GraphStore,
    KeywordStore,
    VectorStore,
)
from searchkernel.search.base_orchestrator import BaseSearchOrchestrator
from searchkernel.search.record_pipeline import (
    QueryEmbeddingProvider,
    RecordHydrator,
    RecordSearchConfig,
    RecordSearchOutcome,
    RecordSearchPipeline,
    RecordSearchPolicy,
)


class SearchOrchestrator(BaseSearchOrchestrator):
    """Drive the one supported local search pipeline.

    The old chunk/file query pipeline intentionally is not adapted here:
    callers provide record stores and a record hydrator directly.
    """

    def __init__(
        self,
        *,
        pipeline: RecordSearchPipeline | None = None,
        hydrator: (
            RecordHydrator
            | Callable[[RecordIdentity | str], Record | None | Awaitable[Record | None]]
            | None
        ) = None,
        keyword_store: KeywordStore | AsyncKeywordStore | None = None,
        vector_store: VectorStore | AsyncVectorStore | None = None,
        graph_store: GraphStore | AsyncGraphStore | None = None,
        embedding_provider: (
            EmbeddingProvider
            | AsyncEmbeddingProvider
            | QueryEmbeddingProvider
            | Callable[[str], Vector | Awaitable[Vector]]
            | None
        ) = None,
        embedding_model_name: str | None = None,
        embedding_dim: int | None = None,
        policy: RecordSearchPolicy | None = None,
        config: RecordSearchConfig | None = None,
    ) -> None:
        if pipeline is not None:
            if any(
                dependency is not None
                for dependency in (
                    hydrator,
                    keyword_store,
                    vector_store,
                    graph_store,
                    embedding_provider,
                    policy,
                    config,
                )
            ):
                raise ValueError("pipeline cannot be combined with search dependencies")
            super().__init__(pipeline)
            return
        if hydrator is None:
            raise ValueError("hydrator is required")
        super().__init__(
            RecordSearchPipeline(
                hydrator=hydrator,
                keyword_store=keyword_store,
                vector_store=vector_store,
                graph_store=graph_store,
                embedding_provider=embedding_provider,
                embedding_model_name=embedding_model_name,
                embedding_dim=embedding_dim,
                policy=policy,
                config=config,
            )
        )

    async def search(
        self,
        query: str,
        *,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> RecordSearchOutcome:
        result = self.record_pipeline.search(query, limit=limit, filters=filters)
        if inspect.isawaitable(result):
            return await result
        return result


__all__ = ["SearchOrchestrator"]
