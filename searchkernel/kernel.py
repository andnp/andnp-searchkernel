"""Composable search-kernel composition root and driving facade."""

from collections.abc import Iterable, Mapping
from typing import Any

from searchkernel.domain import ScoredRef, SearchResult
from searchkernel.ports.content_source import SearchableSource
from searchkernel.ports.embedding import EmbeddingProvider
from searchkernel.ports.rerank import Reranker
from searchkernel.runtime import federation
from searchkernel.runtime.federation import DEFAULT_PER_SOURCE_TIMEOUT_S
from searchkernel.runtime.local import LocalSearchSource
from searchkernel.runtime.registry import SourceRegistry
from searchkernel.search.orchestrator import SearchOrchestrator


class SearchKernel:
    """Daemon-free, source-agnostic search composition root."""

    def __init__(
        self,
        *,
        registry: SourceRegistry,
        reranker: Reranker,
        config: object | None = None,
        embedder: EmbeddingProvider | None = None,
        per_source_timeout_s: float = DEFAULT_PER_SOURCE_TIMEOUT_S,
    ) -> None:
        self._registry = registry
        self._reranker = reranker
        self._config = config
        self._embedder = embedder
        self._per_source_timeout_s = per_source_timeout_s

    @classmethod
    def build(
        cls,
        config: object | None = None,
        *,
        sources: Iterable[SearchableSource] = (),
        embedder: EmbeddingProvider | None = None,
        reranker: Reranker | None = None,
        orchestrator: SearchOrchestrator | None = None,
        registry: SourceRegistry | None = None,
        per_source_timeout_s: float = DEFAULT_PER_SOURCE_TIMEOUT_S,
    ) -> "SearchKernel":
        """Compose a kernel from source adapters and provider instances.

        ``config`` and ``embedder`` are retained as composition dependencies for
        ingestion/admin capabilities; query execution only needs the registered
        sources and reranker. A reranker may be supplied by ``config.reranker``.
        """
        effective_reranker = reranker
        if effective_reranker is None:
            if isinstance(config, Mapping):
                effective_reranker = config.get("reranker")
            elif config is not None:
                effective_reranker = getattr(config, "reranker", None)
        if effective_reranker is None:
            raise ValueError(
                "SearchKernel.build requires a reranker or config.reranker"
            )

        source_registry = registry or SourceRegistry()
        if orchestrator is not None:
            source_registry.register(LocalSearchSource(orchestrator))
        for source in sources:
            source_registry.register(source)

        effective_embedder = embedder
        if effective_embedder is None:
            if isinstance(config, Mapping):
                effective_embedder = config.get("embedder")
            elif config is not None:
                effective_embedder = getattr(config, "embedder", None)

        return cls(
            registry=source_registry,
            reranker=effective_reranker,
            config=config,
            embedder=effective_embedder,
            per_source_timeout_s=per_source_timeout_s,
        )

    @property
    def registry(self) -> SourceRegistry:
        """Return the registry used by this kernel."""
        return self._registry

    @property
    def config(self) -> object | None:
        return self._config

    @property
    def embedder(self) -> EmbeddingProvider | None:
        return self._embedder

    async def search_anything(
        self,
        query: str,
        *,
        sources: list[str] | None = None,
        filters: dict[str, Any] | None = None,
        k: int = 10,
    ) -> list[SearchResult]:
        """Search registered sources through the canonical federation point."""
        scored_refs = await federation.search_anything(
            query,
            registry=self._registry,
            reranker=self._reranker,
            sources=sources,
            top_n=k,
            per_source_k=k,
            per_source_timeout_s=self._per_source_timeout_s,
            filters=filters,
        )
        return [self._to_search_result(ref) for ref in scored_refs]

    @staticmethod
    def _to_search_result(ref: ScoredRef) -> SearchResult:
        return SearchResult(
            record_id=ref.source_id,
            score=ref.score,
            source_kind=ref.source_kind,
            metadata=dict(ref.metadata),
        )


__all__ = ["SearchKernel"]
