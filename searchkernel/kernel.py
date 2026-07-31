"""Composable search-kernel composition root and driving facade."""

from collections.abc import Iterable, Mapping
from typing import Any

from searchkernel.domain import Cursor, ScoredRef, SearchResult
from searchkernel.ports.content_source import (
    ContentSource,
    RecordIngestor,
    SearchableSource,
)
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
        ingestor: RecordIngestor | None = None,
        content_sources: Iterable[ContentSource] = (),
        reranker: Reranker | None = None,
        config: object | None = None,
        embedder: EmbeddingProvider | None = None,
        per_source_timeout_s: float = DEFAULT_PER_SOURCE_TIMEOUT_S,
    ) -> None:
        self._registry = registry
        self._ingestor = ingestor
        self._content_sources: dict[str, ContentSource] = {}
        for source in content_sources:
            self.register_content_source(source)
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
        content_sources: Iterable[ContentSource] = (),
        ingestor: RecordIngestor | None = None,
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
            ingestor=ingestor,
            content_sources=content_sources,
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

    def register_content_source(self, source: ContentSource) -> None:
        """Register an ingestible source without affecting searchable sources."""
        self._content_sources[source.source_kind] = source

    def ingest_source(
        self,
        source_kind: str,
        since: Cursor | None = None,
    ) -> int:
        """Ingest records from a registered source through the injected port."""
        source = self._content_sources.get(source_kind)
        if source is None:
            raise KeyError(f"No content source registered for {source_kind!r}")
        if self._ingestor is None:
            raise RuntimeError(
                "Cannot ingest content: no record ingestor was wired into SearchKernel"
            )

        count = 0
        for record in source.iter_records(since=since):
            self._ingestor.index_record(record)
            count += 1
        return count

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
