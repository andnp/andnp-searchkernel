"""Adapt the kernel's existing orchestrator to the federation source port."""

from collections.abc import Iterable
from typing import Any

from searchkernel.domain import ChunkResult, ScoredRef
from searchkernel.search.orchestrator import SearchOrchestrator


class LocalSearchSource:
    """Expose ``SearchOrchestrator`` results as federated ``ScoredRef`` values."""

    source_kind = "local"

    def __init__(self, orchestrator: SearchOrchestrator):
        self._orchestrator = orchestrator

    async def search(
        self, query: str, k: int, filters: dict[str, Any] | None = None
    ) -> Iterable[ScoredRef]:
        source_filter = (filters or {}).get("source_filter")
        chunk_results, _compression_stats, _strategy_stats = (
            await self._orchestrator.query(
                query,
                top_k=k,
                top_n=k,
                source_filter=source_filter,
            )
        )
        return [self._to_scored_ref(result) for result in chunk_results]

    @staticmethod
    def _to_scored_ref(result: ChunkResult) -> ScoredRef:
        metadata = dict(result.metadata)
        metadata.update(
            {
                "text": result.content,
                "doc_id": result.record_id,
            }
        )
        return ScoredRef(
            source_id=result.chunk_id,
            score=result.score,
            source_kind="local",
            metadata=metadata,
        )
