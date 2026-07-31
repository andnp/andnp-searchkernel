"""Federation source for the canonical local record backend."""

from collections.abc import Iterable
from typing import Any

from searchkernel.domain import ScoredRef
from searchkernel.search.orchestrator import SearchOrchestrator


class LocalSearchSource:
    """Expose canonical local record results to federation."""

    source_kind = "local"

    def __init__(self, orchestrator: SearchOrchestrator):
        self._orchestrator = orchestrator

    async def search(
        self, query: str, k: int, filters: dict[str, Any] | None = None
    ) -> Iterable[ScoredRef]:
        filters = dict(filters or {})
        source_filter = filters.pop("source_filter", None)
        if source_filter is not None:
            filters["source_kinds"] = list(source_filter)
        if "workspace" in filters and "workspace_id" not in filters:
            filters["workspace_id"] = filters.pop("workspace")

        outcome = await self._orchestrator.search(query, limit=k, filters=filters)
        return [
            self._to_scored_ref(result.record, result.score, result.provenance)
            for result in outcome.results
        ]

    @staticmethod
    def _to_scored_ref(record, score: float, provenance) -> ScoredRef:
        metadata = dict(record.metadata)
        metadata.update(
            {
                "text": record.body,
                "title": record.title,
                "uri": record.uri,
                "storage_key": record.storage_key,
                "provenance": provenance.to_dict(),
            }
        )
        return ScoredRef(
            source_id=record.source_id,
            score=score,
            source_kind=record.source_kind,
            workspace_id=record.workspace_id,
            metadata=metadata,
        )


__all__ = ["LocalSearchSource"]
