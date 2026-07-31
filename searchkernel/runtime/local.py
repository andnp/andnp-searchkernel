"""Federation source for the canonical local record backend."""

from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any, Protocol

from searchkernel.domain import (
    ChunkResult,
    Record,
    RecordStatus,
    ScoredRef,
    SearchResultProvenance,
)
from searchkernel.search.record_pipeline import RecordSearchOutcome, RecordSearchResult


class LegacyQueryOrchestrator(Protocol):
    """Protocol-shaped legacy query surface for explicit adaptation."""

    async def query(
        self,
        query_text: str,
        *,
        top_k: int,
        top_n: int,
        source_filter: list[str] | None,
    ) -> tuple[list[ChunkResult], object, object]: ...


class LegacyLocalOrchestratorAdapter:
    """Adapt a legacy chunk query object without restoring it as a search API."""

    def __init__(self, orchestrator: LegacyQueryOrchestrator):
        self._orchestrator = orchestrator

    async def search(
        self,
        query: str,
        *,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> RecordSearchOutcome:
        source_kinds = (filters or {}).get("source_kinds")
        source_filter = (
            list(source_kinds)
            if isinstance(source_kinds, list)
            and all(isinstance(item, str) for item in source_kinds)
            else None
        )
        chunk_results, _compression, _strategy = await self._orchestrator.query(
            query,
            top_k=limit,
            top_n=limit,
            source_filter=source_filter,
        )
        return RecordSearchOutcome(
            results=tuple(self._to_record_result(result) for result in chunk_results)
        )

    @staticmethod
    def _to_record_result(result: ChunkResult) -> RecordSearchResult:
        metadata = dict(result.metadata)
        metadata.setdefault("chunk_id", result.chunk_id)
        metadata.setdefault("doc_id", result.record_id)
        modified_time = metadata.get("modified_time")
        timestamp = (
            modified_time if isinstance(modified_time, datetime) else datetime.now(UTC)
        )
        record = Record(
            source_kind="local",
            source_id=result.chunk_id,
            title=str(
                metadata.get("header_path")
                or metadata.get("file_path")
                or result.chunk_id
            ),
            body=result.content,
            created_at=timestamp,
            updated_at=timestamp,
            metadata=metadata,
            status=RecordStatus.ACTIVE,
        )
        return RecordSearchResult(
            record=record,
            score=result.score,
            provenance=result.provenance or SearchResultProvenance(),
        )


class RecordSearchSource(Protocol):
    """Canonical record search boundary consumed by the local source."""

    async def search(
        self,
        query: str,
        *,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> RecordSearchOutcome: ...


class LocalSearchSource:
    """Expose canonical local record results to federation."""

    source_kind = "local"

    def __init__(self, orchestrator: RecordSearchSource):
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


__all__ = [
    "LegacyLocalOrchestratorAdapter",
    "LegacyQueryOrchestrator",
    "LocalSearchSource",
    "RecordSearchSource",
]
