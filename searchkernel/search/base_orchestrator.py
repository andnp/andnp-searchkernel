"""Canonical record-oriented orchestration boundary."""

from abc import ABC, abstractmethod
from typing import Any

from searchkernel.ports.search_results import RecordSearchOutcome
from searchkernel.search.record_pipeline import (
    RecordSearchPipeline,
)


class BaseSearchOrchestrator(ABC):
    """Small facade shared by local and durable record search sources."""

    def __init__(self, pipeline: RecordSearchPipeline) -> None:
        self._record_pipeline = pipeline

    @property
    def record_pipeline(self) -> RecordSearchPipeline:
        return self._record_pipeline

    @abstractmethod
    async def search(
        self,
        query: str,
        *,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> RecordSearchOutcome:
        """Search canonical records without mutating backend state."""
