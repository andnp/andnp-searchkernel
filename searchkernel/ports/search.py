"""SearchAPI port for the canonical record-oriented search path."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from searchkernel.domain import SearchFilters

if TYPE_CHECKING:
    from searchkernel.search.record_pipeline import RecordSearchOutcome


@runtime_checkable
class SearchAPI(Protocol):
    """Primary driving interface for record-oriented retrieval."""

    async def search(
        self,
        query: str,
        *,
        filters: SearchFilters | None = None,
        limit: int = 10,
    ) -> RecordSearchOutcome:
        """
        Execute the canonical record search pipeline.

        Args:
            query: The search query string.
            filters: Optional source-specific filters (opaque to core).
            limit: Maximum number of hydrated record results to return.

        Returns:
            Record results, provenance, and explicit degradation diagnostics.
        """
        ...
