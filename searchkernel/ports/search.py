"""SearchAPI port for the canonical record-oriented search path."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from searchkernel.domain import SearchFilters, Vector
from searchkernel.ports.search_results import RecordSearchOutcome


@runtime_checkable
class SearchAPI(Protocol):
    """Primary driving interface for record-oriented retrieval."""

    async def search(
        self,
        query: str,
        *,
        filters: SearchFilters | None = None,
        limit: int = 10,
        query_vector: Vector | None = None,
    ) -> RecordSearchOutcome:
        """
        Execute the canonical record search pipeline.

        Args:
            query: The search query string. Still required, and still
                drives the keyword/graph lanes and query routing, even when
                ``query_vector`` is supplied.
            filters: Optional source-specific filters (opaque to core).
            limit: Maximum number of hydrated record results to return.
            query_vector: A precomputed embedding for ``query``, used
                directly for the vector lane instead of embedding ``query``.
                For a caller that already holds a valid vector for this
                query -- e.g. searching for records similar to one whose
                own embedding is already on hand -- this avoids a redundant
                (and, for long text, potentially failing) re-embed.

        Returns:
            Record results, provenance, and explicit degradation diagnostics.
        """
        ...
