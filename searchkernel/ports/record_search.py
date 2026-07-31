"""Async boundaries for canonical record search."""

from typing import Protocol

from searchkernel.domain import Record


class AsyncRecordHydrator(Protocol):
    """Hydrate by canonical identity without mutating source state."""

    async def hydrate_record(
        self,
        record_id: str,
        *,
        source_kind: str | None = None,
        workspace_id: str | None = None,
    ) -> Record | None:
        ...
