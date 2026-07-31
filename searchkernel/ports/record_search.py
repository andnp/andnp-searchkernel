"""Async boundaries for canonical record search."""

from typing import Protocol

from searchkernel.domain import Record, RecordIdentity


class AsyncRecordHydrator(Protocol):
    """Hydrate by canonical identity without mutating source state."""

    async def hydrate_record(
        self,
        identity: RecordIdentity,
    ) -> Record | None:
        ...
