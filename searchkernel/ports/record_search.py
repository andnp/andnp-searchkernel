"""Optional batch boundaries for canonical record search."""

from collections.abc import Awaitable, Mapping, Sequence
from typing import Protocol

from searchkernel.domain import Record, RecordIdentity


class AsyncRecordHydrator(Protocol):
    """Hydrate by canonical identity without mutating source state."""

    async def hydrate_record(
        self,
        identity: RecordIdentity,
    ) -> Record | None:
        ...


class BatchRecordHydrator(Protocol):
    """Hydrate multiple records while retaining canonical storage keys."""

    def hydrate_records(
        self,
        identities: Sequence[RecordIdentity],
    ) -> Mapping[str, Record | None] | Awaitable[Mapping[str, Record | None]]:
        ...
