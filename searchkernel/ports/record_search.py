"""Optional batch boundaries for canonical record search."""

from collections.abc import Awaitable, Mapping, Sequence
from typing import Protocol, runtime_checkable

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


@runtime_checkable
class BatchParentRecordExpander(Protocol):
    """Resolve optional parents for multiple canonical identities."""

    def parent_identities(
        self,
        identities: Sequence[RecordIdentity],
    ) -> Mapping[str, RecordIdentity | None] | Awaitable[
        Mapping[str, RecordIdentity | None]
    ]:
        ...


@runtime_checkable
class ParentRecordExpander(Protocol):
    """Resolve an optional parent while retaining canonical identity."""

    def parent_identity(
        self,
        identity: RecordIdentity,
    ) -> RecordIdentity | None | Awaitable[RecordIdentity | None]:
        ...
