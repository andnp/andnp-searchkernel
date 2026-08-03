"""Batch graph-neighbor port contracts."""

from collections.abc import Awaitable, Mapping, Sequence
from typing import Protocol

from searchkernel.domain import GraphNeighbor, RecordIdentity


class BatchGraphStore(Protocol):
    """Retrieve graph neighbors for multiple canonical seeds."""

    def neighbors_many(
        self,
        identities: Sequence[RecordIdentity],
        *,
        depth: int,
        max_neighbors: int | None = None,
    ) -> Mapping[str, Sequence[GraphNeighbor]] | Awaitable[
        Mapping[str, Sequence[GraphNeighbor]]
    ]:
        ...
