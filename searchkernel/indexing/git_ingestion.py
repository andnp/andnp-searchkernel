"""Wires any ContentSource into the live IndexManager.

Ingests source records (git commits, notes, or any future ContentSource)
through the same chunking/indexing path as documents, so they land in the
shared vector/keyword/graph store and become discoverable via
SearchOrchestrator.query(source_filter=[...]).
"""

import logging
from collections.abc import Callable, Iterable
from typing import Protocol

from searchkernel.domain import Cursor, Record

logger = logging.getLogger(__name__)


class GitIndexManager(Protocol):
    """Minimum index-manager surface required for source ingestion."""

    def index_record(self, record: Record) -> None: ...


class IngestibleSource(Protocol):
    """Minimum ContentSource surface required for ingestion."""

    repo_path: str

    def iter_records(self, since: Cursor | None) -> Iterable[Record]: ...


def ingest_git_source(
    index_manager: GitIndexManager,
    source: IngestibleSource,
    since: Cursor | None = None,
    on_record: Callable[[Record], None] | None = None,
) -> int:
    """Ingest every record yielded by a ContentSource into index_manager.

    Returns the number of records ingested.
    """
    count = 0
    for record in source.iter_records(since):
        index_manager.index_record(record)
        if on_record is not None:
            on_record(record)
        count += 1

    if count:
        logger.info(
            "Ingested %d record(s) from %s into the live index",
            count,
            source.repo_path,
        )

    return count
