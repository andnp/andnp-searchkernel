"""ContentSource port: adapters for ingesting content from external sources.

This is the primary outbound port for the kernel. Content sources implement one
of two flavors:
  - Ingestible: the kernel stores and indexes the content
  - Searchable: the source runs its own search; kernel merges results
"""

from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable

from searchkernel.domain import ChangeSignal, Cursor, Record, ScoredRef


@runtime_checkable
class ContentSource(Protocol):
    """Ingestible source: kernel owns indexing and storage.

    The source yields Records; the kernel chunks, embeds (unless the record
    carries pre-computed embeddings), and indexes them.

    Attributes:
        source_kind: Stable identifier for this source type
                     (e.g., "note", "git_commit", "gmail").
    """

    source_kind: str

    def iter_records(self, since: Cursor | None = None) -> Iterable[Record]:
        """
        Iterate over records to ingest, optionally since a cursor.

        Args:
            since: Optional watermark (e.g., last processed commit SHA, timestamp).
                   If provided, only records modified after this point are returned.

        Yields:
            Records ready for chunking and indexing.
        """
        ...

    def change_signal(self) -> ChangeSignal:
        """
        Return change-detection signal for this source.

        Returns:
            A dict with one of:
              - {"watch": True}: use a file-watcher to detect changes
              - {"poll_interval": 3600}: poll for changes every N seconds
              Any other source-specific config can be included.
        """
        ...


@runtime_checkable
class RecordIngestor(Protocol):
    """Minimal indexing surface required to ingest source records."""

    def index_record(self, record: Record) -> bool:
        """Index one source-agnostic record."""
        ...


@runtime_checkable
class AsyncRecordIngestor(Protocol):
    """Checkpointed batch indexing boundary for record ingestion."""

    async def index_records(
        self,
        records: list[Record],
        *,
        checkpoint: Cursor | None = None,
    ) -> Cursor:
        """Index a batch and return the checkpoint safe to persist."""
        ...


@runtime_checkable
class SearchableSource(Protocol):
    """Federated source: source runs its own retrieval; kernel merges results.

    The source already owns embeddings and ranking. The kernel never stores
    the source's content; it merges ranked results from multiple sources and
    may perform one late rerank according to the federation entrypoint.

    Source scores are source-local, are not assumed comparable across sources,
    and are retained as provenance metadata.

    Attributes:
        source_kind: Stable identifier for this source type
                     (e.g., "memory", "jira").
    """

    source_kind: str

    async def search(
        self, query: str, k: int, filters: dict[str, Any] | None = None
    ) -> Iterable[ScoredRef]:
        """
        Run the source's native search and return ranked references.

        Args:
            query: The search query string.
            k: Maximum number of results to return.
            filters: Optional source-specific filters (opaque to the kernel).

        Returns:
            An iterable of ScoredRefs in descending source-local score order. The kernel
            merges candidates and applies the optional or required late
            rerank defined by ``runtime.federation.search_anything``.
        """
        ...
