"""Store ports: unified storage backends for vectors, keywords, graphs, and cache.

These ports unify the various storage needs of the kernel. Default implementation
uses Postgres + pgvector or the canonical local record stores.
"""

from collections.abc import Awaitable, Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from searchkernel.domain import (
    GraphEdge,
    GraphNeighbor,
    Record,
    RecordHit,
    RecordIdentity,
    SearchFilters,
    Vector,
)


@runtime_checkable
class VectorStore(Protocol):
    """Stores embeddings and supports ANN (approximate nearest neighbor) search."""

    def upsert(
        self, records: list[Record], model_name: str, dim: int
    ) -> None:
        """
        Upsert records with their embeddings into the store.

        Args:
            records: Records with embeddings populated (Record.embedding is set).
            model_name: Name of the embedding model used.
            dim: Dimensionality of the embeddings.

        A vector upsert is an all-vector operation: every record must have a
        non-null embedding. Use ``KeywordStore.index`` for keyword-only
        updates so an omitted embedding cannot leave an older vector
        searchable by accident.

        Per-model embedding isolation ensures safe migration between models.
        Implementations must reject a second dimension for an existing
        ``model_name`` rather than creating a mixed-dimension namespace.
        They should raise ``ModelDimensionMismatchError`` for that violation.
        """
        ...

    def search(
        self,
        query_vector: Vector,
        k: int,
        *,
        model_name: str,
        dim: int,
        filters: SearchFilters | None = None,
    ) -> Sequence[RecordHit]:
        """
        Search for the k nearest neighbors to a query vector.

        Args:
            query_vector: Embedding vector (must match stored dim).
            k: Number of results to return.
            model_name: Embedding model the query_vector was produced with.
                Selects which per-model table to search; not instance state,
                so concurrent callers can query different models safely.
            dim: Dimensionality of query_vector (must match the model's stored dim).
            filters: Optional filters (source-specific, opaque to core).

        Returns:
            Canonical record hits sorted by descending score.
        """
        ...

    def delete(self, record_ids: list[str]) -> None:
        """
        Delete records by ID.

        Args:
            record_ids: IDs to delete.
        """
        ...

    def epoch(self) -> int:
        """
        Return the current index epoch (version counter).

        Used to invalidate cache entries. Incremented on reindex/upsert.
        """
        ...


class BatchVectorStore(Protocol):
    """Optional boundary for exact vector retrieval over query batches."""

    def search_batch(
        self,
        query_vectors: Sequence[Vector],
        k: int,
        *,
        model_name: str,
        dim: int,
        filters: Sequence[SearchFilters | None] | None = None,
    ) -> Sequence[Sequence[RecordHit]]:
        ...


@runtime_checkable
class KeywordStore(Protocol):
    """Indexes and searches records by keyword/BM25."""

    def index(self, records: list[Record]) -> None:
        """
        Index records for full-text search.

        Args:
            records: Records to index (body and title are typically indexed).
        """
        ...

    def search(
        self, query: str, k: int, filters: SearchFilters | None = None
    ) -> Sequence[RecordHit]:
        """
        Search for top-k records matching the query.

        Args:
            query: Free-text search query.
            k: Number of results to return.
            filters: Optional filters (source-specific).

        Returns:
            Canonical record hits sorted by descending score.
        """
        ...


@runtime_checkable
class GraphStore(Protocol):
    """Stores and navigates graph relationships between records."""

    def upsert_edges(
        self,
        edges: Sequence[GraphEdge],
    ) -> None:
        """
        Upsert edges in the graph.

        Args:
            edges: List of (source_id, target_id, edge_type, weight) tuples.
                   Updates if source+target+edge_type exists; inserts otherwise.
        """
        ...

    def delete_edges(
        self,
        edges: Sequence[GraphEdge],
    ) -> None:
        """Delete graph edges and advance the graph mutation epoch."""
        ...

    def neighbors(
        self,
        record_id: str | RecordIdentity,
        edge_types: list[str] | None = None,
        depth: int = 1,
        max_neighbors: int | None = None,
    ) -> Sequence[GraphNeighbor]:
        """
        Retrieve neighbors of a record in the graph.

        Args:
            record_id: Starting record ID.
            edge_types: Optional filter by edge type names.
            depth: Number of hops to traverse (default 1 for one-hop).
            max_neighbors: Optional top-N bound on returned neighbors.

        Returns:
            Canonical graph neighbors sorted by the store's stable ordering.
        """
        ...


@runtime_checkable
class CacheStore(Protocol):
    """Caches stage outputs with epoch-based invalidation."""

    def get(self, key: str) -> Any | None:
        """
        Retrieve a cached value.

        Args:
            key: Cache key.

        Returns:
            Cached value, or None if not found or stale.
        """
        ...

    def set(self, key: str, value: Any, epoch: int) -> None:
        """Store a value with its index epoch."""
        ...

    def invalidate_epoch(self, epoch: int) -> None:
        """Discard entries from the supplied epoch or earlier."""
        ...


class AsyncVectorStore(Protocol):
    """Async boundary for record-oriented vector retrieval."""

    async def search(
        self,
        query_vector: Vector,
        k: int,
        *,
        model_name: str,
        dim: int,
        filters: SearchFilters | None = None,
    ) -> Sequence[RecordHit]:
        ...


class AsyncBatchVectorStore(Protocol):
    """Async optional boundary for exact vector retrieval over query batches."""

    async def search_batch(
        self,
        query_vectors: Sequence[Vector],
        k: int,
        *,
        model_name: str,
        dim: int,
        filters: Sequence[SearchFilters | None] | None = None,
    ) -> Sequence[Sequence[RecordHit]]:
        ...


class AsyncKeywordStore(Protocol):
    """Async boundary for record-oriented keyword retrieval."""

    async def search(
        self, query: str, k: int, filters: SearchFilters | None = None
    ) -> Sequence[RecordHit]:
        ...


class AsyncGraphStore(Protocol):
    """Async boundary for record-oriented graph expansion."""

    async def neighbors(
        self,
        record_id: str | RecordIdentity,
        edge_types: list[str] | None = None,
        depth: int = 1,
        max_neighbors: int | None = None,
    ) -> Sequence[GraphNeighbor]:
        ...


class BatchGraphStore(Protocol):
    """Retrieve graph neighbors for multiple canonical seeds."""

    def neighbors_many(
        self,
        identities: Sequence[RecordIdentity],
        *,
        depth: int,
        max_neighbors: int | None = None,
    ) -> Mapping[
        str,
        Sequence[GraphNeighbor],
    ] | Awaitable[
        Mapping[
            str,
            Sequence[GraphNeighbor],
        ]
    ]:
        ...
