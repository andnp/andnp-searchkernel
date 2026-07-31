"""Live search and indexing ports.

These protocols describe the richer in-process index surface used by the
search and indexing paths. Storage ports remain intentionally narrower and
backend-oriented; concrete indices satisfy these protocols structurally.
"""

from pathlib import Path
from typing import Any, NotRequired, Protocol, TypedDict, runtime_checkable

from searchkernel.domain import Chunk


class LiveSearchResult(TypedDict):
    chunk_id: str
    doc_id: str
    content: NotRequired[str]
    score: NotRequired[float]
    file_path: NotRequired[str]
    header_path: NotRequired[str]
    project_id: NotRequired[str | None]
    metadata: NotRequired[dict[str, Any]]


@runtime_checkable
class VectorIndexPort(Protocol):
    """Vector index surface required by live search and indexing."""

    def add_chunks(self, chunks: list[Chunk]) -> None: ...

    def remove(self, document_id: str) -> None: ...

    def remove_chunk(self, chunk_id: str) -> None: ...

    def search(
        self,
        query: str,
        top_k: int = 10,
        excluded_files: set[str] | None = None,
        docs_root: Path | None = None,
    ) -> list[LiveSearchResult]: ...

    def get_chunk_by_id(self, chunk_id: str) -> dict[str, Any] | None: ...

    def get_chunk_ids_for_document(self, doc_id: str) -> list[str]: ...

    def get_document_ids(self) -> list[str]: ...

    def get_parent_content(self, parent_chunk_id: str) -> str | None: ...

    def get_embedding_for_chunk(self, chunk_id: str) -> list[float] | None: ...

    def expand_query(
        self,
        query: str,
        top_k: int = 3,
        similarity_threshold: float = 0.5,
    ) -> str: ...


@runtime_checkable
class KeywordIndexPort(Protocol):
    """Keyword index surface required by live search and indexing."""

    def add_chunks(self, chunks: list[Chunk]) -> None: ...

    def remove(self, document_id: str) -> None: ...

    def remove_chunks(self, chunk_ids: list[str]) -> None: ...

    def search(
        self,
        query: str,
        top_k: int = 10,
        excluded_files: set[str] | None = None,
        docs_root: Path | None = None,
    ) -> list[LiveSearchResult]: ...

    def get_chunk_by_id(self, chunk_id: str) -> dict[str, Any] | None: ...


@runtime_checkable
class GraphIndexPort(Protocol):
    """Graph index surface required by live search and indexing."""

    def add_node(self, doc_id: str, metadata: dict[str, Any]) -> None: ...

    def remove_node(self, doc_id: str) -> None: ...

    def remove_chunk(self, chunk_id: str) -> None: ...

    def has_node(self, doc_id: str) -> bool: ...

    def add_edge(
        self,
        source: str,
        target: str,
        edge_type: str,
        edge_context: str = "",
    ) -> None: ...

    def get_edges_from(self, source: str) -> list[dict[str, str]]: ...

    def get_all_nodes_with_metadata(
        self,
    ) -> list[tuple[str, dict[str, Any]]]: ...

    def rank_neighbors(self, seed_scores: dict[str, float]) -> list[tuple[str, float]]: ...

    def boost_by_community(
        self,
        doc_ids: list[str],
        seed_doc_ids: set[str],
        boost_factor: float = 1.1,
    ) -> dict[str, float]: ...
