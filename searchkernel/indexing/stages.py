"""Small, behavior-preserving stages for bulk index construction.

The stage objects intentionally only wrap the existing index APIs.  They are
an architectural seam for later progressive indexing work, not a new
readiness or scheduling system.
"""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from searchkernel.domain import Chunk, RecordIdentity, canonical_storage_key
from searchkernel.indexing.batches import PreparedIndexRecord
from searchkernel.indexing.batches import (
    iter_prepared_index_batches as iter_record_batches,
)
from searchkernel.indexing.semantic import SemanticInput, semantic_input_for_chunk
from searchkernel.search.edge_types import infer_edge_type


@runtime_checkable
class LinkExtractingParser(Protocol):
    """Parser surface for link-with-context extraction (e.g. markdown headers).

    Kept as a Protocol so the library never imports a concrete, source-specific
    parser class (parsers are app-owned adapters).
    """

    def extract_links_with_context(self, file_path: str) -> list: ...


@dataclass
class PreparedIndexBatch:
    """Prepared records and the bounded payloads consumed by each stage."""

    records: list[PreparedIndexRecord]
    chunks: list[Chunk] = field(default_factory=list)
    graph_nodes: list[tuple[str, dict]] = field(default_factory=list)
    graph_edges: list[tuple[str, str, str, str]] = field(default_factory=list)
    semantic_inputs: list[SemanticInput] = field(default_factory=list)

    @classmethod
    def from_records(
        cls,
        records: list[PreparedIndexRecord],
        *,
        encoder_namespace: str = "",
    ) -> "PreparedIndexBatch":
        chunks = [chunk for record in records for chunk in record.chunks]
        graph_nodes, graph_edges = build_graph_payload(records)
        return cls(
            records=records,
            chunks=chunks,
            graph_nodes=graph_nodes,
            graph_edges=graph_edges,
            semantic_inputs=[
                semantic_input_for_chunk(
                    chunk,
                    encoder_namespace=encoder_namespace,
                )
                for chunk in chunks
            ],
        )


def build_graph_payload(
    records: list[PreparedIndexRecord],
) -> tuple[list[tuple[str, dict]], list[tuple[str, str, str, str]]]:
    """Shape record, chunk, and link data for the bulk graph APIs."""
    nodes: list[tuple[str, dict]] = []
    edges: list[tuple[str, str, str, str]] = []
    for prepared in records:
        identity = prepared.record.identity
        nodes.append((identity.storage_key, prepared.graph_metadata))
        nodes.extend(
            (
                _graph_storage_key(identity, chunk.chunk_id),
                chunk.metadata,
            )
            for chunk in prepared.chunks
        )
        if isinstance(prepared.parser, LinkExtractingParser):
            links = prepared.parser.extract_links_with_context(prepared.file_path)
            edges.extend(
                (
                    identity.storage_key,
                    _graph_storage_key(identity, link.target),
                    infer_edge_type(link.header_context, link.target).value,
                    link.header_context,
                )
                for link in links
            )
        else:
            edges.extend(
                (
                    identity.storage_key,
                    _graph_storage_key(identity, link),
                    "links_to",
                    "",
                )
                for link in prepared.record.metadata.get("links", [])
            )
    return nodes, edges


def _graph_storage_key(identity: RecordIdentity, value: str) -> str:
    """Normalize source-local graph references to canonical record identities."""
    try:
        return RecordIdentity.from_storage_key(value).storage_key
    except (TypeError, ValueError):
        return canonical_storage_key(
            identity.workspace_id,
            identity.source_kind,
            value,
        )


def iter_prepared_index_batches(
    records: Iterable[PreparedIndexRecord],
    *,
    max_records: int,
    max_chunks: int,
) -> Iterator[PreparedIndexBatch]:
    """Yield prepared batches from bounded single-pass record batches."""
    for current in iter_record_batches(
        records,
        max_records=max_records,
        max_chunks=max_chunks,
    ):
        yield PreparedIndexBatch.from_records(current)


@dataclass(frozen=True)
class StageCounters:
    records: int = 0
    chunks: int = 0
    nodes: int = 0
    edges: int = 0


@dataclass(frozen=True)
class StageResult:
    stage: str
    counters: StageCounters


class IndexStage(Protocol):
    name: str

    def apply(self, batch: PreparedIndexBatch) -> StageResult: ...


class ChunkIndexWriter(Protocol):
    def add_chunks(self, chunks: list[Chunk]) -> None: ...


class GraphIndexWriter(Protocol):
    def add_nodes(self, nodes: list[tuple[str, dict]]) -> None: ...

    def add_edges(self, edges: list[tuple[str, str, str, str]]) -> None: ...


class KeywordStage:
    name = "keyword"

    def __init__(self, keyword: ChunkIndexWriter) -> None:
        self._keyword = keyword

    def apply(self, batch: PreparedIndexBatch) -> StageResult:
        if batch.chunks:
            self._keyword.add_chunks(batch.chunks)
        return StageResult(
            self.name,
            StageCounters(records=len(batch.records), chunks=len(batch.chunks)),
        )


class GraphStage:
    name = "graph"

    def __init__(self, graph: GraphIndexWriter) -> None:
        self._graph = graph

    def apply(self, batch: PreparedIndexBatch) -> StageResult:
        if batch.graph_nodes:
            self._graph.add_nodes(batch.graph_nodes)
        if batch.graph_edges:
            self._graph.add_edges(batch.graph_edges)
        return StageResult(
            self.name,
            StageCounters(
                records=len(batch.records),
                chunks=len(batch.chunks),
                nodes=len(batch.graph_nodes),
                edges=len(batch.graph_edges),
            ),
        )


class SemanticStage:
    name = "semantic"

    def __init__(self, vector: ChunkIndexWriter) -> None:
        self._vector = vector

    def apply(self, batch: PreparedIndexBatch) -> StageResult:
        if batch.chunks:
            self._vector.add_chunks(batch.chunks)
        return StageResult(
            self.name,
            StageCounters(records=len(batch.records), chunks=len(batch.chunks)),
        )
