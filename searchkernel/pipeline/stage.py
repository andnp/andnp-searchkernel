"""SearchStage contract: composable pure transforms over a SearchContext.

A query pipeline is a sequence of stages threading a `SearchContext`
through retrieve -> graph-expand -> fuse -> dedup/rerank -> hydrate, etc.
Each stage is a narrow, independently-testable unit; stages compose by
returning a new context rather than mutating the one they receive.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Protocol, runtime_checkable

from searchkernel.domain import (
    Chunk,
    ChunkResult,
    CompressionStats,
    Record,
    SearchResultProvenance,
)
from searchkernel.search.classifier import QueryType
from searchkernel.search.types import SearchResultDict


def require_state[StateT](value: StateT | None, name: str) -> StateT:
    """Return a required state value or fail with its missing field name."""

    if value is None:
        raise ValueError(f"missing required search state: {name}")
    return value


@dataclass
class SearchState(Mapping[str, object]):
    """Typed, source-agnostic values shared by pipeline stages.

    The mapping interface is a temporary boundary for callers that still
    construct contexts with ``metadata=``. New stages should use attributes
    directly; the state itself remains the single stored representation.
    """

    # Query state
    base_semantic_weight: float | None = None
    base_keyword_weight: float | None = None
    base_graph_weight: float | None = None
    requested_top_k: int | None = None
    top_n: int | None = None
    top_k: int | None = None
    query_type: QueryType | None = None
    strategy_weights: dict[str, float] = field(default_factory=dict)
    project_filter: list[str] | None = None
    source_filter: list[str] | None = None
    active_project: str | None = None
    excluded_files: set[str] | None = None
    docs_root: Path | None = None
    vector_results: list[SearchResultDict] = field(default_factory=list)
    keyword_results: list[SearchResultDict] = field(default_factory=list)
    graph_chunk_ids: list[str] = field(default_factory=list)
    graph_doc_scores: dict[str, float] = field(default_factory=dict)
    chunk_id_to_doc_id: dict[str, str] = field(default_factory=dict)
    all_doc_ids: set[str] = field(default_factory=set)
    seed_scores: dict[str, float] = field(default_factory=dict)
    excluded_chunk_ids: set[str] | None = None
    seed_doc_ids: set[str] = field(default_factory=set)
    skip_tag_expansion: bool = False
    applied_tag_expansion_results: list[SearchResultDict] = field(default_factory=list)
    tag_expansion_count: int = 0
    provenance_strategy_results: dict[str, list[tuple[str, float]]] = field(
        default_factory=dict
    )
    result_provenance: dict[str, SearchResultProvenance] = field(default_factory=dict)
    chunk_results: list[ChunkResult] = field(default_factory=list)
    missing_chunk_ids: list[str] = field(default_factory=list)
    missing_parent_chunk_ids: list[str] = field(default_factory=list)
    compression_stats: CompressionStats | None = None
    get_embedding: Callable[[str], list[float] | None] | None = None
    get_content: Callable[[str], str | None] | None = None

    # Ingestion state
    record: Record | None = None
    chunks: list[Chunk] = field(default_factory=list)
    indexed_chunk_ids: list[str] = field(default_factory=list)
    documents_path: str | Path | None = None
    documents_roots: list[str | Path] = field(default_factory=list)
    include_patterns: list[str] | None = None
    exclude_patterns: list[str] | None = None
    exclude_hidden_dirs: bool | None = None
    discovered_files: list[str] = field(default_factory=list)
    doc_id: str | None = None
    docs_path: Path | None = None
    suffixes: list[str] = field(default_factory=list)
    resolved_path: Path | None = None
    old_doc_id: str | None = None
    new_doc_id: str | None = None
    new_chunks: list[Chunk] = field(default_factory=list)
    move_applied: bool | None = None
    moved_chunk_count: int = 0
    hash_store_updated: bool | None = None
    removed_doc_ids: set[str] = field(default_factory=set)
    added_docs: dict[str, list[Chunk]] = field(default_factory=dict)
    move_detection_threshold: float | None = None
    moved_files: dict[str, str] = field(default_factory=dict)
    _present: frozenset[str] = field(default_factory=frozenset, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self._present:
            self._present = frozenset(
                field.name
                for field in fields(self)
                if not field.name.startswith("_") and self._is_set(field.name)
            )

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> SearchState:
        state = cls()
        valid_fields = {
            field.name for field in fields(cls) if not field.name.startswith("_")
        }
        unknown_fields = set(values) - valid_fields
        if unknown_fields:
            raise TypeError(f"unknown search state fields: {sorted(unknown_fields)}")
        for key, value in values.items():
            setattr(state, key, value)
        state._present = frozenset(values)
        return state

    def __getitem__(self, key: str) -> object:
        if key not in self._present:
            raise KeyError(key)
        try:
            return getattr(self, key)
        except AttributeError as exc:
            raise KeyError(key) from exc

    def __iter__(self) -> Iterator[str]:
        return iter(self._present)

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def get(self, key: str, default: object = None) -> object:
        try:
            return self[key]
        except KeyError:
            return default

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Mapping):
            return dict(self) == dict(other)
        return NotImplemented

    def _is_set(self, key: str) -> bool:
        value = getattr(self, key)
        if value is None or value is False:
            return False
        if isinstance(value, (str, bytes, int, float)):
            return value != 0 and value != ""
        if isinstance(value, (list, dict, set, tuple)):
            return bool(value)
        return True


@dataclass(init=False)
class SearchContext:
    """State threaded through a query pipeline.

    Stages must not mutate a context in place; `run()` returns a new
    `SearchContext` (e.g. via `dataclasses.replace`) reflecting the
    stage's output, so a pipeline run is a strict left-to-right fold.
    """

    query: str
    candidates: list[tuple[str, float]] = field(default_factory=list)
    strategy_results: dict[str, list[tuple[str, float]]] = field(default_factory=dict)
    state: SearchState

    def __init__(
        self,
        query: str,
        candidates: list[tuple[str, float]] | None = None,
        strategy_results: dict[str, list[tuple[str, float]]] | None = None,
        *,
        state: SearchState | None = None,
        metadata: Mapping[str, object] | None = None,
    ):
        if state is not None and metadata is not None:
            raise ValueError("provide state or metadata, not both")
        self.query = query
        self.candidates = [] if candidates is None else candidates
        self.strategy_results = (
            {} if strategy_results is None else strategy_results
        )
        self.state = (
            state
            if state is not None
            else SearchState.from_mapping({} if metadata is None else metadata)
        )

    @property
    def metadata(self) -> SearchState:
        """Compatibility view for callers migrating from string-key metadata."""

        return self.state


def replace_state(context: SearchContext, updates: Mapping[str, object]) -> SearchContext:
    """Return a context with typed state updates applied."""

    return replace(
        context,
        state=SearchState.from_mapping({**dict(context.state), **updates}),
    )


@runtime_checkable
class SearchStage(Protocol):
    """A single composable step of a query (or ingestion) pipeline.

    Implementations must be pure with respect to `context`: given the same
    input they produce the same output, with no hidden mutation of shared
    state between calls (aside from stage-local caches/instrumentation).
    """

    name: str

    def run(self, context: SearchContext) -> SearchContext: ...


@runtime_checkable
class AsyncSearchStage(Protocol):
    """An I/O-bound composable pipeline step (e.g. retrieval).

    Same contract as `SearchStage` -- pure with respect to `context`,
    returns a new context rather than mutating -- but `run` is a
    coroutine. Stages that must await (index lookups, network calls)
    implement this instead of `SearchStage`; a pipeline executor awaits
    `AsyncSearchStage`s and calls `SearchStage`s directly.
    """

    name: str

    async def run(self, context: SearchContext) -> SearchContext: ...
