"""Core domain types for the search kernel.

Pure data types representing the source-agnostic contracts between the kernel
and the outside world. No I/O, no imports from adapters/runtime/stores.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from functools import lru_cache
from typing import Any

# ===== Supporting types =====

class RecordStatus(str, Enum):
    """Lifecycle status of a record in the kernel."""

    ACTIVE = "active"
    STALE = "stale"
    ARCHIVED = "archived"


class Tier(str, Enum):
    """Tier for LLM provider selection (performance vs. quality)."""

    FAST = "fast"      # High-volume, low-latency (SLM, local)
    SMART = "smart"    # Higher quality, higher latency (Claude, etc.)


# Type aliases for clarity in port signatures
Vector = list[float]  # Embedding vector
Cursor = str | None  # Watermark for incremental sync (e.g., commit SHA, timestamp)
SearchFilters = Mapping[str, Any]
"""Read-only search filters; source-specific keys remain opaque to the kernel."""

ChangeSignal = dict[str, Any]  # Source change info: {"watch": bool, "poll_interval": int}


def _validate_identity_part(
    name: str,
    value: str | None,
    *,
    allow_none: bool = False,
) -> None:
    if allow_none and value is None:
        return
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string or None")
    if not value:
        raise ValueError(f"{name} must not be empty")


def canonical_storage_key(
    workspace_id: str | None,
    source_kind: str,
    source_id: str,
) -> str:
    """Return a collision-free key for a record's composite identity.

    JSON array encoding is deliberate: source identifiers may contain any
    delimiter, and ``source_kind`` is part of identity rather than metadata.
    """
    _validate_identity_part("workspace_id", workspace_id, allow_none=True)
    _validate_identity_part("source_kind", source_kind)
    _validate_identity_part("source_id", source_id)
    return "record:" + json.dumps(
        [workspace_id, source_kind, source_id],
        ensure_ascii=False,
        separators=(",", ":"),
    )


@dataclass(frozen=True, slots=True)
class RecordIdentity:
    """Composite identity carried across retrieval, graph, and hydration."""

    workspace_id: str | None
    source_kind: str
    source_id: str
    _storage_key: str | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        _validate_identity_part("workspace_id", self.workspace_id, allow_none=True)
        _validate_identity_part("source_kind", self.source_kind)
        _validate_identity_part("source_id", self.source_id)

    @property
    def storage_key(self) -> str:
        cached = self._storage_key
        if cached is None:
            cached = canonical_storage_key(
                self.workspace_id,
                self.source_kind,
                self.source_id,
            )
            object.__setattr__(self, "_storage_key", cached)
        return cached

    def to_dict(self) -> dict[str, str | None]:
        """Return the portable representation used by API boundaries."""
        return {
            "workspace_id": self.workspace_id,
            "source_kind": self.source_kind,
            "source_id": self.source_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RecordIdentity":
        """Build an identity from a mapping without accepting aliases."""
        try:
            return cls(
                workspace_id=value.get("workspace_id"),
                source_kind=value["source_kind"],
                source_id=value["source_id"],
            )
        except KeyError as exc:
            raise ValueError(f"missing identity field: {exc.args[0]}") from exc

    @classmethod
    def from_storage_key(cls, storage_key: str) -> "RecordIdentity":
        return _parse_storage_key(storage_key)


@lru_cache(maxsize=32_768)
def _parse_storage_key(storage_key: str) -> RecordIdentity:
    # Bounded because a long-lived daemon re-parses the same storage keys
    # across graph edges, hydration, and search hits every rebuild; a
    # whole-corpus rebuild touches on the order of 10^4 distinct keys, so
    # this comfortably covers the working set. lru_cache does not cache
    # exceptions, so malformed/non-canonical keys still raise every call.
    if not storage_key.startswith("record:"):
        raise ValueError("storage key must start with 'record:'")
    try:
        value = json.loads(storage_key.removeprefix("record:"))
    except json.JSONDecodeError as exc:
        raise ValueError("invalid record storage key") from exc
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError("invalid record storage key")
    workspace_id, source_kind, source_id = value
    identity = RecordIdentity(workspace_id, source_kind, source_id)
    if identity.storage_key != storage_key:
        raise ValueError("record storage key is not canonical")
    return identity


@dataclass(frozen=True, slots=True)
class RecordHit:
    """A scored store result with complete record identity."""

    identity: RecordIdentity
    score: float

    @property
    def workspace_id(self) -> str | None:
        return self.identity.workspace_id

    @property
    def source_kind(self) -> str:
        return self.identity.source_kind

    @property
    def source_id(self) -> str:
        return self.identity.source_id

    @property
    def storage_key(self) -> str:
        return self.identity.storage_key

    def __iter__(self):
        """Retain tuple unpacking for adapters during the contract migration."""
        yield self.source_id
        yield self.score

    def __getitem__(self, index: int):
        return (self.source_id, self.score)[index]

    def __len__(self) -> int:
        return 2


@dataclass(frozen=True, slots=True)
class GraphNeighbor:
    """A graph result retaining the neighbor's complete identity."""

    identity: RecordIdentity
    edge_type: str
    weight: float

    @property
    def source_id(self) -> str:
        return self.identity.source_id

    def __iter__(self):
        yield self.source_id
        yield self.edge_type
        yield self.weight

    def __getitem__(self, index: int):
        return (self.source_id, self.edge_type, self.weight)[index]

    def __len__(self) -> int:
        return 3


@dataclass(frozen=True, slots=True)
class GraphEdge:
    """A weighted graph edge retaining both endpoint identities."""

    source: RecordIdentity
    target: RecordIdentity
    edge_type: str
    weight: float

    def __iter__(self):
        yield self.source.source_id
        yield self.target.source_id
        yield self.edge_type
        yield self.weight

    def __getitem__(self, index: int):
        return (
            self.source.source_id,
            self.target.source_id,
            self.edge_type,
            self.weight,
        )[index]

    def __len__(self) -> int:
        return 4


# ===== Core domain types =====

@dataclass
class Chunk:
    """A discrete unit of content to be embedded and indexed.

    Chunks are derived from source records during ingestion. They carry
    enough context to reconstruct their parent record and to hydrate results.
    """

    chunk_id: str
    """Unique identifier for this chunk."""

    record_id: str
    """ID of the parent Record this chunk came from."""

    content: str
    """Plain-text chunk content to be embedded."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Chunk-level metadata (e.g., section headers, position in document)."""

    chunk_index: int = 0
    """Position of this chunk within its parent record."""

    content_hash: str = ""
    """SHA256 hash of content for change detection (computed on demand)."""

    def compute_content_hash(self) -> str:
        """Compute SHA256 hash of chunk content for change detection."""
        import hashlib

        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()

    @property
    def storage_key(self) -> str:
        """Canonical identity for a chunk treated as a ``chunk`` record."""
        return canonical_storage_key(None, "chunk", self.chunk_id)


@dataclass
class ChunkResult:
    """A single hydrated, scored chunk returned from a search query.

    Source-agnostic result shape: source-specific fields (e.g. a markdown
    note's header_path/file_path/project_id) live in `metadata` rather than
    as first-class attributes.
    """

    chunk_id: str
    """ID of the matched chunk."""

    record_id: str
    """ID of the parent Record this chunk came from."""

    score: float
    """Relevance score."""

    content: str = ""
    """Hydrated chunk content."""

    parent_chunk_id: str | None = None
    """ID of the parent chunk, if this is a child chunk."""

    parent_content: str | None = None
    """Content of the parent chunk, if hydrated."""

    provenance: "SearchResultProvenance | None" = None
    """Which strategies/adjustments contributed to this result's ranking."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Source-specific metadata (e.g. header_path, file_path, project_id)."""


@dataclass
class Record:
    """A source-agnostic record representing indexable content.

    Records are the contract between content sources and the kernel. A source
    adapts its native schema into Records; the kernel chunks, embeds, and
    indexes them. Records can carry pre-computed embeddings if the source
    already has them (avoids re-embedding during retrieval).
    """

    source_kind: str
    """
    Source type identifier: "note", "git_commit", "gmail", "jira", etc.
    Determines which adapter produced this record.
    """

    source_id: str
    """
    Stable, namespaced identifier within the source.
    Examples: "git:abc123def456", "gmail:msg-12345", "jira:CORE-999"
    """

    title: str
    """Human-readable title or headline."""

    body: str
    """Main content as plain text (extracted from HTML/Markdown/etc.)."""

    created_at: datetime
    """When the record was originally created in the source."""

    updated_at: datetime
    """Last modified time; used as the watermark for incremental sync."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Source-specific metadata (opaque to the core; preserved in results)."""

    uri: str | None = None
    """Permalink or file path for citation/navigation."""

    status: RecordStatus = RecordStatus.ACTIVE
    """Lifecycle status: active, stale, or archived."""

    embedding: Vector | None = None
    """Pre-computed embedding (if the source brought its own vectors)."""

    embedding_model: str | None = None
    """Model name that produced the embedding (if embedding is set)."""

    workspace_id: str | None = None
    """Optional workspace/tenant scope for the source identity."""

    indexed_text: str | None = None
    """Optional text override used for indexing while retaining ``body``."""

    def __post_init__(self) -> None:
        """Keep persisted timestamps comparable across source adapters."""
        self.created_at = _as_utc(self.created_at)
        self.updated_at = _as_utc(self.updated_at)

    @property
    def storage_key(self) -> str:
        """Canonical storage identity used by stores, fusion, and hydration."""
        return canonical_storage_key(self.workspace_id, self.source_kind, self.source_id)

    @property
    def identity(self) -> RecordIdentity:
        """Return the canonical identity carried by this record."""
        return RecordIdentity(self.workspace_id, self.source_kind, self.source_id)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary for storage or RPC."""
        return {
            "workspace_id": self.workspace_id,
            "source_kind": self.source_kind,
            "source_id": self.source_id,
            "title": self.title,
            "body": self.body,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
            "uri": self.uri,
            "status": self.status.value,
            "embedding": self.embedding,
            "embedding_model": self.embedding_model,
            "indexed_text": self.indexed_text,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Record":
        """Deserialize from a dictionary."""
        from datetime import datetime as dt

        # Parse ISO datetime strings
        created_at = data["created_at"]
        if isinstance(created_at, str):
            created_at = dt.fromisoformat(created_at)

        updated_at = data["updated_at"]
        if isinstance(updated_at, str):
            updated_at = dt.fromisoformat(updated_at)

        # Parse status enum
        status = data.get("status", RecordStatus.ACTIVE)
        if isinstance(status, str):
            status = RecordStatus(status)

        return cls(
            workspace_id=data.get("workspace_id"),
            source_kind=data["source_kind"],
            source_id=data["source_id"],
            title=data["title"],
            body=data["body"],
            created_at=created_at,
            updated_at=updated_at,
            metadata=data.get("metadata", {}),
            uri=data.get("uri"),
            status=status,
            embedding=data.get("embedding"),
            embedding_model=data.get("embedding_model"),
            indexed_text=data.get("indexed_text"),
        )


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


@dataclass(frozen=True)
class StrategyContribution:
    """Rank/score contributed by a single retrieval strategy to a result."""

    rank: int
    raw_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "raw_score": self.raw_score,
        }


@dataclass
class SearchResultProvenance:
    """Tracks which strategies/adjustments contributed to a result's ranking."""

    strategies: tuple[str, ...] = ()
    strategy_details: dict[str, StrategyContribution] = field(default_factory=dict)
    parent_expanded_from: str | None = None
    record_identity: RecordIdentity | None = None
    parent_expanded_from_identity: RecordIdentity | None = None

    def add_strategy(self, strategy: str, rank: int, raw_score: float) -> None:
        if strategy in self.strategy_details:
            return

        self.strategy_details[strategy] = StrategyContribution(
            rank=rank,
            raw_score=raw_score,
        )
        if strategy not in self.strategies:
            self.strategies = (*self.strategies, strategy)

    def clone(self) -> "SearchResultProvenance":
        return SearchResultProvenance(
            record_identity=self.record_identity,
            strategies=tuple(self.strategies),
            strategy_details=dict(self.strategy_details),
            parent_expanded_from=self.parent_expanded_from,
            parent_expanded_from_identity=self.parent_expanded_from_identity,
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, object] = {
            "strategies": list(self.strategies),
        }
        if self.record_identity is not None:
            result["record_identity"] = {
                "workspace_id": self.record_identity.workspace_id,
                "source_kind": self.record_identity.source_kind,
                "source_id": self.record_identity.source_id,
            }
        if self.strategy_details:
            result["strategy_details"] = {
                strategy: contribution.to_dict()
                for strategy, contribution in self.strategy_details.items()
            }

        adjustments: dict[str, object] = {}
        if self.parent_expanded_from is not None:
            adjustments["parent_expanded_from"] = self.parent_expanded_from
        if self.parent_expanded_from_identity is not None:
            adjustments["parent_expanded_from_identity"] = {
                "workspace_id": self.parent_expanded_from_identity.workspace_id,
                "source_kind": self.parent_expanded_from_identity.source_kind,
                "source_id": self.parent_expanded_from_identity.source_id,
            }
        if adjustments:
            result["adjustments"] = adjustments

        return result


@dataclass
class CompressionStats:
    """Counters describing how many results survived each compression stage."""

    original_count: int
    after_threshold: int
    after_content_dedup: int
    after_ngram_dedup: int
    after_dedup: int
    after_doc_limit: int
    clusters_merged: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_count": self.original_count,
            "after_threshold": self.after_threshold,
            "after_content_dedup": self.after_content_dedup,
            "after_ngram_dedup": self.after_ngram_dedup,
            "after_dedup": self.after_dedup,
            "after_doc_limit": self.after_doc_limit,
            "clusters_merged": self.clusters_merged,
        }


@dataclass
class SearchStrategyStats:
    """Per-strategy candidate counts surfaced from a search query."""

    vector_count: int | None = None
    keyword_count: int | None = None
    graph_count: int | None = None
    tag_expansion_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if self.vector_count is not None:
            result["vector_count"] = self.vector_count
        if self.keyword_count is not None:
            result["keyword_count"] = self.keyword_count
        if self.graph_count is not None:
            result["graph_count"] = self.graph_count
        if self.tag_expansion_count is not None:
            result["tag_expansion_count"] = self.tag_expansion_count
        return result
