"""Optional parent-first retrieval for structured source adapters."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace

from searchkernel.domain import RecordIdentity, ScoredRef
from searchkernel.ports.content_source import HierarchicalSearchableSource


@dataclass(frozen=True, slots=True)
class HierarchicalRetrievalConfig:
    """Explicit limits for parent-first retrieval."""

    enabled: bool = False
    parent_candidate_limit: int = 10
    max_parents: int = 5
    children_per_parent: int = 10
    include_parent_fallback: bool = True

    def __post_init__(self) -> None:
        for name in (
            "parent_candidate_limit",
            "max_parents",
            "children_per_parent",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")

    @property
    def parent_k(self) -> int:
        """Alias for the coarse candidate budget."""
        return self.parent_candidate_limit

    @property
    def child_k(self) -> int:
        """Alias for the per-parent fine candidate budget."""
        return self.children_per_parent


async def search_hierarchical(
    source: HierarchicalSearchableSource,
    query: str,
    *,
    k: int,
    config: HierarchicalRetrievalConfig,
    filters: dict[str, object] | None = None,
) -> list[ScoredRef]:
    """Retrieve fine records within the strongest coarse parent records."""
    if k < 1:
        return []
    parents = _top_unique(
        await source.search_parents(
            query,
            config.parent_candidate_limit,
            filters,
        ),
        config.max_parents,
    )
    if not parents:
        return []

    parent_keys = [parent.storage_key for parent in parents]
    parents_by_source_id: dict[str, list[str]] = {}
    for parent in parents:
        parents_by_source_id.setdefault(parent.source_id, []).append(parent.storage_key)

    children = await _search_children(
        source,
        query,
        parent_keys,
        parents_by_source_id,
        config.children_per_parent,
        filters,
    )
    children_by_parent: dict[str, list[ScoredRef]] = {
        parent_key: [] for parent_key in parent_keys
    }
    for child in children:
        parent_key = _parent_key(child, parents_by_source_id)
        if parent_key in children_by_parent:
            children_by_parent[parent_key].append(child)

    results: list[ScoredRef] = []
    for parent in parents:
        parent_children = sorted(
            children_by_parent[parent.storage_key],
            key=lambda item: (-item.score, item.storage_key),
        )
        if parent_children:
            results.extend(
                _with_parent_provenance(child, parent)
                for child in parent_children[: config.children_per_parent]
            )
        elif config.include_parent_fallback:
            results.append(_with_parent_provenance(parent, parent, fallback=True))
    return results[:k]


async def _search_children(
    source: HierarchicalSearchableSource,
    query: str,
    parent_keys: list[str],
    parents_by_source_id: dict[str, list[str]],
    k: int,
    filters: dict[str, object] | None,
) -> list[ScoredRef]:
    children = list(await source.search_children(query, parent_keys, k, filters))
    if _has_resolvable_parent(children, parent_keys, parents_by_source_id):
        return children
    if any(len(keys) != 1 for keys in parents_by_source_id.values()):
        return children

    legacy_parent_ids = [
        RecordIdentity.from_storage_key(parent_key).source_id
        for parent_key in parent_keys
    ]
    return list(await source.search_children(query, legacy_parent_ids, k, filters))


def _has_resolvable_parent(
    children: Iterable[ScoredRef],
    parent_keys: list[str],
    parents_by_source_id: dict[str, list[str]],
) -> bool:
    selected_parent_keys = set(parent_keys)
    return any(
        _parent_key(child, parents_by_source_id) in selected_parent_keys
        for child in children
    )


def _top_unique(results: Iterable[ScoredRef], limit: int) -> list[ScoredRef]:
    unique: dict[str, ScoredRef] = {}
    for result in results:
        unique.setdefault(result.storage_key, result)
    return sorted(unique.values(), key=lambda item: (-item.score, item.storage_key))[:limit]


def _parent_key(
    result: ScoredRef,
    parents_by_source_id: dict[str, list[str]],
) -> str | None:
    for field in ("parent_storage_key", "parent_identity", "parent_id"):
        key = _storage_key(result.metadata.get(field))
        if key is not None:
            return key

    value = result.metadata.get("parent_id")
    if value is None:
        return None
    matches = parents_by_source_id.get(str(value), ())
    return matches[0] if len(matches) == 1 else None


def _storage_key(value: object) -> str | None:
    if isinstance(value, RecordIdentity):
        return value.storage_key
    if isinstance(value, Mapping):
        workspace_id = value.get("workspace_id")
        source_kind = value.get("source_kind")
        source_id = value.get("source_id")
        if (
            (workspace_id is None or isinstance(workspace_id, str))
            and isinstance(source_kind, str)
            and isinstance(source_id, str)
        ):
            return RecordIdentity(workspace_id, source_kind, source_id).storage_key
        return None
    if not isinstance(value, str) or not value.startswith("record:"):
        return None
    try:
        RecordIdentity.from_storage_key(value)
    except (TypeError, ValueError):
        return None
    return value


def _with_parent_provenance(
    result: ScoredRef,
    parent: ScoredRef,
    *,
    fallback: bool = False,
) -> ScoredRef:
    metadata = dict(result.metadata)
    metadata.setdefault("parent_id", parent.source_id)
    metadata.setdefault("parent_storage_key", parent.storage_key)
    metadata["hierarchical_provenance"] = {
        "parent_id": parent.source_id,
        "parent_storage_key": parent.storage_key,
        "parent_score": parent.score,
        "kind": "parent_fallback" if fallback else "fine_child",
    }
    return replace(result, metadata=metadata)


__all__ = ["HierarchicalRetrievalConfig", "search_hierarchical"]
