"""Optional parent-first retrieval for structured source adapters."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace

from searchkernel.domain import ScoredRef
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

    parent_ids = [parent.source_id for parent in parents]
    children = await source.search_children(
        query,
        parent_ids,
        config.children_per_parent,
        filters,
    )
    children_by_parent: dict[str, list[ScoredRef]] = {parent_id: [] for parent_id in parent_ids}
    for child in children:
        parent_id = _parent_id(child)
        if parent_id in children_by_parent:
            children_by_parent[parent_id].append(child)

    results: list[ScoredRef] = []
    for parent in parents:
        parent_children = sorted(
            children_by_parent[parent.source_id],
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


def _top_unique(results: Iterable[ScoredRef], limit: int) -> list[ScoredRef]:
    unique: dict[str, ScoredRef] = {}
    for result in results:
        unique.setdefault(result.storage_key, result)
    return sorted(unique.values(), key=lambda item: (-item.score, item.storage_key))[:limit]


def _parent_id(result: ScoredRef) -> str | None:
    value = result.metadata.get("parent_id")
    return str(value) if value is not None else None


def _with_parent_provenance(
    result: ScoredRef,
    parent: ScoredRef,
    *,
    fallback: bool = False,
) -> ScoredRef:
    metadata = dict(result.metadata)
    metadata.setdefault("parent_id", parent.source_id)
    metadata["hierarchical_provenance"] = {
        "parent_id": parent.source_id,
        "parent_storage_key": parent.storage_key,
        "parent_score": parent.score,
        "kind": "parent_fallback" if fallback else "fine_child",
    }
    return replace(result, metadata=metadata)


__all__ = ["HierarchicalRetrievalConfig", "search_hierarchical"]
