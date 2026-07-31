"""Pure bounded expansion for typed directed graphs."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TypedGraphEdge[NodeT: Hashable, EdgeTypeT: Hashable]:
    """A typed outgoing edge to another graph node."""

    target_id: NodeT
    edge_type: EdgeTypeT


@dataclass(frozen=True, slots=True)
class GraphExpansionProvenance[NodeT: Hashable, EdgeTypeT: Hashable]:
    """The seed and edge type that produced an expanded target."""

    seed_id: NodeT
    edge_type: EdgeTypeT


@dataclass(frozen=True, slots=True)
class GraphExpansionResult[NodeT: Hashable, EdgeTypeT: Hashable]:
    """The winning contribution and its source provenance."""

    contribution: float
    provenance: GraphExpansionProvenance[NodeT, EdgeTypeT]


def expand_bounded_typed_graph[
    NodeT: Hashable, EdgeTypeT: Hashable
](
    ranked_seed_scores: Mapping[NodeT, float],
    outgoing_edges: Callable[[NodeT], Iterable[TypedGraphEdge[NodeT, EdgeTypeT]]],
    edge_type_discounts: Mapping[EdgeTypeT, float],
    max_seed_count: int,
    max_neighbors_per_seed: int,
) -> dict[NodeT, GraphExpansionResult[NodeT, EdgeTypeT]]:
    """Expand ranked seeds through discounted typed edges.

    Seeds are processed in descending score order, limited to
    ``max_seed_count``. For each seed, unsupported edge types are skipped and
    supported edges are limited to ``max_neighbors_per_seed``. Each supported
    edge contributes ``seed_score * discount``; duplicate targets retain the
    highest contribution and its seed/edge provenance. Equal contributions
    retain the first result encountered.

    ``max_seed_count`` and ``max_neighbors_per_seed`` must both be positive.
    """
    _validate_limit("max_seed_count", max_seed_count)
    _validate_limit("max_neighbors_per_seed", max_neighbors_per_seed)

    expanded: dict[NodeT, GraphExpansionResult[NodeT, EdgeTypeT]] = {}
    seed_items = sorted(
        ranked_seed_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:max_seed_count]

    for seed_id, seed_score in seed_items:
        expanded_neighbors = 0
        for edge in outgoing_edges(seed_id):
            discount = edge_type_discounts.get(edge.edge_type)
            if discount is None:
                continue

            expanded_neighbors += 1
            if expanded_neighbors > max_neighbors_per_seed:
                break

            contribution = seed_score * discount
            current = expanded.get(edge.target_id)
            if current is None or contribution > current.contribution:
                expanded[edge.target_id] = GraphExpansionResult(
                    contribution=contribution,
                    provenance=GraphExpansionProvenance(
                        seed_id=seed_id,
                        edge_type=edge.edge_type,
                    ),
                )

    return expanded


def _validate_limit(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")
