"""Pure bounded personalized PageRank expansion for typed directed graphs.

Drop-in alternative to :func:`searchkernel.search.bounded_graph.expand_bounded_typed_graph`
that follows edges beyond a single hop by simulating a bounded random walk with
restart, seeded from the same ranked candidates. See module docstring on
``expand_personalized_pagerank`` for the full contract.
"""

from __future__ import annotations

import heapq
from collections.abc import Callable, Hashable, Iterable, Mapping

from searchkernel.search.bounded_graph import (
    GraphExpansionProvenance,
    GraphExpansionResult,
    TypedGraphEdge,
    _stable_key,
    _validate_limit,
)


def expand_personalized_pagerank[
    NodeT: Hashable, EdgeTypeT: Hashable
](
    ranked_seed_scores: Mapping[NodeT, float],
    outgoing_edges: Callable[[NodeT], Iterable[TypedGraphEdge[NodeT, EdgeTypeT]]],
    edge_type_discounts: Mapping[EdgeTypeT, float],
    *,
    max_seed_count: int,
    max_neighbors_per_seed: int,
    damping: float = 0.85,
    max_depth: int = 3,
    max_visited_nodes: int = 1_000,
    minimum_contribution: float = 1e-4,
) -> dict[NodeT, GraphExpansionResult[NodeT, EdgeTypeT]]:
    """Expand ranked seeds through a bounded personalized PageRank random walk.

    Seeds are sorted by descending score and limited to ``max_seed_count``,
    exactly as :func:`expand_bounded_typed_graph` does. Their scores are then
    renormalised so they sum to 1: this is the restart distribution the walk
    keeps returning to. Each seed then runs an independent bounded random walk:
    at every hop, a node's current mass spreads to its supported outgoing
    edges (unsupported edge types are skipped, exactly as the 1-hop version
    skips them), scaled by ``damping`` and by the edge type's discount.
    ``1 - damping`` is the probability the walk restarts rather than
    continuing, so it is simply the fraction of mass that is *not* propagated
    onward at that hop; it is not added back anywhere because each seed's walk
    is already anchored to its own restart node. Per-seed walks are additive
    (personalized PageRank over a mixture of restart points is the mixture of
    the single-source personalized PageRank vectors), so a node's total
    contribution is the sum of the mass it receives across all seeds and all
    paths.

    Per-hop fan-out is bounded by ``max_neighbors_per_seed``, selected the
    same way :func:`expand_bounded_typed_graph` selects it: descending
    discount, then a stable key on target and edge type, so ordering is
    deterministic even under a high out-degree.

    Provenance choice: a node can be reached from a given seed by many paths,
    each starting with a different first hop. This function records, per
    seed, the mass contributed by each distinct *first-hop edge type* out of
    that seed, then reports the (seed, edge_type) pair with the largest such
    mass as the node's provenance -- i.e. the ORIGINATING SEED and the edge
    type of the FIRST hop away from it, not the last hop before the node was
    reached. The alternative (last hop) is equally defensible -- it describes
    the immediate neighbourhood instead of the causal origin -- but first-hop
    was chosen because it lets a caller trace an expanded node back to
    "which of my retrieved candidates put this here", which is the same
    question :func:`expand_bounded_typed_graph`'s seed/edge-type provenance
    answers for a single hop.

    Hard bounds, always applied, so a dense or cyclic graph cannot make this
    run away:

    - ``max_depth`` caps the number of hops any walk takes from its seed.
      Combined with the fact that every hop multiplies mass by
      ``damping * discount`` (both in [0, 1]), this guarantees termination
      through cycles and self-loops even when ``damping == 1.0`` and a
      discount is ``1.0``, since ``max_depth`` still bounds hop count.
    - ``minimum_contribution`` floors mass: any node whose incoming mass at a
      hop would fall below this floor is not propagated further, pruning
      negligible branches early.
    - ``max_visited_nodes`` caps the total number of distinct nodes this
      expansion is willing to call ``outgoing_edges`` on, across every seed's
      walk combined. Nodes beyond the cap simply are not expanded further --
      any mass that already reached them is still recorded in the result, but
      they do not propagate onward.

    None of the above is an error condition: hitting a cap means the
    expansion returns whatever it has accumulated so far, exactly like a
    partial 1-hop expansion would if fewer than ``max_neighbors_per_seed``
    edges existed.

    ``outgoing_edges`` is treated as an I/O call. It is invoked at most once
    per distinct node for the lifetime of one call to this function -- the
    result is cached the first time a node is expanded and reused for every
    later hop or seed that reaches the same node. Worst case, this makes
    exactly ``max_visited_nodes`` calls (one per distinct node up to the
    cap); it can never make more, and it makes fewer whenever the walk
    terminates naturally (mass floor or depth) before the cap is reached.

    Determinism: seeds are processed in a fixed (sorted) order, nodes within
    a depth layer are processed in a fixed (mass descending, then stable-key)
    order, and per-node fan-out uses the same stable tie-breaking as
    :func:`expand_bounded_typed_graph`. No set is iterated for anything that
    affects output, and there is no randomness anywhere -- the same inputs
    always produce the same result dict, including provenance and floating
    point contributions.

    Edge cases:

    - Empty ``ranked_seed_scores``, or seed scores summing to zero or less,
      produce an empty result.
    - A node with no outgoing edges, or whose edges are all unsupported
      types, simply does not propagate past it.
    - Self-loops and cycles terminate via ``max_depth`` and
      ``minimum_contribution`` as described above.
    - ``damping == 0.0`` means no mass ever leaves a seed, so the result is
      always empty (every walk stops after its first, un-propagated step).
    - ``damping == 1.0`` means no restart probability is subtracted per hop;
      decay then comes only from edge discounts and, ultimately, from
      ``max_depth``.

    ``max_seed_count``, ``max_neighbors_per_seed``, ``max_depth``, and
    ``max_visited_nodes`` must all be positive. ``damping`` must be within
    ``[0.0, 1.0]``. ``minimum_contribution`` must be non-negative.
    """
    _validate_limit("max_seed_count", max_seed_count)
    _validate_limit("max_neighbors_per_seed", max_neighbors_per_seed)
    _validate_limit("max_depth", max_depth)
    _validate_limit("max_visited_nodes", max_visited_nodes)
    if not 0.0 <= damping <= 1.0:
        raise ValueError("damping must be between 0.0 and 1.0")
    if minimum_contribution < 0.0:
        raise ValueError("minimum_contribution must be non-negative")

    seed_items = sorted(
        ranked_seed_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:max_seed_count]
    total_seed_score = sum(score for _, score in seed_items)
    if not seed_items or total_seed_score <= 0.0:
        return {}

    edge_cache: dict[NodeT, list[tuple[TypedGraphEdge[NodeT, EdgeTypeT], float]]] = {}
    visited_count = 0

    def top_supported_edges(
        node: NodeT,
    ) -> list[tuple[TypedGraphEdge[NodeT, EdgeTypeT], float]]:
        nonlocal visited_count
        cached = edge_cache.get(node)
        if cached is not None:
            return cached
        if visited_count >= max_visited_nodes:
            return []
        visited_count += 1
        supported_edges = (
            (edge, discount)
            for edge in outgoing_edges(node)
            if (discount := edge_type_discounts.get(edge.edge_type)) is not None
        )
        top_edges = heapq.nsmallest(
            max_neighbors_per_seed,
            supported_edges,
            key=lambda item: (
                -item[1],
                _stable_key(item[0].target_id),
                _stable_key(item[0].edge_type),
            ),
        )
        edge_cache[node] = top_edges
        return top_edges

    # contributions[target][(seed_id, first_hop_edge_type)] = accumulated mass
    contributions: dict[NodeT, dict[tuple[NodeT, EdgeTypeT], float]] = {}

    for seed_id, seed_score in seed_items:
        restart_mass = seed_score / total_seed_score
        # frontier[node][first_hop_edge_type] = mass; ``None`` marks mass that
        # has not yet taken its first hop away from the seed.
        frontier: dict[NodeT, dict[EdgeTypeT | None, float]] = {
            seed_id: {None: restart_mass}
        }

        for _ in range(max_depth):
            if not frontier:
                break
            layered = sorted(
                (
                    (node, sum(by_hop.values()), by_hop)
                    for node, by_hop in frontier.items()
                ),
                key=lambda item: (-item[1], _stable_key(item[0])),
            )
            next_frontier: dict[NodeT, dict[EdgeTypeT | None, float]] = {}
            for node, total_mass, by_hop in layered:
                if total_mass < minimum_contribution:
                    continue
                for edge, discount in top_supported_edges(node):
                    target = edge.target_id
                    for first_hop_key, hop_mass in by_hop.items():
                        propagated = hop_mass * damping * discount
                        if propagated <= 0.0:
                            continue
                        first_hop = (
                            edge.edge_type if first_hop_key is None else first_hop_key
                        )
                        target_contribs = contributions.setdefault(target, {})
                        key = (seed_id, first_hop)
                        target_contribs[key] = (
                            target_contribs.get(key, 0.0) + propagated
                        )
                        # The mass is always recorded above, but only carried
                        # forward for further propagation if it clears the
                        # floor -- below it, the branch is pruned early.
                        if propagated < minimum_contribution:
                            continue
                        target_hops = next_frontier.setdefault(target, {})
                        target_hops[first_hop] = (
                            target_hops.get(first_hop, 0.0) + propagated
                        )
            frontier = next_frontier

    expanded: dict[NodeT, GraphExpansionResult[NodeT, EdgeTypeT]] = {}
    for node, by_key in contributions.items():
        total_contribution = sum(by_key.values())
        # ``max`` is stable: among equal masses it keeps the first key
        # inserted, and insertion order is itself deterministic (fixed seed
        # and depth-layer processing order), so ties resolve deterministically.
        best_key, _ = max(by_key.items(), key=lambda item: item[1])
        expanded[node] = GraphExpansionResult(
            contribution=total_contribution,
            provenance=GraphExpansionProvenance(
                seed_id=best_key[0],
                edge_type=best_key[1],
            ),
        )

    return expanded
