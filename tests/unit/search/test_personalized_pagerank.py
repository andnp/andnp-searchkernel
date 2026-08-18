import pytest

from searchkernel.search.bounded_graph import (
    GraphExpansionProvenance,
    TypedGraphEdge,
    expand_bounded_typed_graph,
)
from searchkernel.search.personalized_pagerank import expand_personalized_pagerank


def test_known_answer_two_hop_chain_matches_hand_computed_mass():
    # A -[rel]-> B -[rel]-> C, damping=0.5, discount=0.5.
    # Each hop multiplies mass by damping * discount = 0.25.
    edges = {
        "A": [TypedGraphEdge("B", "rel")],
        "B": [TypedGraphEdge("C", "rel")],
        "C": [],
    }

    result = expand_personalized_pagerank(
        {"A": 1.0},
        lambda node: edges[node],
        {"rel": 0.5},
        max_seed_count=1,
        max_neighbors_per_seed=5,
        damping=0.5,
        max_depth=3,
        minimum_contribution=0.0,
    )

    assert result["B"].contribution == pytest.approx(0.25)
    assert result["B"].provenance == GraphExpansionProvenance(seed_id="A", edge_type="rel")
    assert result["C"].contribution == pytest.approx(0.0625)
    assert result["C"].provenance == GraphExpansionProvenance(seed_id="A", edge_type="rel")


def test_transitivity_reaches_two_hop_node_that_one_hop_cannot():
    edges = {
        "A": [TypedGraphEdge("B", "rel")],
        "B": [TypedGraphEdge("C", "rel")],
        "C": [],
    }

    ppr_result = expand_personalized_pagerank(
        {"A": 1.0},
        lambda node: edges[node],
        {"rel": 0.5},
        max_seed_count=1,
        max_neighbors_per_seed=5,
        damping=0.5,
        max_depth=3,
        minimum_contribution=0.0,
    )
    one_hop_result = expand_bounded_typed_graph(
        {"A": 1.0},
        lambda node: edges[node],
        {"rel": 0.5},
        max_seed_count=1,
        max_neighbors_per_seed=5,
    )

    assert "C" in ppr_result
    assert "C" not in one_hop_result
    assert set(one_hop_result) == {"B"}


def test_cycle_terminates_via_max_depth():
    edges = {
        "A": [TypedGraphEdge("B", "rel")],
        "B": [TypedGraphEdge("A", "rel")],
    }

    result = expand_personalized_pagerank(
        {"A": 1.0},
        lambda node: edges[node],
        {"rel": 0.9},
        max_seed_count=1,
        max_neighbors_per_seed=5,
        damping=1.0,
        max_depth=4,
        minimum_contribution=0.0,
    )

    assert set(result) == {"A", "B"}


def test_self_loop_accumulates_but_still_terminates():
    edges = {"A": [TypedGraphEdge("A", "rel")]}

    result = expand_personalized_pagerank(
        {"A": 1.0},
        lambda node: edges[node],
        {"rel": 0.9},
        max_seed_count=1,
        max_neighbors_per_seed=5,
        damping=1.0,
        max_depth=5,
        minimum_contribution=0.0,
    )

    # Each of the 5 hops re-contributes mass * damping * discount = 0.9x the
    # previous hop's mass, so the total is the geometric sum sum(0.9**k, k=1..5).
    expected_total = sum(0.9**k for k in range(1, 6))
    assert result["A"].contribution == pytest.approx(expected_total, rel=1e-9)


def test_node_with_no_outgoing_edges_does_not_propagate():
    result = expand_personalized_pagerank(
        {"A": 1.0},
        lambda node: {"A": [TypedGraphEdge("B", "rel")], "B": []}[node],
        {"rel": 1.0},
        max_seed_count=1,
        max_neighbors_per_seed=5,
        damping=1.0,
        max_depth=5,
        minimum_contribution=0.0,
    )

    assert set(result) == {"B"}


def test_all_unknown_edge_types_are_skipped():
    result = expand_personalized_pagerank(
        {"A": 1.0},
        lambda node: [TypedGraphEdge("B", "unknown")],
        {"known": 1.0},
        max_seed_count=1,
        max_neighbors_per_seed=5,
    )

    assert result == {}


def test_empty_seeds_return_empty_result():
    result = expand_personalized_pagerank(
        {},
        lambda node: [],
        {},
        max_seed_count=1,
        max_neighbors_per_seed=1,
    )

    assert result == {}


def test_damping_zero_never_propagates_mass():
    result = expand_personalized_pagerank(
        {"A": 1.0},
        lambda node: [TypedGraphEdge("B", "rel")],
        {"rel": 1.0},
        max_seed_count=1,
        max_neighbors_per_seed=5,
        damping=0.0,
    )

    assert result == {}


def test_max_seed_count_caps_number_of_seeds_expanded():
    visited_seeds = []

    def read_edges(node):
        visited_seeds.append(node)
        return [TypedGraphEdge(f"{node}-target", "rel")]

    result = expand_personalized_pagerank(
        {"low": 0.1, "high": 0.9, "middle": 0.5},
        read_edges,
        {"rel": 1.0},
        max_seed_count=2,
        max_neighbors_per_seed=5,
        max_depth=1,
        minimum_contribution=0.0,
    )

    assert set(visited_seeds) == {"high", "middle"}
    assert set(result) == {"high-target", "middle-target"}


def test_max_neighbors_per_seed_caps_fan_out_deterministically():
    result = expand_personalized_pagerank(
        {"seed": 1.0},
        lambda node: (
            edge
            for edge in [
                TypedGraphEdge("target-z", "low"),
                TypedGraphEdge("target-b", "high"),
                TypedGraphEdge("target-a", "high"),
                TypedGraphEdge("target-c", "medium"),
            ]
        ),
        {"low": 0.1, "high": 0.9, "medium": 0.5},
        max_seed_count=1,
        max_neighbors_per_seed=2,
        minimum_contribution=0.0,
    )

    assert set(result) == {"target-a", "target-b"}


def test_max_depth_binds_before_mass_floor_would():
    edges = {
        "A": [TypedGraphEdge("B", "rel")],
        "B": [TypedGraphEdge("C", "rel")],
        "C": [TypedGraphEdge("D", "rel")],
    }

    result = expand_personalized_pagerank(
        {"A": 1.0},
        lambda node: edges.get(node, []),
        {"rel": 1.0},
        max_seed_count=1,
        max_neighbors_per_seed=5,
        damping=1.0,
        max_depth=2,
        minimum_contribution=0.0,
    )

    assert set(result) == {"B", "C"}
    assert "D" not in result


def test_minimum_contribution_prunes_before_max_depth():
    edges = {
        "A": [TypedGraphEdge("B", "rel")],
        "B": [TypedGraphEdge("C", "rel")],
    }

    result = expand_personalized_pagerank(
        {"A": 1.0},
        lambda node: edges.get(node, []),
        {"rel": 0.1},
        max_seed_count=1,
        max_neighbors_per_seed=5,
        damping=0.1,
        max_depth=5,
        minimum_contribution=0.02,
    )

    # B gets 1.0 * 0.1 * 0.1 = 0.01, already below the floor, so it is
    # recorded but never propagated onward to C.
    assert set(result) == {"B"}


def test_max_visited_nodes_caps_outgoing_edges_calls():
    call_count = 0

    def read_edges(node):
        nonlocal call_count
        call_count += 1
        return [TypedGraphEdge(f"{node}-child", "rel")]

    result = expand_personalized_pagerank(
        {"seed-1": 0.9, "seed-2": 0.5},
        read_edges,
        {"rel": 1.0},
        max_seed_count=2,
        max_neighbors_per_seed=5,
        damping=1.0,
        max_depth=3,
        max_visited_nodes=1,
        minimum_contribution=0.0,
    )

    assert call_count == 1
    assert set(result) == {"seed-1-child"}


def test_determinism_across_repeated_runs():
    edges = {
        "A": [TypedGraphEdge("B", "rel"), TypedGraphEdge("C", "rel")],
        "B": [TypedGraphEdge("C", "rel")],
        "C": [TypedGraphEdge("A", "rel")],
    }

    def run():
        return expand_personalized_pagerank(
            {"A": 0.7, "B": 0.3},
            lambda node: edges[node],
            {"rel": 0.6},
            max_seed_count=2,
            max_neighbors_per_seed=2,
            damping=0.8,
            max_depth=4,
            minimum_contribution=1e-6,
        )

    first = run()
    for _ in range(5):
        assert run() == first


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_seed_count": 0},
        {"max_seed_count": -1},
        {"max_neighbors_per_seed": 0},
        {"max_neighbors_per_seed": -1},
        {"max_depth": 0},
        {"max_depth": -1},
        {"max_visited_nodes": 0},
        {"max_visited_nodes": -1},
        {"damping": -0.1},
        {"damping": 1.1},
        {"minimum_contribution": -1e-9},
    ],
)
def test_invalid_parameters_raise_value_error(kwargs):
    defaults = {
        "max_seed_count": 1,
        "max_neighbors_per_seed": 1,
        "damping": 0.85,
        "max_depth": 3,
        "max_visited_nodes": 10,
        "minimum_contribution": 1e-4,
    }
    defaults.update(kwargs)

    with pytest.raises(ValueError):
        expand_personalized_pagerank(
            {"seed": 1.0},
            lambda _: [],
            {},
            **defaults,
        )
