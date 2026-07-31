import pytest

from searchkernel.search.bounded_graph import (
    GraphExpansionProvenance,
    TypedGraphEdge,
    expand_bounded_typed_graph,
)


def test_expansion_limits_sorted_seeds_and_neighbors_per_seed():
    edges = {
        "seed_high": [
            TypedGraphEdge("high_first", "supported"),
            TypedGraphEdge("high_second", "supported"),
        ],
        "seed_middle": [TypedGraphEdge("middle_first", "supported")],
        "seed_low": [TypedGraphEdge("low_first", "supported")],
    }
    visited_seeds = []

    def read_edges(seed_id):
        visited_seeds.append(seed_id)
        return edges[seed_id]

    result = expand_bounded_typed_graph(
        {"seed_low": 0.1, "seed_high": 0.9, "seed_middle": 0.5},
        read_edges,
        {"supported": 1.0},
        max_seed_count=2,
        max_neighbors_per_seed=1,
    )

    assert set(result) == {"high_first", "middle_first"}
    assert visited_seeds == ["seed_high", "seed_middle"]


def test_unsupported_edges_do_not_consume_neighbor_limit():
    result = expand_bounded_typed_graph(
        {"seed": 1.0},
        lambda _: [
            TypedGraphEdge("ignored", "unsupported"),
            TypedGraphEdge("included", "supported"),
        ],
        {"supported": 0.5},
        max_seed_count=1,
        max_neighbors_per_seed=1,
    )

    assert set(result) == {"included"}


def test_duplicate_targets_keep_highest_contribution_and_provenance():
    result = expand_bounded_typed_graph(
        {"first_seed": 0.8, "winning_seed": 0.9},
        lambda seed: [
            TypedGraphEdge(
                "target",
                "weak" if seed == "first_seed" else "strong",
            )
        ],
        {"weak": 1.0, "strong": 1.0},
        max_seed_count=2,
        max_neighbors_per_seed=1,
    )

    assert result["target"].contribution == 0.9
    assert result["target"].provenance == GraphExpansionProvenance(
        seed_id="winning_seed",
        edge_type="strong",
    )


def test_duplicate_equal_contributions_keep_first_provenance():
    result = expand_bounded_typed_graph(
        {"first_seed": 0.8, "second_seed": 0.8},
        lambda seed: [TypedGraphEdge("target", seed)],
        {"first_seed": 1.0, "second_seed": 1.0},
        max_seed_count=2,
        max_neighbors_per_seed=1,
    )

    assert result["target"].provenance.seed_id == "first_seed"


@pytest.mark.parametrize(
    ("max_seed_count", "max_neighbors_per_seed"),
    [(0, 1), (-1, 1), (1, 0), (1, -1)],
)
def test_invalid_limits_raise_value_error(max_seed_count, max_neighbors_per_seed):
    with pytest.raises(ValueError):
        expand_bounded_typed_graph(
            {},
            lambda _: [],
            {},
            max_seed_count=max_seed_count,
            max_neighbors_per_seed=max_neighbors_per_seed,
        )
