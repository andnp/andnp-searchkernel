from searchkernel.search.classifier import QueryType
from searchkernel.search.query_plan import QueryRouter, QueryRouterConfig


def test_artifact_plan_prefers_keyword_and_bounds_vector_candidates() -> None:
    plan = QueryRouter().route(
        "src/search_kernel.py",
        limit=3,
        keyword_available=True,
        vector_available=True,
        graph_available=True,
    )

    assert plan.query_type == QueryType.FACTUAL
    assert plan.signals.artifact
    assert plan.keyword_enabled
    assert plan.vector_enabled
    assert plan.vector_candidates_keyword_bounded
    assert plan.keyword_candidate_budget == 15
    assert plan.vector_candidate_budget == 15


def test_lane_budgets_are_independently_configurable() -> None:
    plan = QueryRouter(
        QueryRouterConfig(
            keyword_candidate_multiplier=2,
            vector_candidate_multiplier=4,
            minimum_candidate_limit=1,
        )
    ).route(
        "what is caching?",
        limit=3,
        keyword_available=True,
        vector_available=True,
        graph_available=False,
    )

    assert plan.keyword_candidate_budget == 6
    assert plan.vector_candidate_budget == 12
    assert plan.fusion_weight_map["vector"] > plan.fusion_weight_map["keyword"]


def test_disabled_graph_is_diagnostic_only() -> None:
    plan = QueryRouter().route(
        "what depends on this module?",
        limit=1,
        keyword_available=True,
        vector_available=False,
        graph_available=True,
        graph_enabled=False,
    )

    assert not plan.graph_enabled
    assert "graph:disabled" in plan.diagnostic_skip_reasons


def test_ordinary_query_skips_graph_with_an_explicitly_available_store() -> None:
    plan = QueryRouter().route(
        "what is caching?",
        limit=1,
        keyword_available=True,
        vector_available=True,
        graph_available=True,
    )

    assert not plan.graph_enabled
    assert "graph:query_not_relationship" in plan.diagnostic_skip_reasons


def test_relationship_query_enables_graph_expansion() -> None:
    plan = QueryRouter().route(
        "what depends on this module?",
        limit=1,
        keyword_available=True,
        vector_available=True,
        graph_available=True,
    )

    assert plan.signals.relationship
    assert plan.graph_enabled
    assert "graph:query_not_relationship" not in plan.diagnostic_skip_reasons


def test_unavailable_graph_is_diagnostic_only() -> None:
    plan = QueryRouter().route(
        "what depends on this module?",
        limit=1,
        keyword_available=True,
        vector_available=True,
        graph_available=False,
    )

    assert not plan.graph_enabled
    assert "graph:unavailable" in plan.diagnostic_skip_reasons
