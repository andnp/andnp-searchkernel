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
        "what is dependency injection?",
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
