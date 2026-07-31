"""Deterministic query routing plans for the canonical record pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from searchkernel.search.classifier import (
    QuerySignals,
    QueryType,
    analyze_query,
    classify_query,
    get_adaptive_weights,
)


@dataclass(frozen=True, slots=True)
class QueryPlan:
    """Immutable retrieval decisions made before executing a query."""

    query_type: QueryType
    signals: QuerySignals
    keyword_enabled: bool
    vector_enabled: bool
    graph_enabled: bool
    keyword_candidate_budget: int
    vector_candidate_budget: int
    fusion_weights: tuple[tuple[str, float], ...]
    vector_candidates_keyword_bounded: bool
    graph_depth: int
    graph_seed_budget: int
    rerank_budget: int
    expansion_strategy: str | None = None
    diagnostic_skip_reasons: tuple[str, ...] = ()

    @property
    def enabled_lanes(self) -> tuple[str, ...]:
        return tuple(
            lane
            for lane, enabled in (
                ("keyword", self.keyword_enabled),
                ("vector", self.vector_enabled),
                ("graph", self.graph_enabled),
            )
            if enabled
        )

    @property
    def lane_budgets(self) -> dict[str, int]:
        return {
            "keyword": self.keyword_candidate_budget,
            "vector": self.vector_candidate_budget,
            "graph": self.graph_seed_budget,
        }

    @property
    def fusion_weight_map(self) -> dict[str, float]:
        return dict(self.fusion_weights)


@dataclass(frozen=True, slots=True)
class QueryRouterConfig:
    """Measured, backwards-compatible routing bounds."""

    candidate_multiplier: int = 5
    minimum_candidate_limit: int = 1
    keyword_candidate_budget: int | None = None
    vector_candidate_budget: int | None = None
    keyword_candidate_multiplier: int | None = None
    vector_candidate_multiplier: int | None = None
    graph_seed_budget: int = 10
    graph_depth: int = 1
    rerank_budget: int = 0
    expansion_enabled: bool = False
    expansion_strategy: str = "query_expansion"
    base_semantic_weight: float = 1.0
    base_keyword_weight: float = 1.0
    base_graph_weight: float = 1.0


class QueryRouter:
    """Build query plans from lexical signals and configured capabilities."""

    def __init__(self, config: QueryRouterConfig | None = None) -> None:
        self.config = config or QueryRouterConfig()
        self._validate_config()

    def route(
        self,
        query: str,
        *,
        limit: int,
        keyword_available: bool,
        vector_available: bool,
        graph_available: bool,
        graph_enabled: bool = True,
        rerank_available: bool = False,
    ) -> QueryPlan:
        if limit < 1:
            raise ValueError("limit must be positive")

        signals = analyze_query(query)
        query_type = classify_query(query)
        keyword_budget = self._budget(
            limit,
            self.config.keyword_candidate_budget,
            self.config.keyword_candidate_multiplier,
        )
        vector_budget = self._budget(
            limit,
            self.config.vector_candidate_budget,
            self.config.vector_candidate_multiplier,
        )
        keyword_enabled = keyword_available
        vector_enabled = vector_available
        graph_is_enabled = graph_available and graph_enabled
        skip_reasons: list[str] = []

        if not keyword_enabled:
            skip_reasons.append("keyword:unavailable")
        if not vector_enabled:
            skip_reasons.append("vector:unavailable")
        if not graph_available:
            skip_reasons.append("graph:unavailable")
        elif not graph_enabled:
            skip_reasons.append("graph:disabled")

        if signals.artifact:
            vector_candidates_keyword_bounded = True
            weights = get_adaptive_weights(
                QueryType.FACTUAL,
                self.config.base_semantic_weight,
                self.config.base_keyword_weight,
                self.config.base_graph_weight,
            )
            skip_reasons.append("vector:awaiting_keyword_confidence")
        elif query_type == QueryType.EXPLORATORY:
            vector_candidates_keyword_bounded = False
            weights = get_adaptive_weights(
                query_type,
                self.config.base_semantic_weight,
                self.config.base_keyword_weight,
                self.config.base_graph_weight,
            )
        else:
            vector_candidates_keyword_bounded = False
            weights = get_adaptive_weights(
                query_type,
                self.config.base_semantic_weight,
                self.config.base_keyword_weight,
                self.config.base_graph_weight,
            )

        expansion_strategy = (
            self.config.expansion_strategy
            if self.config.expansion_enabled
            and not signals.artifact
            and query_type == QueryType.EXPLORATORY
            else None
        )
        if self.config.expansion_enabled and expansion_strategy is None:
            skip_reasons.append("expansion:query_not_eligible")

        if not rerank_available or self.config.rerank_budget <= 0:
            rerank_budget = 0
            if self.config.rerank_budget > 0:
                skip_reasons.append("rerank:unavailable")
        else:
            rerank_budget = min(self.config.rerank_budget, max(limit, 1))

        return QueryPlan(
            query_type=query_type,
            signals=signals,
            keyword_enabled=keyword_enabled,
            vector_enabled=vector_enabled,
            graph_enabled=graph_is_enabled,
            keyword_candidate_budget=keyword_budget,
            vector_candidate_budget=vector_budget,
            fusion_weights=(
                ("vector", weights[0]),
                ("keyword", weights[1]),
                ("graph", weights[2]),
            ),
            vector_candidates_keyword_bounded=vector_candidates_keyword_bounded,
            graph_depth=self.config.graph_depth,
            graph_seed_budget=self.config.graph_seed_budget,
            rerank_budget=rerank_budget,
            expansion_strategy=expansion_strategy,
            diagnostic_skip_reasons=tuple(skip_reasons),
        )

    def _budget(
        self,
        limit: int,
        configured_budget: int | None,
        multiplier: int | None,
    ) -> int:
        return max(
            configured_budget
            if configured_budget is not None
            else limit * (multiplier or self.config.candidate_multiplier),
            self.config.minimum_candidate_limit,
        )

    def _validate_config(self) -> None:
        if self.config.candidate_multiplier < 1:
            raise ValueError("candidate_multiplier must be positive")
        if self.config.minimum_candidate_limit < 1:
            raise ValueError("minimum_candidate_limit must be positive")
        for name in ("keyword_candidate_multiplier", "vector_candidate_multiplier"):
            value = getattr(self.config, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} must be positive")
        for name in ("keyword_candidate_budget", "vector_candidate_budget"):
            value = getattr(self.config, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} must be positive")
        if self.config.graph_seed_budget < 1:
            raise ValueError("graph_seed_budget must be positive")
        if self.config.graph_depth < 1:
            raise ValueError("graph_depth must be positive")
        for name in (
            "base_semantic_weight",
            "base_keyword_weight",
            "base_graph_weight",
        ):
            if getattr(self.config, name) < 0:
                raise ValueError(f"{name} must not be negative")


def route_query(
    query: str,
    *,
    limit: int,
    keyword_available: bool = True,
    vector_available: bool = True,
    graph_available: bool = False,
    config: QueryRouterConfig | None = None,
) -> QueryPlan:
    """Convenience wrapper for callers that do not need a router instance."""
    return QueryRouter(config).route(
        query,
        limit=limit,
        keyword_available=keyword_available,
        vector_available=vector_available,
        graph_available=graph_available,
    )


__all__ = [
    "QueryPlan",
    "QueryRouter",
    "QueryRouterConfig",
    "route_query",
]
