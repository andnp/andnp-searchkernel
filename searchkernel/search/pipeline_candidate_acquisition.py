"""Source candidate acquisition for the record search pipeline."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import replace as dataclass_replace
from typing import TYPE_CHECKING, Any, cast

from searchkernel.domain import RecordHit, Vector
from searchkernel.ports import (
    AsyncKeywordStore,
    AsyncVectorStore,
    KeywordStore,
    VectorStore,
)
from searchkernel.ports.search_results import (
    DiagnosticCapability,
    FailureStage,
    RecordSearchFailure,
)
from searchkernel.search.lane_confidence import keyword_confidence
from searchkernel.search.query_plan import QueryPlan

if TYPE_CHECKING:
    from searchkernel.search.record_pipeline import (
        RecordSearchCandidate,
        RecordSearchPipeline,
        RecordSearchPolicy,
        RecordSearchQueryContext,
        _SearchExecution,
    )

QueryEmbedding = Callable[[str], Awaitable[tuple[Vector, str, int]]]


class CandidateAcquirer:
    """Acquire normalized keyword and vector candidate rankings."""

    def __init__(
        self,
        *,
        keyword_store: KeywordStore | AsyncKeywordStore | None,
        vector_store: VectorStore | AsyncVectorStore | None,
        policy: RecordSearchPolicy,
        query_embedding: QueryEmbedding,
    ) -> None:
        self._keyword_store = keyword_store
        self._vector_store = vector_store
        self._policy = policy
        self._query_embedding = query_embedding

    async def keyword(
        self,
        query: str,
        acquisition_limit: int,
        filters: dict[str, object],
    ) -> list[RecordHit]:
        store = self._keyword_store
        if store is None:
            return []
        search = cast(Callable[..., Any], store.search)
        return _normalize_hits(
            await _call_async(search, query, acquisition_limit, filters),
        )

    async def vector(
        self,
        embedding: tuple[Vector, str, int] | None,
        acquisition_limit: int,
        filters: dict[str, object],
        rankings: Mapping[str, Sequence[RecordHit]],
        *,
        context: RecordSearchQueryContext,
        query: str | None = None,
        plan: QueryPlan | None = None,
    ) -> list[RecordHit]:
        if embedding is None:
            if query is None:
                raise ValueError("query is required when embedding is absent")
            embedding = await self._query_embedding(query)
        vector, model_name, dim = embedding
        vector_filters = filters
        if self._policy.vector_candidate_ids is not None:
            candidate_ids = self._policy.vector_candidate_ids(
                list(rankings.get("keyword", ())),
                context,
            )
            if candidate_ids is not None:
                vector_filters = dict(filters)
                vector_filters["candidate_ids"] = list(candidate_ids)
        elif plan is not None and plan.vector_candidates_keyword_bounded:
            keyword_ranking = rankings.get("keyword", ())
            vector_filters = dict(filters)
            vector_filters["candidate_storage_keys"] = [
                hit.storage_key for hit in keyword_ranking
            ]
        vector_store = self._vector_store
        if vector_store is None:
            return []
        vector_ranking = _normalize_hits(
            await _search_vector_store(
                vector_store,
                vector,
                acquisition_limit,
                model_name=model_name,
                dim=dim,
                filters=vector_filters,
            ),
            sort=False,
        )
        if self._policy.vector_ranking_order is not None:
            vector_ranking = _normalize_hits(
                self._policy.vector_ranking_order(
                    vector_ranking,
                    context,
                ),
                sort=False,
            )
        return vector_ranking


class CandidateAcquisition:
    """Coordinate cache-aware candidate acquisition and fusion stages."""

    def __init__(
        self,
        pipeline: RecordSearchPipeline,
        raw_overlap: Callable[
            [Mapping[str, Sequence[RecordHit]], Sequence[RecordSearchFailure]],
            DiagnosticCapability,
        ],
    ) -> None:
        self._pipeline = pipeline
        self._raw_overlap = raw_overlap

    async def run(self, execution: _SearchExecution) -> None:
        await self._load_cached_candidates(execution)
        if execution.candidates is not None:
            return
        await self._acquire_candidates(execution)
        execution.base_candidates = self._fuse_candidates(execution)
        self._reroute_for_adaptive_graph(execution)
        await self._expand_graph_stage(execution)
        await self._expand_query_stage(execution)
        self._finalise_candidates(execution)
        await self._store_acquired_candidates(execution)

    async def _load_cached_candidates(self, execution: _SearchExecution) -> None:
        pipeline = self._pipeline
        plan = execution.routed_plan
        acquisition_limit = max(
            plan.keyword_candidate_budget,
            plan.vector_candidate_budget,
        )
        candidate_key = pipeline._candidate_cache_policy.key(
            execution.query,
            execution.filters,
            execution.limit,
            acquisition_limit,
            execution.cache_diagnostics,
        )
        execution.candidate_key = candidate_key
        candidates: list[RecordSearchCandidate] | None = None
        if candidate_key is not None and not execution.failures:
            candidates = pipeline._candidate_cache_policy.get(
                candidate_key,
                execution.cache_diagnostics,
            )
            if candidates is None:
                candidates = (
                    await pipeline._candidate_cache_policy.async_wait_for_miss(
                        candidate_key,
                        execution.cache_diagnostics,
                    )
                )
        execution.candidates = candidates

    async def _acquire_candidates(self, execution: _SearchExecution) -> None:
        pipeline = self._pipeline
        execution.rankings = {}
        artifact_path_complete = False
        plan = execution.routed_plan
        if plan.signals.artifact:
            artifact_path_complete = await self._acquire_artifact_candidates(
                execution
            )
            plan = execution.routed_plan
        if plan.vector_enabled and not plan.signals.artifact and (
            pipeline._policy.vector_candidate_ids is not None
        ):
            await self._acquire_bounded_vector_candidates(execution)
        elif not artifact_path_complete and (
            plan.vector_enabled or plan.keyword_enabled
        ):
            await self._acquire_parallel_candidates(execution)

    async def _acquire_artifact_candidates(
        self, execution: _SearchExecution
    ) -> bool:
        pipeline = self._pipeline
        plan = execution.routed_plan
        keyword_result = await pipeline._capture_optional_stage(
            "keyword",
            plan.keyword_enabled,
            lambda: pipeline._candidate_acquirer.keyword(
                execution.query,
                plan.keyword_candidate_budget,
                execution.filters,
            ),
            execution.failures,
        )
        if keyword_result is not None:
            execution.rankings["keyword"] = keyword_result
        artifact_confident = _artifact_results_are_confident(
            execution.rankings.get("keyword", ()),
            requested_limit=execution.limit,
            threshold=pipeline._config.artifact_confidence_threshold,
            saturation_k=pipeline._config.keyword_saturation_k,
        )
        candidate_set_eligible = (
            pipeline._policy.query_candidate_set_eligible
            if artifact_confident
            else None
        )
        eligible = artifact_confident and (
            candidate_set_eligible is None
            or candidate_set_eligible(
                execution.rankings.get("keyword", ()),
                execution.query_context_value,
            )
        )
        if eligible:
            execution.plan = dataclass_replace(
                plan,
                vector_enabled=False,
                diagnostic_skip_reasons=(
                    *plan.diagnostic_skip_reasons,
                    "vector:artifact_keyword_confident",
                ),
            )
            execution.diagnostics.append("vector:artifact_keyword_confident")
        elif artifact_confident:
            execution.diagnostics.append("vector:artifact_keyword_ineligible")
        return eligible

    async def _acquire_bounded_vector_candidates(
        self, execution: _SearchExecution
    ) -> None:
        pipeline = self._pipeline
        plan = execution.routed_plan
        stage_results = await _gather_tasks(
            [
                asyncio.create_task(
                    _capture_stage(
                        "keyword",
                        lambda: pipeline._candidate_acquirer.keyword(
                            execution.query,
                            plan.keyword_candidate_budget,
                            execution.filters,
                        ),
                    )
                )
                if plan.keyword_enabled
                else None,
                asyncio.create_task(
                    _capture_stage(
                        "vector",
                        lambda: pipeline._query_embedding(execution.query),
                    )
                )
                if pipeline._vector_store is not None
                else None,
            ]
        )
        keyword_result = pipeline._consume_stage(
            _find_stage(stage_results, "keyword"), execution.failures
        )
        if keyword_result is not None:
            execution.rankings["keyword"] = cast(list[RecordHit], keyword_result)
        embedding_result = pipeline._consume_stage(
            _find_stage(stage_results, "vector"), execution.failures
        )
        if embedding_result is not None:
            vector_result = await _capture_stage(
                "vector",
                lambda: pipeline._candidate_acquirer.vector(
                    cast(tuple[Vector, str, int], embedding_result),
                    plan.vector_candidate_budget,
                    execution.filters,
                    execution.rankings,
                    context=execution.query_context_value,
                    plan=plan,
                ),
            )
            vector_value = pipeline._consume_stage(
                vector_result, execution.failures
            )
            if vector_value is not None:
                execution.rankings["vector"] = cast(
                    list[RecordHit], vector_value
                )

    async def _acquire_parallel_candidates(self, execution: _SearchExecution) -> None:
        pipeline = self._pipeline
        plan = execution.routed_plan
        rankings = execution.rankings
        stage_results = await _gather_tasks(
            [
                asyncio.create_task(
                    _capture_stage(
                        "keyword",
                        lambda: pipeline._candidate_acquirer.keyword(
                            execution.query,
                            plan.keyword_candidate_budget,
                            execution.filters,
                        ),
                    )
                )
                if plan.keyword_enabled and "keyword" not in rankings
                else None,
                asyncio.create_task(
                    _capture_stage(
                        "vector",
                        lambda: pipeline._candidate_acquirer.vector(
                            None,
                            plan.vector_candidate_budget,
                            execution.filters,
                            rankings,
                            context=execution.query_context_value,
                            query=execution.query,
                            plan=plan,
                        ),
                    )
                )
                if plan.vector_enabled
                else None,
            ]
        )
        for stage, value, error in stage_results:
            if error is not None:
                pipeline._handle_error(stage, error, execution.failures)
            elif stage == "keyword":
                rankings["keyword"] = cast(list[RecordHit], value)
            else:
                rankings["vector"] = cast(list[RecordHit], value)

    def _fuse_candidates(
        self,
        execution: _SearchExecution,
        *,
        fused_scores: Mapping[str, float] | None = None,
    ) -> list[RecordSearchCandidate]:
        pipeline = self._pipeline
        execution.raw_pre_fusion_overlap = self._raw_overlap(
            execution.rankings, execution.failures
        )
        if execution.rankings:
            execution.candidate_counts = {
                strategy: len(ranking)
                for strategy, ranking in execution.rankings.items()
            }
        fused_scores = fused_scores or (
            pipeline._fuse_rankings(execution.rankings, execution.routed_plan)
            if execution.rankings
            else {}
        )
        execution.fused_scores = dict(fused_scores)
        return pipeline._apply_candidate_policy(
            pipeline._build_candidates(fused_scores, execution.rankings),
            execution.query_context_value,
        )

    def _reroute_for_adaptive_graph(self, execution: _SearchExecution) -> None:
        pipeline = self._pipeline
        plan = execution.routed_plan
        if not (
            pipeline._config.adaptive_graph_enabled
            and not plan.signals.relationship
        ):
            return
        adaptive_plan = pipeline._router.route(
            execution.query,
            limit=execution.limit,
            keyword_available=pipeline._keyword_store is not None,
            vector_available=pipeline._vector_store is not None,
            graph_available=pipeline._graph_store is not None,
            graph_enabled=pipeline._config.graph_enabled
            and not execution.semantic_only,
            adaptive_graph_ready=pipeline._adaptive_graph_ready(
                execution.base_candidates
            ),
            rerank_available=pipeline._reranker is not None,
        )
        if not adaptive_plan.graph_enabled:
            return
        execution.plan = adaptive_plan
        execution.diagnostics[:] = [
            diagnostic
            for diagnostic in execution.diagnostics
            if diagnostic != "query_plan:skip:graph:awaiting_seed_confidence"
        ]
        execution.diagnostics.append("query_plan:graph:adaptive")
        if execution.trace is not None:
            execution.trace.provenance = {
                **(execution.trace.provenance or {}),
                "query_plan": {
                    "type": adaptive_plan.query_type.name.lower(),
                    "signals": adaptive_plan.signals.names,
                    "lanes": adaptive_plan.enabled_lanes,
                    "budgets": adaptive_plan.lane_budgets,
                    "skip_reasons": adaptive_plan.diagnostic_skip_reasons,
                },
            }

    async def _expand_graph_stage(self, execution: _SearchExecution) -> None:
        pipeline = self._pipeline
        plan = execution.routed_plan
        base_candidates = execution.base_candidates
        if plan.graph_enabled and base_candidates:
            try:
                graph_seeds = await pipeline._resolve_graph_targets(
                    base_candidates,
                    execution.query_context_value,
                    execution.filters,
                )
                graph_ranking = await pipeline._expand_graph(
                    graph_seeds, plan, execution.filters
                )
                if graph_ranking:
                    execution.candidate_counts["graph"] = len(graph_ranking)
                    execution.rankings["graph"] = graph_ranking
                    if pipeline._config.graph_fusion == "max":
                        fused_scores = dict(execution.fused_scores)
                        for hit in graph_ranking:
                            fused_scores[hit.storage_key] = max(
                                fused_scores.get(hit.storage_key, 0.0), hit.score
                            )
                        candidates = self._fuse_candidates(
                            execution, fused_scores=fused_scores
                        )
                    else:
                        candidates = self._fuse_candidates(execution)
                    execution.candidates = pipeline._apply_graph_priority(
                        candidates,
                        direct_keys={
                            candidate.storage_key
                            for candidate in base_candidates
                        },
                        plan=plan,
                    )
                else:
                    execution.candidates = base_candidates
            except Exception as error:  # noqa: BLE001 - preserve degraded mode
                pipeline._handle_error("graph", error, execution.failures)
                execution.candidates = base_candidates
        else:
            execution.candidates = base_candidates

    async def _expand_query_stage(self, execution: _SearchExecution) -> None:
        pipeline = self._pipeline
        plan = execution.routed_plan
        candidates = execution.acquired_candidates
        if not (
            plan.expansion_strategy is not None
            and _needs_conditional_expansion(candidates, execution.limit)
        ):
            return
        expanded_query = await pipeline._expand_query(
            execution.query,
            plan.expansion_strategy,
            execution.diagnostics,
        )
        if expanded_query is None or expanded_query == execution.query:
            return
        expansion_ranking = await pipeline._capture_optional_stage(
            "keyword",
            plan.keyword_enabled,
            lambda: pipeline._candidate_acquirer.keyword(
                expanded_query,
                plan.keyword_candidate_budget,
                execution.filters,
            ),
            execution.failures,
        )
        if not expansion_ranking:
            return
        execution.candidate_counts["expansion"] = len(expansion_ranking)
        execution.rankings["expansion"] = expansion_ranking
        execution.candidates = self._fuse_candidates(execution)

    def _finalise_candidates(self, execution: _SearchExecution) -> None:
        pipeline = self._pipeline
        plan = execution.routed_plan
        candidates = pipeline._apply_score_adjustments(
            execution.acquired_candidates,
            execution.query_context_value,
        )
        candidates = pipeline._apply_exact_identifier_priority(
            candidates, execution.query
        )
        candidates = pipeline._sort_candidates(candidates)
        if plan.signals.relationship and plan.graph_enabled:
            candidates = pipeline._apply_graph_priority(
                candidates,
                direct_keys={
                    candidate.storage_key
                    for candidate in execution.base_candidates
                },
                plan=plan,
            )
        execution.candidates = candidates

    async def _store_acquired_candidates(self, execution: _SearchExecution) -> None:
        pipeline = self._pipeline
        execution.candidates = await pipeline._expand_parents(
            execution.acquired_candidates, execution.failures
        )
        if execution.candidate_key is not None:
            pipeline._candidate_cache_policy.set(
                execution.candidate_key,
                execution.candidates,
                execution.cache_diagnostics,
            )
        execution.raw_pre_fusion_overlap = self._raw_overlap(
            execution.rankings, execution.failures
        )


async def _capture_stage(
    stage: FailureStage,
    operation: Callable[[], Awaitable[Any]],
) -> tuple[FailureStage, Any, Exception | None]:
    try:
        return stage, await operation(), None
    except Exception as error:  # noqa: BLE001 - stage errors are caller-owned
        return stage, None, error


async def _gather_tasks(
    tasks: Sequence[asyncio.Task[Any] | None],
) -> list[Any]:
    pending = [task for task in tasks if task is not None]
    if not pending:
        return []
    try:
        return list(await asyncio.gather(*pending))
    except BaseException:
        for task in pending:
            if not task.done():
                task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        raise


def _find_stage(
    results: Sequence[tuple[FailureStage, Any, Exception | None]],
    stage: FailureStage,
) -> tuple[FailureStage, Any, Exception | None] | None:
    for result_stage, value, error in results:
        if result_stage == stage:
            return result_stage, value, error
    return None


def _artifact_results_are_confident(
    ranking: Sequence[RecordHit],
    *,
    requested_limit: int,
    threshold: float,
    saturation_k: float,
) -> bool:
    return (
        len(ranking) >= requested_limit
        and bool(ranking)
        and keyword_confidence(ranking[0].score, saturation_k=saturation_k)
        >= threshold
    )


def _needs_conditional_expansion(
    candidates: Sequence[RecordSearchCandidate],
    requested_limit: int,
) -> bool:
    if len(candidates) < requested_limit:
        return True
    return not candidates or candidates[0].score <= 0.0


def _normalize_hits(
    results: Sequence[RecordHit],
    *,
    sort: bool = True,
) -> list[RecordHit]:
    best: dict[str, RecordHit] = {}
    for hit in results:
        current = best.get(hit.storage_key)
        if current is None or hit.score > current.score:
            best[hit.storage_key] = hit
    normalized = list(best.values())
    if sort:
        normalized.sort(key=lambda hit: (-hit.score, hit.storage_key))
    return normalized


async def _search_vector_store(
    store: VectorStore | AsyncVectorStore,
    vector: Vector,
    k: int,
    *,
    model_name: str,
    dim: int,
    filters: dict[str, object] | None,
) -> Sequence[RecordHit]:
    async_search = getattr(store, "async_search", None)
    if callable(async_search):
        return await _call_async(
            cast(Callable[..., Any], async_search),
            vector,
            k,
            model_name=model_name,
            dim=dim,
            filters=filters,
        )
    return await _call_async(
        cast(Callable[..., Any], store.search),
        vector,
        k,
        model_name=model_name,
        dim=dim,
        filters=filters,
    )


async def _call_async[T](
    function: Callable[..., T | Awaitable[T]],
    *args: Any,
    **kwargs: Any,
) -> T:
    if _is_async_callable(function):
        value = function(*args, **kwargs)
    else:
        value = await asyncio.to_thread(function, *args, **kwargs)
    if inspect.isawaitable(value):
        return await value
    return value


def _is_async_callable(function: Callable[..., Any]) -> bool:
    """Return whether a callable invokes an async function directly."""
    return inspect.iscoroutinefunction(function) or inspect.iscoroutinefunction(
        type(function).__call__
    )
