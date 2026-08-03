"""Source-agnostic retrieval for hydrated records."""

from __future__ import annotations

import asyncio
import heapq
import inspect
import logging
import math
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from types import MappingProxyType
from typing import Any, Literal, Protocol, cast

from searchkernel.domain import (
    GraphNeighbor,
    Record,
    RecordHit,
    RecordIdentity,
    SearchResultProvenance,
    Vector,
)
from searchkernel.ports import (
    AsyncEmbeddingProvider,
    AsyncGraphStore,
    AsyncKeywordStore,
    AsyncVectorStore,
    EmbeddingProvider,
    GraphStore,
    KeywordStore,
    ParentRecordExpander,
    VectorStore,
)
from searchkernel.ports.rerank import Reranker
from searchkernel.ports.search_results import (
    FailureStage,
    RecordSearchFailure,
    RecordSearchOutcome,
    RecordSearchResult,
)
from searchkernel.runtime import (
    CandidateResultCache,
    HydrationCache,
    HydrationCacheKey,
    QueryEmbeddingCache,
)
from searchkernel.runtime.trace import QueryTrace
from searchkernel.search.adaptive_limit import resolve_adaptive_result_limit
from searchkernel.search.bounded_graph import (
    TypedGraphEdge,
    expand_bounded_typed_graph,
)
from searchkernel.search.fusion import fuse_reciprocal_rank
from searchkernel.search.pipeline_candidate_acquisition import (
    CandidateAcquirer,
    _is_async_callable,
)
from searchkernel.search.pipeline_candidate_cache import CandidateCachePolicy
from searchkernel.search.query_plan import (
    QueryPlan,
    QueryRouter,
    QueryRouterConfig,
)

logger = logging.getLogger(__name__)

RecordHydratorCallable = Callable[
    [RecordIdentity], Record | None | Awaitable[Record | None]
]
ParentIdentityResolver = Callable[
    [RecordIdentity],
    RecordIdentity | None | Awaitable[RecordIdentity | None],
]
QueryEmbeddingCallable = Callable[[str], Vector | Awaitable[Vector]]
class RecordHydrator(Protocol):
    """Hydrate a record without mutating source state."""

    def hydrate_record(
        self,
        record_id: RecordIdentity,
    ) -> Record | None | Awaitable[Record | None]: ...


class QueryEmbeddingProvider(Protocol):
    """Generate one query embedding for a vector search."""

    model_name: str
    dim: int

    def embed_query(self, query: str) -> Vector: ...


@dataclass(frozen=True, slots=True)
class RecordSearchQueryContext(Mapping[str, object]):
    """Read-only query state supplied to query-aware policy callbacks."""

    query: str
    filters: Mapping[str, object]
    limit: int

    def __post_init__(self) -> None:
        if self.limit < 1:
            raise ValueError("limit must be positive")
        object.__setattr__(self, "filters", MappingProxyType(dict(self.filters)))

    def __getitem__(self, key: str) -> object:
        return self.filters[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.filters)

    def __len__(self) -> int:
        return len(self.filters)


@dataclass(frozen=True, slots=True)
class RecordSearchPolicy:
    """Optional application-owned filtering, ranking, and post-processing."""

    candidate_filter: Callable[[RecordSearchCandidate], bool] | None = None
    vector_candidate_ids: (
        Callable[
            [Sequence[RecordHit], RecordSearchQueryContext],
            Sequence[str] | None,
        ]
        | None
    ) = None
    vector_ranking_order: (
        Callable[
            [Sequence[RecordHit], RecordSearchQueryContext],
            Sequence[RecordHit],
        ]
        | None
    ) = None
    score_adjuster: Callable[[RecordSearchCandidate], float] | None = None
    result_filter: Callable[[RecordSearchResult], bool] | None = None
    post_process: (
        Callable[[list[RecordSearchResult]], Sequence[RecordSearchResult]] | None
    ) = None
    parent_expander: ParentRecordExpander | ParentIdentityResolver | None = None


@dataclass(frozen=True, slots=True)
class RecordSearchCandidate:
    """An unhydrated ranked record candidate."""

    identity: RecordIdentity
    score: float
    provenance: SearchResultProvenance

    @property
    def record_id(self) -> str:
        return self.identity.source_id

    @property
    def workspace_id(self) -> str | None:
        return self.identity.workspace_id

    @property
    def source_kind(self) -> str:
        return self.identity.source_kind

    @property
    def storage_key(self) -> str:
        return self.identity.storage_key


class RecordSearchError(RuntimeError):
    """Raised when strict retrieval cannot complete a pipeline stage."""

    def __init__(self, stage: FailureStage, error: Exception) -> None:
        super().__init__(f"{stage} retrieval failed: {error}")
        self.stage = stage
        self.error = error


@dataclass(frozen=True, slots=True)
class RecordSearchConfig:
    """Deterministic limits and optional graph/adaptive retrieval settings.

    ``graph_enabled`` is an explicit opt-out; enabled graph work still
    requires a relationship signal in the query.
    """

    candidate_multiplier: int = 5
    minimum_candidate_limit: int = 1
    keyword_candidate_budget: int | None = None
    vector_candidate_budget: int | None = None
    keyword_candidate_multiplier: int | None = None
    vector_candidate_multiplier: int | None = None
    rrf_k: float = 60.0
    weighted_rrf_enabled: bool = False
    base_semantic_weight: float = 1.0
    base_keyword_weight: float = 1.0
    base_graph_weight: float = 1.0
    graph_fusion: Literal["rrf", "max"] = "rrf"
    graph_depth: int = 1
    max_graph_seeds: int = 10
    graph_enabled: bool = True
    max_neighbors_per_seed: int = 10
    max_graph_concurrency: int = 8
    max_hydration_concurrency: int = 8
    artifact_confidence_threshold: float = 0.75
    rerank_budget: int = 0
    expansion_enabled: bool = False
    expansion_timeout_s: float = 0.25
    expansion_top_k: int = 3
    expansion_similarity_threshold: float = 0.5
    capture_trace: bool = False
    adaptive_enabled: bool = False
    maximum_limit: int = 100
    score_ratio_floor: float = 0.5
    minimum_score: float = 0.0
    maximum_score_gap: float = 1.0
    failure_mode: Literal["strict", "lenient"] = "strict"


class RecordSearchPipeline:
    """Acquire, fuse, filter, graph-expand, and hydrate generic records.

    The pipeline is read-only. Domain lifecycle, authorization, supersession,
    and side effects remain injectable application policies.
    """

    def __init__(
        self,
        *,
        hydrator: RecordHydrator | RecordHydratorCallable,
        keyword_store: KeywordStore | AsyncKeywordStore | None = None,
        vector_store: VectorStore | AsyncVectorStore | None = None,
        graph_store: GraphStore | AsyncGraphStore | None = None,
        embedding_provider: (
            EmbeddingProvider
            | AsyncEmbeddingProvider
            | QueryEmbeddingProvider
            | Callable[[str], Vector | Awaitable[Vector]]
            | None
        ) = None,
        embedding_model_name: str | None = None,
        embedding_dim: int | None = None,
        reranker: Reranker | None = None,
        policy: RecordSearchPolicy | None = None,
        config: RecordSearchConfig | None = None,
        continue_on_error: bool | None = None,
        query_embedding_cache: QueryEmbeddingCache | None = None,
        candidate_cache: CandidateResultCache[
            tuple[RecordSearchCandidate, ...]
        ]
        | None = None,
        hydration_cache: HydrationCache[Record | None] | None = None,
        encoder_namespace: str | None = None,
        routing_fingerprint: str = "record-search-v1",
        policy_version: str | None = None,
        hydration_version: object | None = None,
        hydration_version_provider: (
            Callable[[RecordIdentity], object | Awaitable[object]] | None
        ) = None,
    ) -> None:
        if vector_store is not None and embedding_provider is None:
            raise ValueError("vector_store requires embedding_provider")
        self._hydrator = hydrator
        self._keyword_store = keyword_store
        self._vector_store = vector_store
        self._graph_store = graph_store
        self._embedding_provider = embedding_provider
        self._embedding_model_name = embedding_model_name
        self._embedding_dim = embedding_dim
        self._reranker = reranker
        self._policy = policy or RecordSearchPolicy()
        self._config = config or RecordSearchConfig()
        self._continue_on_error = (
            self._config.failure_mode == "lenient"
            if continue_on_error is None
            else continue_on_error
        )
        self._query_embedding_cache = query_embedding_cache or QueryEmbeddingCache()
        self._candidate_cache = candidate_cache or CandidateResultCache()
        self._hydration_cache = hydration_cache
        self._encoder_namespace = encoder_namespace
        self._routing_fingerprint = routing_fingerprint
        self._policy_version = policy_version
        self._hydration_version = hydration_version
        self._hydration_version_provider = hydration_version_provider
        self._candidate_cache_policy = CandidateCachePolicy(
            self._candidate_cache,
            config=self._config,
            policy=self._policy,
            keyword_store=self._keyword_store,
            vector_store=self._vector_store,
            graph_store=self._graph_store,
            embedding_provider=self._embedding_provider,
            embedding_model_name=self._embedding_model_name,
            embedding_dim=self._embedding_dim,
            encoder_namespace=self._encoder_namespace,
            routing_fingerprint=self._routing_fingerprint,
            policy_version=self._policy_version,
        )
        self._candidate_acquirer = CandidateAcquirer(
            keyword_store=self._keyword_store,
            vector_store=self._vector_store,
            policy=self._policy,
            query_embedding=self._query_embedding,
        )
        self._router = QueryRouter(
            QueryRouterConfig(
                candidate_multiplier=self._config.candidate_multiplier,
                minimum_candidate_limit=self._config.minimum_candidate_limit,
                keyword_candidate_budget=self._config.keyword_candidate_budget,
                vector_candidate_budget=self._config.vector_candidate_budget,
                keyword_candidate_multiplier=(
                    self._config.keyword_candidate_multiplier
                ),
                vector_candidate_multiplier=(
                    self._config.vector_candidate_multiplier
                ),
                graph_seed_budget=self._config.max_graph_seeds,
                graph_depth=self._config.graph_depth,
                rerank_budget=self._config.rerank_budget,
                expansion_enabled=self._config.expansion_enabled,
                base_semantic_weight=self._config.base_semantic_weight,
                base_keyword_weight=self._config.base_keyword_weight,
                base_graph_weight=self._config.base_graph_weight,
            )
        )
        self._validate_config()

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        filters: dict[str, object] | None = None,
    ) -> RecordSearchOutcome | Awaitable[RecordSearchOutcome]:
        """Search synchronously outside a loop or awaitably inside one."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.async_search(query, limit=limit, filters=filters)
            )
        return self.async_search(query, limit=limit, filters=filters)

    async def async_search(
        self,
        query: str,
        *,
        limit: int = 10,
        filters: dict[str, object] | None = None,
    ) -> RecordSearchOutcome:
        """Return deterministic hydrated results for ``query``."""
        if limit < 1:
            raise ValueError("limit must be positive")
        if not query.strip():
            return RecordSearchOutcome()

        failures: list[RecordSearchFailure] = []
        missing_record_ids: list[str] = []
        cache_diagnostics: list[str] = []
        diagnostics: list[str] = []
        trace = (
            QueryTrace(query_text=query, include_query=False)
            if self._config.capture_trace
            else None
        )
        filters = dict(filters or {})
        filters.setdefault("statuses", ["active"])
        query_context = RecordSearchQueryContext(
            query=query,
            filters=filters,
            limit=limit,
        )
        plan = self._router.route(
            query,
            limit=limit,
            keyword_available=self._keyword_store is not None,
            vector_available=self._vector_store is not None,
            graph_available=self._graph_store is not None,
            graph_enabled=self._config.graph_enabled,
            rerank_available=self._reranker is not None,
        )
        diagnostics.extend(_plan_diagnostics(plan))
        if trace is not None:
            trace.provenance = {
                "query_plan": {
                    "type": plan.query_type.name.lower(),
                    "signals": plan.signals.names,
                    "lanes": plan.enabled_lanes,
                    "budgets": plan.lane_budgets,
                    "skip_reasons": plan.diagnostic_skip_reasons,
                }
            }
        acquisition_limit = max(
            plan.keyword_candidate_budget,
            plan.vector_candidate_budget,
        )
        candidate_key = self._candidate_cache_policy.key(
            query,
            filters,
            limit,
            acquisition_limit,
            cache_diagnostics,
        )
        candidates: list[RecordSearchCandidate] | None = None
        if candidate_key is not None and not failures:
            candidates = self._candidate_cache_policy.get(
                candidate_key,
                cache_diagnostics,
            )

        if candidates is None:
            rankings: dict[str, list[RecordHit]] = {}
            if plan.signals.artifact:
                keyword_result = await self._capture_optional_stage(
                    "keyword",
                    plan.keyword_enabled,
                    lambda: self._candidate_acquirer.keyword(
                        query,
                        plan.keyword_candidate_budget,
                        filters,
                    ),
                    failures,
                )
                if keyword_result is not None:
                    rankings["keyword"] = keyword_result
                if _artifact_results_are_confident(
                    rankings.get("keyword", ()),
                    requested_limit=limit,
                    threshold=self._config.artifact_confidence_threshold,
                ):
                    plan = dataclass_replace(
                        plan,
                        vector_enabled=False,
                        diagnostic_skip_reasons=(
                            *plan.diagnostic_skip_reasons,
                            "vector:artifact_keyword_confident",
                        ),
                    )
                    diagnostics.append("vector:artifact_keyword_confident")

            if plan.vector_enabled and plan.signals.artifact is False and (
                self._policy.vector_candidate_ids is not None
            ):
                stage_results = await _gather_tasks(
                    [
                        asyncio.create_task(
                            _capture_stage(
                                "keyword",
                                lambda: self._candidate_acquirer.keyword(
                                    query, plan.keyword_candidate_budget, filters
                                ),
                            )
                        )
                        if plan.keyword_enabled
                        else None,
                        asyncio.create_task(
                            _capture_stage(
                                "vector",
                                lambda: self._query_embedding(query),
                            )
                        )
                        if self._vector_store is not None
                        else None,
                    ]
                )
                keyword_result = self._consume_stage(
                    _find_stage(stage_results, "keyword"),
                    failures,
                )
                if keyword_result is not None:
                    rankings["keyword"] = cast(list[RecordHit], keyword_result)
                embedding_result = self._consume_stage(
                    _find_stage(stage_results, "vector"),
                    failures,
                )
                if embedding_result is not None:
                    vector_result = await _capture_stage(
                        "vector",
                        lambda: self._candidate_acquirer.vector(
                            cast(tuple[Vector, str, int], embedding_result),
                            plan.vector_candidate_budget,
                            filters,
                            rankings,
                            context=query_context,
                            plan=plan,
                        ),
                    )
                    vector_value = self._consume_stage(vector_result, failures)
                    if vector_value is not None:
                        rankings["vector"] = cast(list[RecordHit], vector_value)
            elif plan.vector_enabled or plan.keyword_enabled:
                stage_results = await _gather_tasks(
                    [
                        asyncio.create_task(
                            _capture_stage(
                                "keyword",
                                lambda: self._candidate_acquirer.keyword(
                                    query, plan.keyword_candidate_budget, filters
                                ),
                            )
                        )
                        if plan.keyword_enabled and "keyword" not in rankings
                        else None,
                        asyncio.create_task(
                            _capture_stage(
                                "vector",
                                lambda: self._candidate_acquirer.vector(
                                    None,
                                    plan.vector_candidate_budget,
                                    filters,
                                    rankings,
                                    context=query_context,
                                    query=query,
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
                        self._handle_error(stage, error, failures)
                    elif stage == "keyword":
                        rankings["keyword"] = cast(list[RecordHit], value)
                    else:
                        rankings["vector"] = cast(list[RecordHit], value)

            fused_scores: dict[str, float] = {}
            if rankings:
                fused_scores = fuse_reciprocal_rank(
                    {
                        strategy: [hit.storage_key for hit in ranking]
                        for strategy, ranking in rankings.items()
                    },
                    k=self._config.rrf_k,
                    strategy_weights=(
                        plan.fusion_weight_map
                        if self._config.weighted_rrf_enabled
                        else None
                    ),
                )
            base_candidates = self._build_candidates(fused_scores, rankings)
            base_candidates = self._apply_candidate_policy(base_candidates)

            if plan.graph_enabled and base_candidates:
                try:
                    graph_ranking = await self._expand_graph(base_candidates, plan)
                    if graph_ranking:
                        rankings["graph"] = graph_ranking
                        if self._config.graph_fusion == "max":
                            fused_scores = dict(fused_scores)
                            for hit in graph_ranking:
                                fused_scores[hit.storage_key] = max(
                                    fused_scores.get(hit.storage_key, 0.0),
                                    hit.score,
                                )
                        else:
                            fused_scores = fuse_reciprocal_rank(
                                {
                                    strategy: [
                                        hit.storage_key for hit in ranking
                                    ]
                                    for strategy, ranking in rankings.items()
                                },
                                k=self._config.rrf_k,
                                strategy_weights=(
                                    plan.fusion_weight_map
                                    if self._config.weighted_rrf_enabled
                                    else None
                                ),
                            )
                        candidates = self._build_candidates(fused_scores, rankings)
                        candidates = self._apply_candidate_policy(candidates)
                    else:
                        candidates = base_candidates
                except Exception as error:  # noqa: BLE001 - degraded mode captures backend failures
                    self._handle_error("graph", error, failures)
                    candidates = base_candidates
            else:
                candidates = base_candidates

            if (
                plan.expansion_strategy is not None
                and _needs_conditional_expansion(candidates, limit)
            ):
                expanded_query = await self._expand_query(
                    query,
                    plan.expansion_strategy,
                    diagnostics,
                )
                if expanded_query is not None and expanded_query != query:
                    expansion_ranking = await self._capture_optional_stage(
                        "keyword",
                        plan.keyword_enabled,
                        lambda: self._candidate_acquirer.keyword(
                            expanded_query,
                            plan.keyword_candidate_budget,
                            filters,
                        ),
                        failures,
                    )
                    if expansion_ranking:
                        rankings["expansion"] = expansion_ranking
                        fused_scores = fuse_reciprocal_rank(
                            {
                                strategy: [
                                    hit.storage_key for hit in ranking
                                ]
                                for strategy, ranking in rankings.items()
                            },
                            k=self._config.rrf_k,
                            strategy_weights=(
                                plan.fusion_weight_map
                                if self._config.weighted_rrf_enabled
                                else None
                            ),
                        )
                        candidates = self._apply_candidate_policy(
                            self._build_candidates(fused_scores, rankings)
                        )

            candidates = self._apply_score_adjustments(candidates)
            candidates = self._sort_candidates(candidates)
            candidates = await self._expand_parents(candidates, failures)
            if candidate_key is not None:
                self._candidate_cache_policy.set(
                    candidate_key,
                    candidates,
                    cache_diagnostics,
                )

        assert candidates is not None
        result_limit = resolve_adaptive_result_limit(
            [candidate.score for candidate in candidates],
            requested_limit=limit,
            adaptive_enabled=self._config.adaptive_enabled,
            maximum_limit=self._config.maximum_limit,
            score_ratio_floor=self._config.score_ratio_floor,
            minimum_score=self._config.minimum_score,
            maximum_score_gap=self._config.maximum_score_gap,
        )

        hydrated: list[RecordSearchResult] = []
        hydration_offset = 0
        while hydration_offset < len(candidates) and len(hydrated) < result_limit:
            hydration_batch = candidates[
                hydration_offset : hydration_offset + result_limit
            ]
            hydration_offset += len(hydration_batch)
            batch_hydration = await self._hydrate_candidates(
                hydration_batch,
                failures,
                cache_diagnostics,
            )
            for candidate, record in batch_hydration:
                if record is None:
                    missing_record_ids.append(candidate.record_id)
                    continue
                result = RecordSearchResult(
                    record=record,
                    score=candidate.score,
                    provenance=candidate.provenance,
                )
                if self._policy.result_filter is None or self._policy.result_filter(result):
                    hydrated.append(result)

        hydrated = await self._rerank_results(
            query,
            hydrated,
            plan,
            failures,
            diagnostics,
        )
        if self._policy.post_process is not None:
            hydrated = list(self._policy.post_process(hydrated))

        if trace is not None:
            trace.provenance = {
                **(trace.provenance or {}),
                "diagnostics": tuple(diagnostics),
            }
            trace.close()

        return RecordSearchOutcome(
            results=tuple(hydrated),
            failures=tuple(failures),
            missing_record_ids=tuple(missing_record_ids),
            cache_diagnostics=tuple(cache_diagnostics),
            diagnostics=tuple(diagnostics),
            trace=trace,
        )

    async def _rerank_results(
        self,
        query: str,
        results: Sequence[RecordSearchResult],
        plan: QueryPlan,
        failures: list[RecordSearchFailure],
        diagnostics: list[str],
    ) -> list[RecordSearchResult]:
        reranker = self._reranker
        if reranker is None or plan.rerank_budget <= 0 or not results:
            return list(results)
        selected = list(results[: plan.rerank_budget])
        texts = [
            f"{result.record.title}\n{result.record.body[:1000]}".strip()
            for result in selected
        ]
        if not all(texts):
            diagnostics.append("rerank:fallback:empty_text")
            return list(results)
        try:
            scores = await _call_async(reranker.rerank, query, texts)
            if len(scores) != len(selected):
                raise ValueError(
                    f"reranker returned {len(scores)} scores for "
                    f"{len(selected)} candidates"
                )
            reranked = []
            for result, score in zip(selected, scores):
                score = float(score)
                if not math.isfinite(score):
                    raise ValueError("reranker returned a non-finite score")
                reranked.append(
                    RecordSearchResult(
                        record=result.record,
                        score=score,
                        provenance=result.provenance,
                    )
                )
        except Exception as error:  # noqa: BLE001 - reranking is optional
            self._handle_error("rerank", error, failures)
            diagnostics.append(f"rerank:fallback:{type(error).__name__}")
            return list(results)
        reranked.sort(key=lambda item: (-item.score, item.storage_key))
        diagnostics.append(f"rerank:applied:{len(reranked)}")
        return [*reranked, *results[len(selected) :]]

    async def _capture_optional_stage(
        self,
        stage: FailureStage,
        enabled: bool,
        operation: Callable[[], Awaitable[Any]],
        failures: list[RecordSearchFailure],
    ) -> Any | None:
        if not enabled:
            return None
        return self._consume_stage(
            await _capture_stage(stage, operation),
            failures,
        )

    async def _expand_query(
        self,
        query: str,
        strategy: str,
        diagnostics: list[str],
    ) -> str | None:
        if strategy != "query_expansion":
            diagnostics.append(f"expansion:fallback:unsupported:{strategy}")
            return None
        vector_store = self._vector_store
        if getattr(vector_store, "query_expansion_supported", True) is False:
            diagnostics.append("expansion:skip:unsupported")
            return None
        expand_query = getattr(vector_store, "expand_query", None)
        if not callable(expand_query):
            diagnostics.append("expansion:skip:unsupported")
            return None
        try:
            expanded = await asyncio.wait_for(
                _call_async(
                    expand_query,
                    query,
                    top_k=self._config.expansion_top_k,
                    similarity_threshold=self._config.expansion_similarity_threshold,
                ),
                timeout=self._config.expansion_timeout_s,
            )
        except TimeoutError:
            diagnostics.append("expansion:fallback:timeout")
            return None
        except Exception as error:  # noqa: BLE001 - expansion is optional
            diagnostics.append(
                f"expansion:fallback:{type(error).__name__}"
            )
            return None
        if not isinstance(expanded, str) or not expanded.strip():
            diagnostics.append("expansion:fallback:empty")
            return None
        diagnostics.append("expansion:applied")
        return expanded

    def _consume_stage(
        self,
        result: tuple[FailureStage, Any, Exception | None] | None,
        failures: list[RecordSearchFailure],
    ) -> Any | None:
        if result is None:
            return None
        stage, value, error = result
        if error is not None:
            self._handle_error(stage, error, failures)
            return None
        return value

    async def _hydrate_candidates(
        self,
        candidates: Sequence[RecordSearchCandidate],
        failures: list[RecordSearchFailure],
        diagnostics: list[str],
    ) -> list[tuple[RecordSearchCandidate, Record | None]]:
        if not candidates:
            return []
        versioned: list[
            tuple[RecordSearchCandidate, HydrationCacheKey]
        ] = []
        cached: list[tuple[RecordSearchCandidate, Record | None]] = []
        misses: list[RecordSearchCandidate] = []
        if self._hydration_cache is not None and self._policy_version is not None:
            for candidate in candidates:
                try:
                    version = await self._hydration_version_for(candidate.identity)
                except Exception as error:  # noqa: BLE001 - cache is optional
                    misses.append(candidate)
                    diagnostics.append(
                        f"hydration_cache:bypass:{type(error).__name__}"
                    )
                    continue
                if version is None:
                    misses.append(candidate)
                    continue
                try:
                    key = HydrationCacheKey.build(
                        candidate.identity,
                        record_version=version,
                        policy_version=self._policy_version,
                    )
                    hit, record = self._hydration_cache.lookup(key)
                    versioned.append((candidate, key))
                except Exception as error:  # noqa: BLE001 - cache is optional
                    misses.append(candidate)
                    diagnostics.append(
                        f"hydration_cache:bypass:{type(error).__name__}"
                    )
                    continue
                if hit:
                    cached.append((candidate, record))
                    diagnostics.append("hydration_cache:hit")
                else:
                    misses.append(candidate)
                    diagnostics.append("hydration_cache:miss")
        else:
            misses = list(candidates)
            if self._hydration_cache is not None:
                diagnostics.append("hydration_cache:bypass:missing_policy_version")

        if not misses:
            return cached
        hydrate_records = getattr(self._hydrator, "hydrate_records", None)
        if callable(hydrate_records):
            result = await _capture_stage(
                "hydration",
                lambda: _call_async(
                    hydrate_records,
                    [candidate.identity for candidate in misses],
                ),
            )
            records = self._consume_stage(result, failures)
            if records is None:
                return cached
            records_by_key = cast(Mapping[str, Record | None], records)
            loaded = [
                (candidate, records_by_key.get(candidate.storage_key))
                for candidate in misses
            ]
            self._store_hydration_cache(
                versioned,
                loaded,
                diagnostics,
            )
            hydrated_by_key = {
                candidate.storage_key: record
                for candidate, record in [*cached, *loaded]
            }
            return [
                (candidate, hydrated_by_key[candidate.storage_key])
                for candidate in candidates
            ]

        semaphore = asyncio.Semaphore(self._config.max_hydration_concurrency)

        async def hydrate(
            candidate: RecordSearchCandidate,
        ) -> tuple[RecordSearchCandidate, Record | None, Exception | None]:
            async with semaphore:
                try:
                    return candidate, await self._hydrate(candidate.identity), None
                except Exception as error:  # noqa: BLE001 - captured per candidate
                    return candidate, None, error

        loaded = await _gather_tasks(
            [asyncio.create_task(hydrate(candidate)) for candidate in misses]
        )
        hydrated: list[tuple[RecordSearchCandidate, Record | None]] = []
        for candidate, record, error in cast(
            list[tuple[RecordSearchCandidate, Record | None, Exception | None]],
            loaded,
        ):
            if error is not None:
                self._handle_error("hydration", error, failures)
                continue
            hydrated.append((candidate, record))
        self._store_hydration_cache(versioned, hydrated, diagnostics)
        hydrated_by_key = {
            candidate.storage_key: record
            for candidate, record in [*cached, *hydrated]
        }
        return [
            (candidate, hydrated_by_key[candidate.storage_key])
            for candidate in candidates
            if candidate.storage_key in hydrated_by_key
        ]

    async def _hydration_version_for(
        self,
        identity: RecordIdentity,
    ) -> object | None:
        if self._hydration_version is not None:
            return self._hydration_version
        provider = self._hydration_version_provider
        if provider is not None:
            return await _call_async(provider, identity)
        for name in ("record_epoch", "hydration_epoch"):
            value = getattr(self._hydrator, name, None)
            if callable(value):
                return await _call_async(value)
            if value is not None:
                return value
        return None

    def _store_hydration_cache(
        self,
        versioned: Sequence[tuple[RecordSearchCandidate, HydrationCacheKey]],
        loaded: Sequence[tuple[RecordSearchCandidate, Record | None]],
        diagnostics: list[str],
    ) -> None:
        if self._hydration_cache is None:
            return
        keys = {candidate.storage_key: key for candidate, key in versioned}
        for candidate, record in loaded:
            key = keys.get(candidate.storage_key)
            if key is None:
                continue
            try:
                self._hydration_cache.set(key, record)
            except Exception as error:  # noqa: BLE001 - cache is optional
                diagnostics.append(
                    f"hydration_cache:error:{type(error).__name__}"
                )

    def _build_candidates(
        self,
        fused_scores: Mapping[str, float],
        rankings: Mapping[str, Sequence[RecordHit]],
    ) -> list[RecordSearchCandidate]:
        candidates: list[RecordSearchCandidate] = []
        strategy_maps = {
            strategy: {
                hit.storage_key: (rank, hit.score, hit.identity)
                for rank, hit in enumerate(ranking, start=1)
            }
            for strategy, ranking in rankings.items()
        }
        identities = {
            storage_key: identity
            for strategy_map in strategy_maps.values()
            for storage_key, (_rank, _score, identity) in strategy_map.items()
        }
        for storage_key, score in fused_scores.items():
            identity = identities[storage_key]
            provenance = SearchResultProvenance(record_identity=identity)
            for strategy, strategy_map in strategy_maps.items():
                contribution = strategy_map.get(storage_key)
                if contribution is not None:
                    rank, raw_score, _identity = contribution
                    provenance.add_strategy(strategy, rank, raw_score)
            candidate = RecordSearchCandidate(
                identity, score, provenance
            )
            candidates.append(candidate)
        return candidates

    def _apply_score_adjustments(
        self, candidates: Sequence[RecordSearchCandidate]
    ) -> list[RecordSearchCandidate]:
        if self._policy.score_adjuster is None:
            return list(candidates)
        return [
            RecordSearchCandidate(
                identity=candidate.identity,
                score=self._policy.score_adjuster(candidate),
                provenance=candidate.provenance,
            )
            for candidate in candidates
        ]

    async def _expand_parents(
        self,
        candidates: Sequence[RecordSearchCandidate],
        failures: list[RecordSearchFailure],
    ) -> list[RecordSearchCandidate]:
        expander = self._policy.parent_expander
        if expander is None:
            return list(candidates)

        ordered = self._sort_candidates(candidates)
        parent_identities = getattr(expander, "parent_identities", None)
        if callable(parent_identities):
            try:
                resolved = await _call_async(
                    parent_identities,
                    [candidate.identity for candidate in ordered],
                )
                if not isinstance(resolved, Mapping):
                    raise TypeError("parent_expander must return a mapping")
            except Exception as error:  # noqa: BLE001 - policy failure is staged
                self._handle_error("parent_expansion", error, failures)
                return list(ordered)
            return self._apply_parent_identities(
                ordered,
                cast(Mapping[str, RecordIdentity | None], resolved),
            )

        expanded: list[RecordSearchCandidate] = []
        seen: set[str] = set()
        for candidate in ordered:
            try:
                parent_identity = await _resolve_parent_identity(
                    expander,
                    candidate.identity,
                )
            except Exception as error:  # noqa: BLE001 - policy failure is staged
                self._handle_error("parent_expansion", error, failures)
                parent_identity = None

            if parent_identity is None:
                if candidate.storage_key not in seen:
                    seen.add(candidate.storage_key)
                    expanded.append(candidate)
                continue

            if parent_identity.storage_key in seen:
                continue
            provenance = candidate.provenance.clone()
            provenance.record_identity = parent_identity
            provenance.parent_expanded_from = candidate.record_id
            provenance.parent_expanded_from_identity = candidate.identity
            seen.add(parent_identity.storage_key)
            expanded.append(
                RecordSearchCandidate(
                    identity=parent_identity,
                    score=candidate.score,
                    provenance=provenance,
                )
            )
        return expanded

    def _apply_parent_identities(
        self,
        candidates: Sequence[RecordSearchCandidate],
        parent_identities: Mapping[str, RecordIdentity | None],
    ) -> list[RecordSearchCandidate]:
        expanded: list[RecordSearchCandidate] = []
        seen: set[str] = set()
        for candidate in candidates:
            parent_identity = parent_identities.get(candidate.storage_key)
            if parent_identity is None:
                if candidate.storage_key not in seen:
                    seen.add(candidate.storage_key)
                    expanded.append(candidate)
                continue
            if not isinstance(parent_identity, RecordIdentity):
                raise TypeError("parent_expander returned a non-canonical identity")
            if parent_identity.storage_key in seen:
                continue
            provenance = candidate.provenance.clone()
            provenance.record_identity = parent_identity
            provenance.parent_expanded_from = candidate.record_id
            provenance.parent_expanded_from_identity = candidate.identity
            seen.add(parent_identity.storage_key)
            expanded.append(
                RecordSearchCandidate(
                    identity=parent_identity,
                    score=candidate.score,
                    provenance=provenance,
                )
            )
        return expanded

    def _apply_candidate_policy(
        self, candidates: Sequence[RecordSearchCandidate]
    ) -> list[RecordSearchCandidate]:
        if self._policy.candidate_filter is None:
            return list(candidates)
        return [
            candidate
            for candidate in candidates
            if self._policy.candidate_filter(candidate)
        ]

    async def _expand_graph(
        self,
        candidates: Sequence[RecordSearchCandidate],
        plan: QueryPlan,
    ) -> list[RecordHit]:
        graph_store = self._graph_store
        if graph_store is None:
            return []
        graph_seeds = self._sort_candidates(candidates)[
            : plan.graph_seed_budget
        ]
        seed_scores = {
            candidate.storage_key: candidate.score
            for candidate in graph_seeds
        }
        edges_by_seed: dict[str, list[TypedGraphEdge[str, tuple[str, float]]]] = {}
        discounts: dict[tuple[str, float], float] = {}
        seed_by_key = {
            candidate.storage_key: candidate
            for candidate in graph_seeds
        }
        neighbors_by_seed = await self._load_graph_neighbors(
            graph_store,
            graph_seeds,
            plan,
        )
        for seed_key in seed_scores:
            raw_neighbors = neighbors_by_seed.get(seed_key, ())
            top_neighbors = heapq.nsmallest(
                self._config.max_neighbors_per_seed,
                (
                    _normalize_graph_neighbor(neighbor)
                    for neighbor in raw_neighbors
                ),
                key=lambda item: (-item[2], item[0], item[1]),
            )
            edges: list[TypedGraphEdge[str, tuple[str, float]]] = []
            for target_id, edge_type, weight in top_neighbors:
                edge_key = (edge_type, weight)
                edges.append(TypedGraphEdge(target_id, edge_key))
                discounts[edge_key] = weight
            edges_by_seed[seed_key] = edges

        expanded = expand_bounded_typed_graph(
            seed_scores,
            lambda seed_id: edges_by_seed[seed_id],
            discounts,
            max_seed_count=plan.graph_seed_budget,
            max_neighbors_per_seed=self._config.max_neighbors_per_seed,
        )
        return sorted(
            (
                _graph_hit(record_id, expansion, seed_by_key)
                for record_id, expansion in expanded.items()
            ),
            key=lambda item: (-item.score, item.storage_key),
        )

    async def _load_graph_neighbors(
        self,
        graph_store: GraphStore | AsyncGraphStore,
        graph_seeds: Sequence[RecordSearchCandidate],
        plan: QueryPlan,
    ) -> dict[str, Sequence[GraphNeighbor]]:
        identities = [candidate.identity for candidate in graph_seeds]
        neighbors_many = getattr(graph_store, "neighbors_many", None)
        if callable(neighbors_many):
            kwargs: dict[str, Any] = {"depth": plan.graph_depth}
            if _supports_keyword(neighbors_many, "max_neighbors"):
                kwargs["max_neighbors"] = self._config.max_neighbors_per_seed
            result = await _call_async(
                neighbors_many,
                identities,
                **kwargs,
            )
            return dict(
                cast(
                    Mapping[
                        str,
                        Sequence[GraphNeighbor],
                    ],
                    result,
                )
            )

        semaphore = asyncio.Semaphore(self._config.max_graph_concurrency)

        async def load(
            seed: RecordSearchCandidate,
        ) -> tuple[str, Sequence[GraphNeighbor]]:
            async with semaphore:
                kwargs: dict[str, Any] = {"depth": plan.graph_depth}
                if _supports_keyword(graph_store.neighbors, "max_neighbors"):
                    kwargs["max_neighbors"] = self._config.max_neighbors_per_seed
                neighbors = await _call_async(
                    graph_store.neighbors,
                    seed.identity,
                    **kwargs,
                )
                return seed.storage_key, cast(
                    Sequence[GraphNeighbor],
                    neighbors,
                )

        loaded = await _gather_tasks(
            [asyncio.create_task(load(seed)) for seed in graph_seeds]
        )
        return dict(cast(list[tuple[str, Sequence[GraphNeighbor]]], loaded))

    async def _query_embedding(self, query: str) -> tuple[Vector, str, int]:
        provider = self._embedding_provider
        if provider is None:
            raise ValueError("embedding_provider is required for vector search")

        model_name = self._embedding_model_name or getattr(
            provider, "model_name", None
        )
        dim = self._embedding_dim or getattr(provider, "dim", None)
        if model_name is None or dim is None:
            raise ValueError(
                "vector search requires embedding model name and dimension"
            )
        async def compute() -> Vector:
            if hasattr(provider, "embed_query"):
                return await _call_async(
                    cast(QueryEmbeddingProvider, provider).embed_query,
                    query,
                )
            embed_method = getattr(provider, "embed", None)
            if callable(embed_method):
                embeddings = await _call_async(
                    cast(Callable[[list[str]], list[Vector]], embed_method),
                    [query],
                )
                if len(embeddings) != 1:
                    raise ValueError(
                        "embedding provider must return one query vector"
                    )
                return embeddings[0]
            return await _call_async(
                cast(QueryEmbeddingCallable, provider),
                query,
            )

        try:
            vector = await self._query_embedding_cache.async_get_or_compute(
                encoder_namespace=self._candidate_cache_policy.encoder_namespace(),
                query=query,
                compute=compute,
            )
        except Exception:
            logger.debug("query embedding cache bypassed", exc_info=True)
            vector = await compute()
        if len(vector) != dim:
            raise ValueError(
                f"query embedding has dimension {len(vector)}, expected {dim}"
            )
        return vector, model_name, dim

    async def _hydrate(self, identity: RecordIdentity) -> Record | None:
        if hasattr(self._hydrator, "hydrate_record"):
            return await _call_async(
                cast(RecordHydrator, self._hydrator).hydrate_record,
                identity,
            )
        hydrator = getattr(self._hydrator, "hydrate_record", self._hydrator)
        return await _call_async(cast(RecordHydratorCallable, hydrator), identity)

    def _handle_error(
        self,
        stage: FailureStage,
        error: Exception,
        failures: list[RecordSearchFailure],
    ) -> None:
        if not self._continue_on_error:
            raise RecordSearchError(stage, error) from error
        failures.append(
            RecordSearchFailure(stage, str(error), type(error).__name__)
        )

    def _validate_config(self) -> None:
        if self._config.candidate_multiplier < 1:
            raise ValueError("candidate_multiplier must be positive")
        if self._config.minimum_candidate_limit < 1:
            raise ValueError("minimum_candidate_limit must be positive")
        for name in (
            "keyword_candidate_multiplier",
            "vector_candidate_multiplier",
        ):
            value = getattr(self._config, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} must be positive")
        for name in ("keyword_candidate_budget", "vector_candidate_budget"):
            value = getattr(self._config, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} must be positive")
        if self._config.rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
        for name in (
            "base_semantic_weight",
            "base_keyword_weight",
            "base_graph_weight",
        ):
            if getattr(self._config, name) < 0:
                raise ValueError(f"{name} must not be negative")
        if self._config.artifact_confidence_threshold < 0:
            raise ValueError("artifact_confidence_threshold must not be negative")
        if self._config.rerank_budget < 0:
            raise ValueError("rerank_budget must not be negative")
        if self._config.expansion_timeout_s <= 0:
            raise ValueError("expansion_timeout_s must be positive")
        if self._config.expansion_top_k < 1:
            raise ValueError("expansion_top_k must be positive")
        if self._config.expansion_similarity_threshold < 0:
            raise ValueError(
                "expansion_similarity_threshold must not be negative"
            )
        if self._config.graph_fusion not in {"rrf", "max"}:
            raise ValueError("graph_fusion must be 'rrf' or 'max'")
        if self._config.graph_depth < 1:
            raise ValueError("graph_depth must be positive")
        if self._config.max_graph_seeds < 1:
            raise ValueError("max_graph_seeds must be positive")
        if self._config.max_neighbors_per_seed < 1:
            raise ValueError("max_neighbors_per_seed must be positive")
        if self._config.max_graph_concurrency < 1:
            raise ValueError("max_graph_concurrency must be positive")
        if self._config.max_hydration_concurrency < 1:
            raise ValueError("max_hydration_concurrency must be positive")
        if self._config.failure_mode not in {"strict", "lenient"}:
            raise ValueError("failure_mode must be 'strict' or 'lenient'")

    @staticmethod
    def _sort_candidates(
        candidates: Sequence[RecordSearchCandidate],
    ) -> list[RecordSearchCandidate]:
        return sorted(candidates, key=lambda item: (-item.score, item.storage_key))


async def _resolve_parent_identity(
    expander: ParentRecordExpander | ParentIdentityResolver,
    identity: RecordIdentity,
) -> RecordIdentity | None:
    resolver = getattr(expander, "parent_identity", None)
    if callable(resolver):
        parent_identity = await _call_async(resolver, identity)
    elif callable(expander):
        parent_identity = await _call_async(expander, identity)
    else:
        raise TypeError("parent_expander must resolve canonical record identities")
    if parent_identity is not None and not isinstance(
        parent_identity, RecordIdentity
    ):
        raise TypeError("parent_expander returned a non-canonical identity")
    return parent_identity


def _graph_hit(
    record_id: str,
    expansion: Any,
    seed_by_key: Mapping[str, RecordSearchCandidate],
) -> RecordHit:
    seed = seed_by_key[expansion.provenance.seed_id]
    if record_id == seed.storage_key:
        identity = seed.identity
    else:
        identity = RecordIdentity.from_storage_key(record_id)
    return RecordHit(
        identity,
        expansion.contribution,
    )


def _normalize_graph_neighbor(
    neighbor: GraphNeighbor,
) -> tuple[str, str, float]:
    return neighbor.identity.storage_key, neighbor.edge_type, neighbor.weight


def _supports_keyword(function: Callable[..., Any], name: str) -> bool:
    try:
        parameters = inspect.signature(function).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.name == name or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


async def _call_async[T](
    function: Callable[..., T | Awaitable[T]],
    *args: Any,
    **kwargs: Any,
) -> T:
    """Run blocking adapter calls away from the async event loop."""
    if _is_async_callable(function):
        value = function(*args, **kwargs)
    else:
        value = await asyncio.to_thread(function, *args, **kwargs)
    if inspect.isawaitable(value):
        return await value
    return value


async def _capture_stage(
    stage: FailureStage,
    operation: Callable[[], Awaitable[Any]],
) -> tuple[FailureStage, Any, Exception | None]:
    try:
        return stage, await operation(), None
    except Exception as error:  # noqa: BLE001 - stage errors are handled by the caller
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


def _plan_diagnostics(plan: QueryPlan) -> list[str]:
    diagnostics = [
        f"query_plan:type:{plan.query_type.name.lower()}",
        f"query_plan:signals:{','.join(plan.signals.names) or 'none'}",
        f"query_plan:lanes:{','.join(plan.enabled_lanes) or 'none'}",
        (
            "query_plan:budgets:"
            f"keyword={plan.keyword_candidate_budget},"
            f"vector={plan.vector_candidate_budget},"
            f"graph_seeds={plan.graph_seed_budget},"
            f"rerank={plan.rerank_budget}"
        ),
    ]
    diagnostics.extend(
        f"query_plan:skip:{reason}"
        for reason in plan.diagnostic_skip_reasons
    )
    return diagnostics


def _artifact_results_are_confident(
    ranking: Sequence[RecordHit],
    *,
    requested_limit: int,
    threshold: float,
) -> bool:
    return (
        len(ranking) >= requested_limit
        and bool(ranking)
        and ranking[0].score >= threshold
    )


def _needs_conditional_expansion(
    candidates: Sequence[RecordSearchCandidate],
    requested_limit: int,
) -> bool:
    if len(candidates) < requested_limit:
        return True
    return not candidates or candidates[0].score <= 0.0


def _find_stage(
    results: Sequence[tuple[FailureStage, Any, Exception | None]],
    stage: FailureStage,
) -> tuple[FailureStage, Any, Exception | None] | None:
    for result_stage, value, error in results:
        if result_stage == stage:
            return result_stage, value, error
    return None
