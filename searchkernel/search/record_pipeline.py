"""Source-agnostic retrieval for hydrated records."""

from __future__ import annotations

import asyncio
import heapq
import inspect
import logging
import math
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from dataclasses import replace as dataclass_replace
from types import MappingProxyType
from typing import Any, Literal, Protocol, cast

from searchkernel.domain import (
    ChunkResult,
    GraphNeighbor,
    Record,
    RecordHit,
    RecordIdentity,
    SearchResultProvenance,
    Vector,
)
from searchkernel.domain.vector_filters import candidate_storage_keys
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
    DiagnosticCapability,
    FailureStage,
    RecordSearchDiagnostics,
    RecordSearchFailure,
    RecordSearchOutcome,
    RecordSearchResult,
)
from searchkernel.runtime.canonical_cache import (
    CandidateResultCache,
    HydrationCache,
    HydrationCacheKey,
)
from searchkernel.runtime.query_embedding_cache import (
    QueryEmbeddingCache,
)
from searchkernel.runtime.trace import QueryTrace
from searchkernel.search.adaptive_limit import resolve_adaptive_result_limit
from searchkernel.search.bounded_graph import (
    TypedGraphEdge,
    expand_bounded_typed_graph,
)
from searchkernel.search.fusion import (
    fuse_calibrated_scores,
    fuse_reciprocal_rank,
)
from searchkernel.search.lane_confidence import keyword_confidence, lane_confidence
from searchkernel.search.normalization import normalize_scores
from searchkernel.search.pipeline_candidate_acquisition import (
    CandidateAcquirer,
    _is_async_callable,
)
from searchkernel.search.pipeline_candidate_cache import (
    CandidateCacheKey,
    CandidateCachePolicy,
)
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
GraphTargetResolver = Callable[
    [str, "RecordSearchQueryContext"],
    Sequence[RecordHit] | Awaitable[Sequence[RecordHit]],
]


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
    """Read-only query state supplied to query-aware policy callbacks.

    The optional ``retrieval_mode`` filter defaults to ``"hybrid"``. An
    application may set it to ``"semantic"`` (or ``"semantic_only"``) to
    retain vector retrieval while disabling keyword and graph lanes.
    """

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
    query_candidate_filter: (
        Callable[[RecordSearchCandidate, RecordSearchQueryContext], bool] | None
    ) = None
    query_candidate_set_eligible: (
        Callable[[Sequence[RecordHit], RecordSearchQueryContext], bool] | None
    ) = None
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
    query_score_adjuster: (
        Callable[[RecordSearchCandidate, RecordSearchQueryContext], float] | None
    ) = None
    result_filter: Callable[[RecordSearchResult], bool] | None = None
    post_process: (
        Callable[[list[RecordSearchResult]], Sequence[RecordSearchResult]] | None
    ) = None
    graph_target_resolver: GraphTargetResolver | None = None
    query_expander: (
        Callable[[str], str | Sequence[str] | Awaitable[str | Sequence[str]]] | None
    ) = None
    parent_expander: ParentRecordExpander | ParentIdentityResolver | None = None


@dataclass(frozen=True, slots=True)
class RecordSearchCandidate:
    """An unhydrated ranked record candidate."""

    identity: RecordIdentity
    score: float
    provenance: SearchResultProvenance
    priority: int = 0

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

    def __deepcopy__(self, memo: dict[int, object]) -> RecordSearchCandidate:
        cached = memo.get(id(self))
        if isinstance(cached, RecordSearchCandidate):
            return cached
        clone = RecordSearchCandidate(
            identity=self.identity,
            score=self.score,
            provenance=self.provenance.clone(),
            priority=self.priority,
        )
        memo[id(self)] = clone
        return clone


@dataclass
class _SearchExecution:
    """Mutable working state threaded through one ``async_search`` call.

    Deliberately not frozen: the fields below are reassigned and mutated
    in place as the pipeline routes, fuses, re-fuses, and hydrates a single
    request. Do not make this immutable.
    """

    query: str = ""
    limit: int = 0
    filters: dict[str, object] = field(default_factory=dict)
    query_context: RecordSearchQueryContext | None = None
    plan: QueryPlan | None = None
    rankings: dict[str, list[RecordHit]] = field(default_factory=dict)
    fused_scores: dict[str, float] = field(default_factory=dict)
    candidates: list[RecordSearchCandidate] | None = None
    base_candidates: list[RecordSearchCandidate] = field(default_factory=list)
    hydrated: list[RecordSearchResult] = field(default_factory=list)
    failures: list[RecordSearchFailure] = field(default_factory=list)
    missing_record_ids: list[str] = field(default_factory=list)
    cache_diagnostics: list[str] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)
    candidate_counts: dict[str, int] = field(default_factory=dict)
    raw_pre_fusion_overlap: DiagnosticCapability | None = None
    trace: QueryTrace | None = None
    candidate_key: CandidateCacheKey | None = None
    semantic_only: bool = False

    @property
    def routed_plan(self) -> QueryPlan:
        """The plan for this search, once routing has produced one."""
        plan = self.plan
        if plan is None:
            raise RuntimeError("search was not routed before use")
        return plan


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
    fusion_mode: Literal["rrf", "calibrated"] = "rrf"
    base_semantic_weight: float = 1.0
    base_keyword_weight: float = 1.0
    base_graph_weight: float = 1.0
    keyword_saturation_k: float = 10.0
    graph_fusion: Literal["rrf", "max"] = "rrf"
    graph_depth: int = 1
    max_graph_seeds: int = 10
    graph_enabled: bool = True
    adaptive_graph_enabled: bool = True
    adaptive_graph_min_seed_score: float = 0.75
    adaptive_graph_min_seed_count: int = 1
    graph_only_penalty: float = 0.5
    max_neighbors_per_seed: int = 10
    max_graph_concurrency: int = 8
    max_hydration_concurrency: int = 8
    max_hydration_batch_size: int = 100
    artifact_confidence_threshold: float = 0.75
    rerank_budget: int = 0
    expansion_enabled: bool = False
    expansion_timeout_s: float = 0.25
    expansion_top_k: int = 3
    expansion_similarity_threshold: float = 0.5
    synonym_expansion_enabled: bool = False
    synonym_expansion_max_terms: int = 3
    capture_trace: bool = False
    adaptive_enabled: bool = False
    maximum_limit: int = 100
    score_ratio_floor: float = 0.5
    minimum_score: float = 0.0
    maximum_score_gap: float = 1.0
    max_chunk_matches: int = 3
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
                adaptive_graph_enabled=self._config.adaptive_graph_enabled,
                rerank_budget=self._config.rerank_budget,
                expansion_enabled=(
                    self._config.expansion_enabled
                    or self._config.synonym_expansion_enabled
                ),
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

    def _begin_search(
        self,
        query: str,
        limit: int,
        filters: dict[str, object] | None,
    ) -> _SearchExecution | None:
        """Validate the request and build its working state, or signal empty.

        Returns ``None`` when the query is blank, telling the caller to
        short-circuit with an empty outcome without routing or acquisition.
        """
        if limit < 1:
            raise ValueError("limit must be positive")
        if not query.strip():
            return None

        failures: list[RecordSearchFailure] = []
        missing_record_ids: list[str] = []
        cache_diagnostics: list[str] = []
        diagnostics: list[str] = []
        candidate_counts: dict[str, int] = {}
        raw_pre_fusion_overlap = DiagnosticCapability(
            state="unavailable",
            reason="raw pre-fusion overlap is not retained by the pipeline",
        )
        trace = (
            QueryTrace(query_text=query, include_query=False)
            if self._config.capture_trace
            else None
        )
        filters = dict(filters or {})
        filters.setdefault("statuses", ["active"])
        semantic_only = _semantic_only_requested(filters)
        query_context = RecordSearchQueryContext(
            query=query,
            filters=filters,
            limit=limit,
        )
        return _SearchExecution(
            query=query,
            limit=limit,
            filters=filters,
            query_context=query_context,
            failures=failures,
            missing_record_ids=missing_record_ids,
            cache_diagnostics=cache_diagnostics,
            diagnostics=diagnostics,
            candidate_counts=candidate_counts,
            raw_pre_fusion_overlap=raw_pre_fusion_overlap,
            trace=trace,
            semantic_only=semantic_only,
        )

    def _plan_query(self, execution: _SearchExecution) -> None:
        """Route the query and record the plan's diagnostics and trace.

        Advances ``execution`` in place; the routed plan drives every
        acquisition, fusion, and re-fusion decision downstream.
        """
        semantic_only = execution.semantic_only
        plan = self._router.route(
            execution.query,
            limit=execution.limit,
            keyword_available=(
                self._keyword_store is not None and not semantic_only
            ),
            vector_available=self._vector_store is not None,
            graph_available=self._graph_store is not None and not semantic_only,
            graph_enabled=self._config.graph_enabled and not semantic_only,
            rerank_available=self._reranker is not None,
        )
        execution.plan = plan
        execution.diagnostics.extend(_plan_diagnostics(plan))
        if execution.trace is not None:
            execution.trace.provenance = {
                "query_plan": {
                    "type": plan.query_type.name.lower(),
                    "signals": plan.signals.names,
                    "lanes": plan.enabled_lanes,
                    "budgets": plan.lane_budgets,
                    "skip_reasons": plan.diagnostic_skip_reasons,
                }
            }

    async def _load_cached_candidates(self, execution: _SearchExecution) -> None:
        """Look up cached candidates for the routed plan, or record a miss key.

        ``acquisition_limit`` only shapes the cache key, so it stays local
        rather than joining the execution state.
        """
        plan = execution.routed_plan
        acquisition_limit = max(
            plan.keyword_candidate_budget,
            plan.vector_candidate_budget,
        )
        candidate_key = self._candidate_cache_policy.key(
            execution.query,
            execution.filters,
            execution.limit,
            acquisition_limit,
            execution.cache_diagnostics,
        )
        execution.candidate_key = candidate_key
        candidates: list[RecordSearchCandidate] | None = None
        if candidate_key is not None and not execution.failures:
            candidates = self._candidate_cache_policy.get(
                candidate_key,
                execution.cache_diagnostics,
            )
            if candidates is None:
                candidates = await self._candidate_cache_policy.async_wait_for_miss(
                    candidate_key,
                    execution.cache_diagnostics,
                )
        execution.candidates = candidates

    async def _acquire_artifact_candidates(self, execution: _SearchExecution) -> bool:
        """Try the artifact fast path: keyword-only acquisition for identifier-like queries.

        Returns ``True`` when a confident, policy-eligible keyword result set
        disabled the vector lane, meaning the caller must not run the other
        acquisition branches for this request; ``False`` otherwise.
        """
        plan = execution.routed_plan
        keyword_result = await self._capture_optional_stage(
            "keyword",
            plan.keyword_enabled,
            lambda: self._candidate_acquirer.keyword(
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
            threshold=self._config.artifact_confidence_threshold,
            saturation_k=self._config.keyword_saturation_k,
        )
        candidate_set_eligible = (
            self._policy.query_candidate_set_eligible
            if artifact_confident
            else None
        )
        eligible = artifact_confident and (
            candidate_set_eligible is None
            or candidate_set_eligible(
                execution.rankings.get("keyword", ()),
                execution.query_context,
            )
        )
        if eligible:
            plan = dataclass_replace(
                plan,
                vector_enabled=False,
                diagnostic_skip_reasons=(
                    *plan.diagnostic_skip_reasons,
                    "vector:artifact_keyword_confident",
                ),
            )
            execution.plan = plan
            execution.diagnostics.append("vector:artifact_keyword_confident")
        elif artifact_confident:
            execution.diagnostics.append("vector:artifact_keyword_ineligible")
        return eligible

    async def _acquire_bounded_vector_candidates(
        self, execution: _SearchExecution
    ) -> None:
        """Acquire keyword hits and a vector lane constrained to their ids.

        Runs keyword acquisition and query embedding concurrently, then
        (if a policy narrows candidate ids or ranking order) runs the vector
        lane bounded to the keyword result. Mutates ``execution.rankings``
        and ``execution.failures`` in place.
        """
        plan = execution.routed_plan
        query = execution.query
        filters = execution.filters
        failures = execution.failures
        rankings = execution.rankings
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
                    context=execution.query_context,
                    plan=plan,
                ),
            )
            vector_value = self._consume_stage(vector_result, failures)
            if vector_value is not None:
                rankings["vector"] = cast(list[RecordHit], vector_value)

    async def _acquire_parallel_candidates(self, execution: _SearchExecution) -> None:
        """Acquire keyword and vector candidates concurrently.

        Bind the plan, query, filters, rankings, and query context once up
        front so the task lambdas close over these fixed locals rather than
        re-reading ``execution`` at call time. The vector task is handed the
        same ``rankings`` dict the keyword task may still be writing into
        concurrently; that pre-existing interleaving is preserved as-is.
        Mutates ``execution.rankings`` and ``execution.failures`` in place.
        """
        plan = execution.routed_plan
        query = execution.query
        filters = execution.filters
        query_context = execution.query_context
        failures = execution.failures
        rankings = execution.rankings
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

    def _fuse_candidates(self, execution: _SearchExecution) -> None:
        """Fuse acquired rankings into a scored, policy-filtered candidate set.

        ``candidate_counts`` is only reassigned (not mutated) when there are
        rankings to count, matching the original conditional; the call site
        re-aliases it from ``execution`` afterward so the rebind is visible.
        """
        rankings = execution.rankings
        failures = execution.failures
        plan = execution.routed_plan
        execution.raw_pre_fusion_overlap = _raw_pre_fusion_overlap(
            rankings, failures
        )
        fused_scores: dict[str, float] = {}
        if rankings:
            execution.candidate_counts = {
                strategy: len(ranking) for strategy, ranking in rankings.items()
            }
            fused_scores = self._fuse_rankings(rankings, plan)
        execution.fused_scores = fused_scores
        base_candidates = self._build_candidates(fused_scores, rankings)
        execution.base_candidates = self._apply_candidate_policy(
            base_candidates, execution.query_context
        )

    def _reroute_for_adaptive_graph(self, execution: _SearchExecution) -> None:
        """Re-route with adaptive-graph readiness once base candidates exist.

        The readiness check runs against ``execution.base_candidates`` as
        produced under the plan already in place; this does not recompute
        them under any replacement plan. The diagnostics filter mutates
        ``execution.diagnostics`` in place (slice-assignment) rather than
        rebinding it, so the list object stays the one aliased elsewhere.
        """
        plan = execution.routed_plan
        if not (
            self._config.adaptive_graph_enabled and not plan.signals.relationship
        ):
            return
        semantic_only = execution.semantic_only
        adaptive_plan = self._router.route(
            execution.query,
            limit=execution.limit,
            keyword_available=self._keyword_store is not None,
            vector_available=self._vector_store is not None,
            graph_available=self._graph_store is not None,
            graph_enabled=self._config.graph_enabled and not semantic_only,
            adaptive_graph_ready=self._adaptive_graph_ready(
                execution.base_candidates
            ),
            rerank_available=self._reranker is not None,
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
            execution.trace.provenance["query_plan"] = {
                "type": adaptive_plan.query_type.name.lower(),
                "signals": adaptive_plan.signals.names,
                "lanes": adaptive_plan.enabled_lanes,
                "budgets": adaptive_plan.lane_budgets,
                "skip_reasons": adaptive_plan.diagnostic_skip_reasons,
            }

    async def _expand_graph_stage(self, execution: _SearchExecution) -> None:
        """Expand base candidates with graph neighbors and re-fuse, or pass through.

        Sets ``execution.candidates`` on every path — the graph-enabled
        success path, the empty-expansion ``else``, and the exception
        handler — so the caller never sees the stale cache-lookup value.
        ``direct_keys`` for graph priority is built from
        ``execution.base_candidates`` (the pre-graph set), not the
        post-expansion candidates.
        """
        plan = execution.routed_plan
        base_candidates = execution.base_candidates
        if plan.graph_enabled and base_candidates:
            try:
                graph_seeds = await self._resolve_graph_targets(
                    base_candidates,
                    execution.query_context,
                    execution.filters,
                )
                graph_ranking = await self._expand_graph(
                    graph_seeds,
                    plan,
                    execution.filters,
                )
                if graph_ranking:
                    execution.candidate_counts["graph"] = len(graph_ranking)
                    execution.rankings["graph"] = graph_ranking
                    if self._config.graph_fusion == "max":
                        fused_scores = dict(execution.fused_scores)
                        for hit in graph_ranking:
                            fused_scores[hit.storage_key] = max(
                                fused_scores.get(hit.storage_key, 0.0),
                                hit.score,
                            )
                    else:
                        fused_scores = self._fuse_rankings(
                            execution.rankings, plan
                        )
                    execution.fused_scores = fused_scores
                    candidates = self._build_candidates(
                        fused_scores, execution.rankings
                    )
                    candidates = self._apply_candidate_policy(
                        candidates, execution.query_context
                    )
                    candidates = self._apply_graph_priority(
                        candidates,
                        direct_keys={
                            candidate.storage_key
                            for candidate in base_candidates
                        },
                        plan=plan,
                    )
                    execution.candidates = candidates
                else:
                    execution.candidates = base_candidates
            except Exception as error:  # noqa: BLE001 - degraded mode captures backend failures
                self._handle_error("graph", error, execution.failures)
                execution.candidates = base_candidates
        else:
            execution.candidates = base_candidates

    async def async_search(
        self,
        query: str,
        *,
        limit: int = 10,
        filters: dict[str, object] | None = None,
    ) -> RecordSearchOutcome:
        """Return deterministic hydrated results for ``query``.

        ``filters["retrieval_mode"]`` accepts ``"hybrid"`` (the default),
        ``"semantic"``, or ``"semantic_only"``. Semantic-only requests keep
        vector retrieval and disable keyword and graph retrieval.
        """
        execution = self._begin_search(query, limit, filters)
        if execution is None:
            return RecordSearchOutcome()
        filters = execution.filters
        query_context = execution.query_context
        failures = execution.failures
        missing_record_ids = execution.missing_record_ids
        cache_diagnostics = execution.cache_diagnostics
        diagnostics = execution.diagnostics
        candidate_counts = execution.candidate_counts
        raw_pre_fusion_overlap = execution.raw_pre_fusion_overlap
        trace = execution.trace

        self._plan_query(execution)
        plan = execution.plan
        await self._load_cached_candidates(execution)
        candidate_key = execution.candidate_key
        candidates = execution.candidates

        if candidates is None:
            rankings: dict[str, list[RecordHit]] = {}
            execution.rankings = rankings
            artifact_path_complete = False
            if plan.signals.artifact:
                artifact_path_complete = await self._acquire_artifact_candidates(
                    execution
                )
                plan = execution.plan

            if plan.vector_enabled and plan.signals.artifact is False and (
                self._policy.vector_candidate_ids is not None
            ):
                await self._acquire_bounded_vector_candidates(execution)
            elif not artifact_path_complete and (
                plan.vector_enabled or plan.keyword_enabled
            ):
                await self._acquire_parallel_candidates(execution)

            self._fuse_candidates(execution)
            raw_pre_fusion_overlap = execution.raw_pre_fusion_overlap
            fused_scores = execution.fused_scores
            candidate_counts = execution.candidate_counts
            base_candidates = execution.base_candidates

            self._reroute_for_adaptive_graph(execution)
            plan = execution.plan

            await self._expand_graph_stage(execution)
            candidates = execution.candidates
            fused_scores = execution.fused_scores

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
                        candidate_counts["expansion"] = len(expansion_ranking)
                        rankings["expansion"] = expansion_ranking
                        fused_scores = self._fuse_rankings(rankings, plan)
                        candidates = self._apply_candidate_policy(
                            self._build_candidates(fused_scores, rankings),
                            query_context,
                        )

            candidates = self._apply_score_adjustments(candidates, query_context)
            candidates = self._apply_exact_identifier_priority(candidates, query)
            candidates = self._sort_candidates(candidates)
            if plan.signals.relationship and plan.graph_enabled:
                candidates = self._apply_graph_priority(
                    candidates,
                    direct_keys={
                        candidate.storage_key for candidate in base_candidates
                    },
                    plan=plan,
                )
            candidates = await self._expand_parents(candidates, failures)
            if candidate_key is not None:
                self._candidate_cache_policy.set(
                    candidate_key,
                    candidates,
                    cache_diagnostics,
                )
            raw_pre_fusion_overlap = _raw_pre_fusion_overlap(rankings, failures)

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

        chunk_candidates = [
            candidate
            for candidate in candidates[hydration_offset:]
            if _is_chunk_candidate(candidate)
        ]
        if chunk_candidates:
            batch_hydration = await self._hydrate_candidates(
                chunk_candidates,
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
        exact_identifier_keys = _exact_identifier_result_keys(hydrated, query)
        hydrated = self._apply_exact_identifier_result_priority(hydrated, query)
        hydrated = await self._aggregate_chunk_results(
            hydrated,
            failures,
            missing_record_ids,
            limit=result_limit,
            priority_keys=exact_identifier_keys,
        )
        if self._policy.post_process is not None:
            hydrated = list(self._policy.post_process(hydrated))
        hydrated = hydrated[:limit]
        normalized_scores = normalize_scores([result.score for result in hydrated])
        hydrated = [
            dataclass_replace(result, normalized_score=normalized_score)
            for result, normalized_score in zip(hydrated, normalized_scores)
        ]

        if trace is not None:
            trace.provenance = {
                **(trace.provenance or {}),
                "diagnostics": tuple(diagnostics),
            }
            trace.close()
        stage_timings_ms: dict[str, float] = {}
        if trace is not None:
            if trace.total_duration_ms is not None:
                stage_timings_ms["search"] = trace.total_duration_ms
            stage_timings_ms.update(
                {
                    name: span.duration_ms
                    for name, span in trace.spans.items()
                    if span.duration_ms is not None
                }
            )

        return RecordSearchOutcome(
            results=tuple(hydrated),
            failures=tuple(failures),
            missing_record_ids=tuple(missing_record_ids),
            cache_diagnostics=tuple(cache_diagnostics),
            diagnostics=tuple(diagnostics),
            candidate_count=len(candidates),
            candidate_counts=candidate_counts,
            stage_timings_ms=stage_timings_ms,
            trace=trace,
            diagnostic_evidence=RecordSearchDiagnostics(
                enabled_lanes=plan.enabled_lanes,
                lane_budgets=plan.lane_budgets,
                skipped_lanes=plan.diagnostic_skip_decisions,
                failures=tuple(failures),
                missing_record_ids=tuple(missing_record_ids),
                stage_timings_ms=stage_timings_ms,
                result_provenance={
                    result.storage_key: result.provenance.strategies
                    for result in hydrated
                },
                final_duplicate_count=_duplicate_count(hydrated),
                raw_pre_fusion_overlap=raw_pre_fusion_overlap,
            ),
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
        rerankable: list[tuple[int, RecordSearchResult, str]] = []
        bypassed_positions: set[int] = set()
        for position, result in enumerate(selected):
            indexed_text = result.record.indexed_text or result.record.body
            text = f"{result.record.title}\n{indexed_text}".strip()
            max_chars = getattr(reranker, "max_input_chars", None)
            if isinstance(max_chars, int) and max_chars > 0:
                text = text[:max_chars]
            if text:
                rerankable.append((position, result, text))
            else:
                bypassed_positions.add(position)
        if len(rerankable) != len(selected):
            diagnostics.append("reranker_bypassed_empty_text")
        if not rerankable:
            return list(results)
        try:
            scores = await _call_async(
                reranker.rerank, query, [text for _, _, text in rerankable]
            )
            if len(scores) != len(rerankable):
                raise ValueError(
                    f"reranker returned {len(scores)} scores for "
                    f"{len(rerankable)} candidates"
                )
            reranked: list[tuple[int, RecordSearchResult]] = []
            for (position, result, _), score in zip(
                rerankable, scores, strict=False
            ):
                score = float(score)
                if not math.isfinite(score):
                    raise ValueError("reranker returned a non-finite score")
                reranked.append(
                    (
                        position,
                        RecordSearchResult(
                            record=result.record,
                            score=score,
                            provenance=result.provenance,
                            chunk_matches=result.chunk_matches,
                        ),
                    )
                )
        except Exception as error:  # noqa: BLE001 - reranking is optional
            self._handle_error("rerank", error, failures)
            diagnostics.append(f"rerank:fallback:{type(error).__name__}")
            return list(results)
        reranked.sort(key=lambda item: (-item[1].score, item[1].storage_key))
        ranked_results = iter(result for _, result in reranked)
        merged = [
            selected[position]
            if position in bypassed_positions
            else next(ranked_results)
            for position in range(len(selected))
        ]
        diagnostics.append(f"rerank:applied:{len(reranked)}")
        return [*merged, *results[len(selected) :]]

    async def _aggregate_chunk_results(
        self,
        results: Sequence[RecordSearchResult],
        failures: list[RecordSearchFailure],
        missing_record_ids: list[str],
        *,
        limit: int,
        priority_keys: frozenset[str] = frozenset(),
    ) -> list[RecordSearchResult]:
        grouped: dict[str, list[RecordSearchResult]] = {}
        ordinary: list[RecordSearchResult] = []
        for result in results:
            if not result.record.metadata.get("_searchkernel_chunk"):
                ordinary.append(result)
                continue
            parent_key = result.record.metadata.get("_chunk_parent_storage_key")
            if not isinstance(parent_key, str):
                ordinary.append(result)
                continue
            grouped.setdefault(parent_key, []).append(result)

        if not grouped:
            return list(results)

        by_parent = {result.storage_key: result for result in ordinary}
        aggregated = list(ordinary)
        for parent_key, matches in grouped.items():
            parent = by_parent.get(parent_key)
            if parent is None:
                parent_identity: RecordIdentity | None = None
                try:
                    parent_identity = RecordIdentity.from_storage_key(parent_key)
                    parent_record = await self._hydrate(
                        parent_identity
                    )
                except Exception as error:  # noqa: BLE001 - staged hydration failure
                    self._handle_error("hydration", error, failures)
                    if parent_identity is not None:
                        missing_record_ids.append(parent_identity.source_id)
                    continue
                if parent_record is None:
                    assert parent_identity is not None
                    missing_record_ids.append(parent_identity.source_id)
                    continue
                best = max(matches, key=lambda item: (-item.score, item.storage_key))
                parent = RecordSearchResult(
                    record=parent_record,
                    score=best.score,
                    provenance=best.provenance.clone(),
                )
                aggregated.append(parent)

            chunk_matches = [
                ChunkResult(
                    chunk_id=str(match.record.metadata["_chunk_id"]),
                    record_id=parent.record.source_id,
                    score=match.score,
                    content=match.record.body,
                    parent_chunk_id=cast(
                        str | None,
                        match.record.metadata.get("_chunk_metadata", {}).get(
                            "parent_chunk_id"
                        ),
                    ),
                    provenance=match.provenance,
                    metadata=dict(match.record.metadata.get("_chunk_metadata", {})),
                )
                for match in matches
            ]
            chunk_matches.sort(
                key=lambda item: (
                    -item.score,
                    int(item.metadata.get("start_pos", 0)),
                    item.chunk_id,
                )
            )
            combined = tuple(
                [*parent.chunk_matches, *chunk_matches][
                    : self._config.max_chunk_matches
                ]
            )
            replacement = RecordSearchResult(
                record=parent.record,
                score=max(parent.score, *(match.score for match in matches)),
                provenance=parent.provenance,
                chunk_matches=combined,
            )
            aggregated = [
                replacement if item.storage_key == parent.storage_key else item
                for item in aggregated
            ]

        aggregated.sort(
            key=lambda item: (
                item.storage_key not in priority_keys,
                -item.score,
                item.storage_key,
            )
        )
        return aggregated[:limit]

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
        if self._config.synonym_expansion_enabled:
            expander = self._policy.query_expander
            if expander is not None:
                try:
                    expanded = await asyncio.wait_for(
                        _call_async(expander, query),
                        timeout=self._config.expansion_timeout_s,
                    )
                except TimeoutError:
                    diagnostics.append("synonym_expansion:fallback:timeout")
                except Exception as error:  # noqa: BLE001 - expansion is optional
                    diagnostics.append(
                        f"synonym_expansion:fallback:{type(error).__name__}"
                    )
                else:
                    normalized = _normalize_query_expansion(
                        query,
                        expanded,
                        self._config.synonym_expansion_max_terms,
                    )
                    if normalized is not None:
                        diagnostics.append("synonym_expansion:applied")
                        return normalized
                    diagnostics.append("synonym_expansion:fallback:empty")
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

    async def _resolve_graph_targets(
        self,
        candidates: Sequence[RecordSearchCandidate],
        context: RecordSearchQueryContext,
        filters: Mapping[str, object],
    ) -> list[RecordSearchCandidate]:
        resolver = self._policy.graph_target_resolver
        if resolver is None or not context.query.strip():
            return list(candidates)
        resolved = await _call_async(resolver, context.query, context)
        if not isinstance(resolved, Sequence):
            raise TypeError("graph_target_resolver must return a sequence")
        normalized_hits = [
            RecordHit(
                await self._canonical_graph_target_identity(hit.identity),
                hit.score,
            )
            for hit in resolved
            if isinstance(hit, RecordHit)
        ]
        if len(normalized_hits) != len(resolved):
            raise TypeError("graph_target_resolver returned a non-canonical hit")
        targets: list[RecordSearchCandidate] = []
        seen: set[str] = set()
        for rank, hit in enumerate(
            sorted(
                normalized_hits,
                key=lambda item: (-item.score, item.storage_key),
            ),
            start=1,
        ):
            if hit.storage_key in seen or not _graph_identity_matches_filters(
                hit.identity, filters
            ):
                continue
            provenance = SearchResultProvenance(record_identity=hit.identity)
            provenance.add_strategy("graph_target", rank, hit.score)
            targets.append(
                RecordSearchCandidate(
                    identity=hit.identity,
                    score=hit.score,
                    provenance=provenance,
                )
            )
            seen.add(hit.storage_key)
        return targets or list(candidates)

    async def _canonical_graph_target_identity(
        self,
        identity: RecordIdentity,
    ) -> RecordIdentity:
        if not _looks_like_chunk_identity(identity.source_id):
            return identity
        parent_resolver = getattr(self._hydrator, "chunk_parent", None)
        if callable(parent_resolver):
            parent = await _call_async(parent_resolver, identity)
            if parent is not None:
                if not isinstance(parent, RecordIdentity):
                    raise TypeError("chunk_parent returned a non-canonical identity")
                return parent
        record = await self._hydrate(identity)
        if record is None:
            return identity
        return _parent_identity_from_record(record) or identity

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
        versioned_keys: dict[str, HydrationCacheKey] = {}
        cached: list[tuple[RecordSearchCandidate, Record | None]] = []
        misses: list[RecordSearchCandidate] = []
        if self._hydration_cache is not None and self._policy_version is not None:
            hydration_versions = await self._hydration_versions_for(
                [candidate.identity for candidate in candidates],
                diagnostics,
            )
            for candidate in candidates:
                try:
                    if candidate.storage_key in hydration_versions:
                        version = hydration_versions[candidate.storage_key]
                    else:
                        version = await self._hydration_version_for(
                            candidate.identity
                        )
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
                    versioned_keys[candidate.storage_key] = key
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
                    try:
                        leader, shared = await self._hydration_cache.async_wait_for_miss(
                            key
                        )
                    except Exception as error:  # noqa: BLE001 - cache is optional
                        leader, shared = True, None
                        diagnostics.append(
                            f"hydration_cache:bypass:{type(error).__name__}"
                        )
                    if leader:
                        misses.append(candidate)
                        diagnostics.append("hydration_cache:miss")
                    else:
                        cached.append((candidate, shared))
                        diagnostics.append("hydration_cache:coalesced")
        else:
            misses = list(candidates)
            if self._hydration_cache is not None:
                diagnostics.append("hydration_cache:bypass:missing_policy_version")

        if not misses:
            return cached
        hydrate_records = getattr(self._hydrator, "hydrate_records", None)
        if callable(hydrate_records):
            loaded: list[tuple[RecordSearchCandidate, Record | None]] = []
            for offset in range(0, len(misses), self._config.max_hydration_batch_size):
                hydration_batch = misses[
                    offset : offset + self._config.max_hydration_batch_size
                ]
                result = await _capture_stage(
                    "hydration",
                    lambda batch=hydration_batch: _call_async(
                        hydrate_records,
                        [candidate.identity for candidate in batch],
                    ),
                )
                if result[2] is not None:
                    for candidate in hydration_batch:
                        key = versioned_keys.get(candidate.storage_key)
                        if key is not None:
                            self._hydration_cache.fail(key, result[2])
                records = self._consume_stage(result, failures)
                if records is None:
                    continue
                records_by_key = cast(Mapping[str, Record | None], records)
                loaded.extend(
                    (candidate, records_by_key.get(candidate.storage_key))
                    for candidate in hydration_batch
                )
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
                key = versioned_keys.get(candidate.storage_key)
                if key is not None:
                    self._hydration_cache.fail(key, error)
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

    async def _hydration_versions_for(
        self,
        identities: Sequence[RecordIdentity],
        diagnostics: list[str],
    ) -> Mapping[str, object | None]:
        provider = self._hydration_version_provider
        if provider is None or self._hydration_version is not None:
            return {}
        batch_provider = getattr(provider, "hydration_versions", None)
        if not callable(batch_provider):
            return {}
        try:
            versions = await _call_async(batch_provider, identities)
            if not isinstance(versions, Mapping):
                raise TypeError("hydration_versions must return a mapping")
            return cast(Mapping[str, object | None], versions)
        except Exception as error:  # noqa: BLE001 - scalar fallback preserves compatibility
            diagnostics.append(
                f"hydration_cache:batch_version_fallback:{type(error).__name__}"
            )
            return {}

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
                self._hydration_cache.fail(key, error)
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

    def _fuse_rankings(
        self,
        rankings: Mapping[str, Sequence[RecordHit]],
        plan: QueryPlan,
    ) -> dict[str, float]:
        weights = (
            plan.fusion_weight_map if self._config.weighted_rrf_enabled else None
        )
        if self._config.fusion_mode == "calibrated":
            return fuse_calibrated_scores(
                {
                    strategy: [
                        (hit.storage_key, hit.score) for hit in ranking
                    ]
                    for strategy, ranking in rankings.items()
                },
                strategy_weights=weights,
            )
        return fuse_reciprocal_rank(
            {
                strategy: [hit.storage_key for hit in ranking]
                for strategy, ranking in rankings.items()
            },
            k=self._config.rrf_k,
            strategy_weights=weights,
        )

    def _apply_score_adjustments(
        self,
        candidates: Sequence[RecordSearchCandidate],
        context: RecordSearchQueryContext,
    ) -> list[RecordSearchCandidate]:
        if (
            self._policy.score_adjuster is None
            and self._policy.query_score_adjuster is None
        ):
            return list(candidates)
        adjusted: list[RecordSearchCandidate] = []
        for candidate in candidates:
            score = candidate.score
            if self._policy.score_adjuster is not None:
                score = self._policy.score_adjuster(
                    dataclass_replace(candidate, score=score)
                )
            if self._policy.query_score_adjuster is not None:
                score = self._policy.query_score_adjuster(
                    dataclass_replace(candidate, score=score),
                    context,
                )
            adjusted.append(dataclass_replace(candidate, score=score))
        return adjusted

    @staticmethod
    def _apply_exact_identifier_priority(
        candidates: Sequence[RecordSearchCandidate],
        query: str,
    ) -> list[RecordSearchCandidate]:
        normalized_query = query.strip().casefold()
        if not normalized_query:
            return list(candidates)
        exact = [
            candidate
            for candidate in candidates
            if normalized_query in _candidate_identifiers(candidate)
        ]
        if not exact:
            return list(candidates)
        exact_keys = {candidate.storage_key for candidate in exact}
        return [
            dataclass_replace(
                candidate,
                priority=1,
                provenance=_identifier_provenance(candidate.provenance, candidate.score),
            )
            if candidate.storage_key in exact_keys
            else candidate
            for candidate in candidates
        ]

    @staticmethod
    def _apply_exact_identifier_result_priority(
        results: Sequence[RecordSearchResult],
        query: str,
    ) -> list[RecordSearchResult]:
        exact_keys = _exact_identifier_result_keys(results, query)
        if not exact_keys:
            return list(results)
        adjusted = [
            dataclass_replace(
                result,
                provenance=_identifier_provenance(result.provenance, result.score),
            )
            if result.storage_key in exact_keys
            else result
            for result in results
        ]
        return sorted(
            adjusted,
            key=lambda item: (
                item.storage_key not in exact_keys,
                -item.score,
                item.storage_key,
            ),
        )

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
        self,
        candidates: Sequence[RecordSearchCandidate],
        context: RecordSearchQueryContext,
    ) -> list[RecordSearchCandidate]:
        candidate_filter = self._policy.candidate_filter
        query_filter = self._policy.query_candidate_filter
        return [
            candidate
            for candidate in candidates
            if candidate_filter is None or candidate_filter(candidate)
            if query_filter is None or query_filter(candidate, context)
        ]

    async def _expand_graph(
        self,
        candidates: Sequence[RecordSearchCandidate],
        plan: QueryPlan,
        filters: Mapping[str, object],
    ) -> list[RecordHit]:
        graph_store = self._graph_store
        if graph_store is None:
            return []
        graph_seeds = self._sort_candidates(candidates)[
            : plan.graph_seed_budget
        ]
        graph_seeds = [
            seed
            for seed in graph_seeds
            if _graph_identity_matches_filters(seed.identity, filters)
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
            filters,
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
        filters: Mapping[str, object],
    ) -> dict[str, Sequence[GraphNeighbor]]:
        if plan.signals.graph_direction == "both":
            outgoing_plan = dataclass_replace(
                plan,
                signals=dataclass_replace(plan.signals, graph_direction="outgoing"),
            )
            outgoing = await self._load_graph_neighbors(
                graph_store, graph_seeds, outgoing_plan, filters
            )
            incoming_neighbors: dict[str, Sequence[GraphNeighbor]] = {}
            if getattr(graph_store, "direction", None) != "both" and getattr(
                graph_store, "_direction", None
            ) != "both" and any(
                callable(getattr(graph_store, name, None))
                for name in ("incoming_neighbors", "incoming_neighbors_many")
            ):
                incoming_plan = dataclass_replace(
                    plan,
                    signals=dataclass_replace(
                        plan.signals, graph_direction="incoming"
                    ),
                )
                incoming_neighbors = await self._load_graph_neighbors(
                    graph_store, graph_seeds, incoming_plan, filters
                )
            merged: dict[str, dict[tuple[str, str], GraphNeighbor]] = {}
            for values in (outgoing, incoming_neighbors):
                for seed_key, neighbors in values.items():
                    by_identity = merged.setdefault(seed_key, {})
                    for neighbor in neighbors:
                        key = (neighbor.identity.storage_key, neighbor.edge_type)
                        previous = by_identity.get(key)
                        if previous is None or neighbor.weight > previous.weight:
                            by_identity[key] = neighbor
            return {
                seed_key: tuple(
                    sorted(
                        neighbors.values(),
                        key=lambda neighbor: (
                            -neighbor.weight,
                            neighbor.identity.storage_key,
                            neighbor.edge_type,
                        ),
                    )
                )
                for seed_key, neighbors in merged.items()
            }
        identities = [candidate.identity for candidate in graph_seeds]
        incoming = plan.signals.graph_direction == "incoming"
        neighbors_many = getattr(
            graph_store,
            "incoming_neighbors_many" if incoming else "neighbors_many",
            None,
        )
        if not callable(neighbors_many):
            neighbors_many = getattr(graph_store, "neighbors_many", None)
        neighbor_loader = getattr(
            graph_store,
            "incoming_neighbors" if incoming else "neighbors",
            graph_store.neighbors,
        )
        if not callable(neighbor_loader):
            neighbor_loader = graph_store.neighbors
        normalized: dict[str, Sequence[GraphNeighbor]] = {}
        if callable(neighbors_many):
            kwargs: dict[str, Any] = {"depth": plan.graph_depth}
            if _supports_keyword(neighbors_many, "max_neighbors"):
                kwargs["max_neighbors"] = self._config.max_neighbors_per_seed
            if _supports_keyword(neighbors_many, "filters"):
                kwargs["filters"] = filters
            result = await _call_async(
                neighbors_many,
                identities,
                **kwargs,
            )
            normalized = _normalize_graph_neighbor_map(
                cast(Mapping[object, Sequence[GraphNeighbor]], result),
                graph_seeds,
            )
            normalized = {
                key: tuple(
                    neighbor
                    for neighbor in values
                    if _graph_neighbor_matches_filters(neighbor, filters)
                )
                for key, values in normalized.items()
            }
            if len(normalized) == len(graph_seeds):
                return normalized
            graph_seeds = [
                seed
                for seed in graph_seeds
                if seed.storage_key not in normalized
            ]

        semaphore = asyncio.Semaphore(self._config.max_graph_concurrency)

        async def load(
            seed: RecordSearchCandidate,
        ) -> tuple[str, Sequence[GraphNeighbor]]:
            async with semaphore:
                kwargs: dict[str, Any] = {"depth": plan.graph_depth}
                if _supports_keyword(neighbor_loader, "max_neighbors"):
                    kwargs["max_neighbors"] = self._config.max_neighbors_per_seed
                if _supports_keyword(neighbor_loader, "filters"):
                    kwargs["filters"] = filters
                loaded_neighbors = cast(
                    Sequence[GraphNeighbor],
                    await _call_async(
                        neighbor_loader,
                        seed.identity,
                        **kwargs,
                    ),
                )
                return seed.storage_key, cast(
                    Sequence[GraphNeighbor],
                    tuple(
                        neighbor
                        for neighbor in loaded_neighbors
                        if _graph_neighbor_matches_filters(neighbor, filters)
                    ),
                )

        loaded = await _gather_tasks(
            [asyncio.create_task(load(seed)) for seed in graph_seeds]
        )
        return {
            **normalized,
            **dict(cast(list[tuple[str, Sequence[GraphNeighbor]]], loaded)),
        }

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
            RecordSearchFailure(
                stage,
                str(error),
                type(error).__name__,
                detail=str(error),
            )
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
        if not math.isfinite(self._config.keyword_saturation_k) or (
            self._config.keyword_saturation_k <= 0
        ):
            raise ValueError("keyword_saturation_k must be finite and positive")
        if self._config.fusion_mode not in {"rrf", "calibrated"}:
            raise ValueError("fusion_mode must be 'rrf' or 'calibrated'")
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
        if self._config.synonym_expansion_max_terms < 1:
            raise ValueError("synonym_expansion_max_terms must be positive")
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
        if self._config.adaptive_graph_min_seed_score < 0:
            raise ValueError("adaptive_graph_min_seed_score must not be negative")
        if self._config.adaptive_graph_min_seed_count < 1:
            raise ValueError("adaptive_graph_min_seed_count must be positive")
        if not 0 <= self._config.graph_only_penalty <= 1:
            raise ValueError("graph_only_penalty must be between zero and one")
        if self._config.max_hydration_concurrency < 1:
            raise ValueError("max_hydration_concurrency must be positive")
        if self._config.max_hydration_batch_size < 1:
            raise ValueError("max_hydration_batch_size must be positive")
        if self._config.max_chunk_matches < 1:
            raise ValueError("max_chunk_matches must be positive")
        if self._config.failure_mode not in {"strict", "lenient"}:
            raise ValueError("failure_mode must be 'strict' or 'lenient'")

    @staticmethod
    def _sort_candidates(
        candidates: Sequence[RecordSearchCandidate],
    ) -> list[RecordSearchCandidate]:
        return sorted(
            candidates,
            key=lambda item: (-item.priority, -item.score, item.storage_key),
        )

    def _adaptive_graph_ready(
        self, candidates: Sequence[RecordSearchCandidate]
    ) -> bool:
        strong_seed_count = sum(
            any(
                strategy in {"keyword", "vector"}
                and lane_confidence(
                    strategy,
                    contribution.raw_score,
                    saturation_k=self._config.keyword_saturation_k,
                )
                >= self._config.adaptive_graph_min_seed_score
                for strategy, contribution in candidate.provenance.strategy_details.items()
            )
            for candidate in self._sort_candidates(candidates)[
                : self._config.max_graph_seeds
            ]
        )
        return strong_seed_count >= self._config.adaptive_graph_min_seed_count

    def _apply_graph_priority(
        self,
        candidates: Sequence[RecordSearchCandidate],
        *,
        direct_keys: set[str],
        plan: QueryPlan,
    ) -> list[RecordSearchCandidate]:
        if plan.signals.relationship:
            direct: list[RecordSearchCandidate] = []
            graph: list[RecordSearchCandidate] = []
            remainder: list[RecordSearchCandidate] = []
            for candidate in candidates:
                if candidate.storage_key in direct_keys:
                    direct.append(candidate)
                elif "graph" in candidate.provenance.strategies:
                    graph.append(candidate)
                else:
                    remainder.append(candidate)
            return [*direct[:1], *graph, *direct[1:], *remainder]
        if not plan.adaptive_graph:
            return list(candidates)
        adjusted = [
            candidate
            if candidate.storage_key in direct_keys
            else dataclass_replace(
                candidate,
                score=candidate.score * self._config.graph_only_penalty,
            )
            for candidate in candidates
        ]
        return sorted(
            adjusted,
            key=lambda candidate: (
                candidate.storage_key not in direct_keys,
                -candidate.priority,
                -candidate.score,
                candidate.storage_key,
            ),
        )


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


def _is_chunk_candidate(candidate: RecordSearchCandidate) -> bool:
    return "#chunk:" in candidate.record_id


def _looks_like_chunk_identity(source_id: str) -> bool:
    return "#chunk:" in source_id or "_chunk_" in source_id


def _identifier_provenance(
    provenance: SearchResultProvenance,
    score: float,
) -> SearchResultProvenance:
    enriched = provenance.clone()
    enriched.add_strategy("exact_identifier", 1, score)
    return enriched


def _exact_identifier_result_keys(
    results: Sequence[RecordSearchResult],
    query: str,
) -> frozenset[str]:
    normalized_query = query.strip().casefold()
    return frozenset(
        result.storage_key
        for result in results
        if normalized_query in _record_identifiers(result.record)
    )


def _candidate_identifiers(candidate: RecordSearchCandidate) -> frozenset[str]:
    return _identity_identifiers(candidate.identity)


def _record_identifiers(record: Record) -> frozenset[str]:
    return _identity_identifiers(
        RecordIdentity(record.workspace_id, record.source_kind, record.source_id)
    )


def _identity_identifiers(identity: RecordIdentity) -> frozenset[str]:
    qualified = f"{identity.source_kind}:{identity.source_id}"
    if identity.workspace_id is None:
        return frozenset(
            value.casefold()
            for value in (identity.source_id, identity.storage_key, qualified)
        )
    scoped = f"{identity.workspace_id}:{qualified}"
    return frozenset(
        value.casefold()
        for value in (
            identity.source_id,
            identity.storage_key,
            qualified,
            scoped,
        )
    )


def _parent_identity_from_record(record: Record) -> RecordIdentity | None:
    for key in ("_chunk_parent_storage_key", "parent_storage_key"):
        value = record.metadata.get(key)
        if isinstance(value, str):
            try:
                return RecordIdentity.from_storage_key(value)
            except ValueError:
                continue
    for key in ("doc_id", "document_id", "parent_id"):
        value = record.metadata.get(key)
        if isinstance(value, str) and value and value != record.source_id:
            return RecordIdentity(
                record.workspace_id,
                record.source_kind,
                value,
            )
    return None


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


def _graph_identity_matches_filters(
    identity: RecordIdentity,
    filters: Mapping[str, object],
) -> bool:
    workspace_id = filters.get("workspace_id")
    if workspace_id is not None and identity.workspace_id != workspace_id:
        return False

    source_values = filters.get("source_kinds")
    if source_values is None:
        source_values = filters.get("source_kind")
    if source_values is None:
        source_values = filters.get("source_filter")
    if source_values is not None:
        if isinstance(source_values, str):
            allowed_sources = {source_values}
        elif isinstance(source_values, Sequence):
            allowed_sources = {str(value) for value in source_values}
        else:
            allowed_sources = {str(source_values)}
        if identity.source_kind not in allowed_sources:
            return False

    candidate_values = filters.get("candidate_ids")
    if candidate_values is None:
        candidate_values = filters.get("candidate_storage_keys")
    if candidate_values is None:
        return True
    return identity.storage_key in candidate_storage_keys(candidate_values)


def _graph_neighbor_matches_filters(
    neighbor: GraphNeighbor,
    filters: Mapping[str, object],
) -> bool:
    return _graph_identity_matches_filters(neighbor.identity, filters)


def _normalize_graph_neighbor_map(
    neighbors: Mapping[object, Sequence[GraphNeighbor]],
    seeds: Sequence[RecordSearchCandidate],
) -> dict[str, Sequence[GraphNeighbor]]:
    """Accept legacy source-id keys while preferring canonical seed keys."""
    by_storage_key = {
        seed.storage_key: seed.storage_key
        for seed in seeds
    }
    source_id_keys: dict[str, str | None] = {}
    for seed in seeds:
        current = source_id_keys.get(seed.record_id)
        source_id_keys[seed.record_id] = (
            seed.storage_key
            if current is None and seed.record_id not in source_id_keys
            else None
        )
    normalized: dict[str, Sequence[GraphNeighbor]] = {}
    for raw_key, values in neighbors.items():
        if isinstance(raw_key, RecordIdentity):
            key = raw_key.storage_key
        elif isinstance(raw_key, str):
            key = by_storage_key.get(raw_key) or source_id_keys.get(raw_key)
        else:
            key = None
        if key is not None:
            normalized[key] = values
    return normalized


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


def _duplicate_count(results: Sequence[RecordSearchResult]) -> int:
    storage_keys = [result.storage_key for result in results]
    return len(storage_keys) - len(set(storage_keys))


def _raw_pre_fusion_overlap(
    rankings: Mapping[str, Sequence[RecordHit]],
    failures: Sequence[RecordSearchFailure],
) -> DiagnosticCapability:
    if not rankings:
        return DiagnosticCapability(
            state="unavailable",
            reason="raw pre-fusion overlap is not retained by the pipeline",
        )
    if any(failure.stage in {"keyword", "vector", "graph"} for failure in failures):
        return DiagnosticCapability(
            state="unavailable",
            reason="raw pre-fusion overlap is not retained by the pipeline",
        )
    if len(rankings) == 1:
        return DiagnosticCapability(state="available", count=0)

    lane_keys = [
        {hit.storage_key for hit in ranking} for ranking in rankings.values()
    ]
    overlap = lane_keys[0].copy()
    for keys in lane_keys[1:]:
        overlap.intersection_update(keys)
    return DiagnosticCapability(state="available", count=len(overlap))


def _semantic_only_requested(filters: Mapping[str, object]) -> bool:
    mode = filters.get("retrieval_mode", "hybrid")
    if not isinstance(mode, str):
        raise TypeError(
            "retrieval_mode must be 'hybrid', 'semantic', or 'semantic_only'"
        )
    normalized_mode = mode.strip().lower()
    if normalized_mode == "hybrid":
        return False
    if normalized_mode in {"semantic", "semantic_only"}:
        return True
    raise ValueError(
        "retrieval_mode must be 'hybrid', 'semantic', or 'semantic_only'"
    )


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


def _normalize_query_expansion(
    query: str,
    expanded: str | Sequence[str],
    maximum_terms: int,
) -> str | None:
    if isinstance(expanded, str):
        terms = [" ".join(expanded.split()[:maximum_terms])]
    elif isinstance(expanded, Sequence):
        terms = [
            item.strip()
            for item in expanded
            if isinstance(item, str) and item.strip()
        ][:maximum_terms]
    else:
        return None
    additions = [term for term in terms if term.casefold() != query.casefold()]
    if not additions:
        return None
    return " ".join((query.strip(), *additions))


def _find_stage(
    results: Sequence[tuple[FailureStage, Any, Exception | None]],
    stage: FailureStage,
) -> tuple[FailureStage, Any, Exception | None] | None:
    for result_stage, value, error in results:
        if result_stage == stage:
            return result_stage, value, error
    return None
