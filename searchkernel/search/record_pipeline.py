"""Source-agnostic retrieval for hydrated records."""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast

from searchkernel.domain import (
    GraphNeighbor,
    Record,
    RecordHit,
    RecordIdentity,
    SearchResult,
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
    VectorStore,
)
from searchkernel.runtime import (
    CandidateCacheKey,
    CandidateResultCache,
    HydrationCache,
    HydrationCacheKey,
    QueryEmbeddingCache,
    SearchEpochs,
    UnstableCacheKey,
    fingerprint,
)
from searchkernel.search.adaptive_limit import resolve_adaptive_result_limit
from searchkernel.search.bounded_graph import (
    TypedGraphEdge,
    expand_bounded_typed_graph,
)
from searchkernel.search.fusion import fuse_reciprocal_rank

logger = logging.getLogger(__name__)

RecordHydratorCallable = Callable[
    [RecordIdentity | str],
    Record | None | Awaitable[Record | None],
]
QueryEmbeddingCallable = Callable[[str], Vector | Awaitable[Vector]]
FailureStage = Literal["keyword", "vector", "graph", "hydration"]


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
class RecordSearchPolicy:
    """Optional application-owned filtering, ranking, and post-processing."""

    candidate_filter: Callable[[RecordSearchCandidate], bool] | None = None
    vector_candidate_ids: (
        Callable[
            [Sequence[tuple[str, float]], dict[str, object]],
            Sequence[str] | None,
        ]
        | None
    ) = None
    vector_ranking_order: (
        Callable[
            [Sequence[tuple[str, float]], dict[str, object]],
            Sequence[tuple[str, float]],
        ]
        | None
    ) = None
    score_adjuster: Callable[[RecordSearchCandidate], float] | None = None
    result_filter: Callable[[RecordSearchResult], bool] | None = None
    post_process: (
        Callable[[list[RecordSearchResult]], Sequence[RecordSearchResult]] | None
    ) = None


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


@dataclass(frozen=True, slots=True)
class RecordSearchResult:
    """A ranked, hydrated record with reusable kernel provenance."""

    record: Record
    score: float
    provenance: SearchResultProvenance

    @property
    def record_id(self) -> str:
        return self.record.source_id

    @property
    def storage_key(self) -> str:
        return self.record.storage_key

    def as_search_result(self) -> SearchResult:
        """Adapt this record result to the kernel's generic result model."""
        return SearchResult(
            record_id=self.record_id,
            score=self.score,
            source_kind=self.record.source_kind,
            workspace_id=self.record.workspace_id,
            metadata={"provenance": self.provenance.to_dict()},
        )


@dataclass(frozen=True, slots=True)
class RecordSearchFailure:
    """A source or hydration failure captured in degraded mode."""

    stage: FailureStage
    message: str
    exception_type: str = "Exception"


@dataclass(frozen=True, slots=True)
class RecordSearchOutcome:
    """Search results plus explicit degradation diagnostics."""

    results: tuple[RecordSearchResult, ...] = ()
    failures: tuple[RecordSearchFailure, ...] = ()
    missing_record_ids: tuple[str, ...] = ()
    cache_diagnostics: tuple[str, ...] = ()

    @property
    def degraded(self) -> bool:
        return bool(self.failures or self.missing_record_ids)


class RecordSearchError(RuntimeError):
    """Raised when strict retrieval cannot complete a pipeline stage."""

    def __init__(self, stage: FailureStage, error: Exception) -> None:
        super().__init__(f"{stage} retrieval failed: {error}")
        self.stage = stage
        self.error = error


@dataclass(frozen=True, slots=True)
class RecordSearchConfig:
    """Deterministic limits and optional graph/adaptive retrieval settings."""

    candidate_multiplier: int = 5
    minimum_candidate_limit: int = 1
    rrf_k: float = 60.0
    graph_fusion: Literal["rrf", "max"] = "rrf"
    graph_depth: int = 1
    max_graph_seeds: int = 10
    max_neighbors_per_seed: int = 10
    max_graph_concurrency: int = 8
    max_hydration_concurrency: int = 8
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
        filters = dict(filters or {})
        filters.setdefault("statuses", ["active"])
        acquisition_limit = max(
            max(limit, 1) * self._config.candidate_multiplier,
            self._config.minimum_candidate_limit,
        )
        candidate_key = self._candidate_cache_key(
            query,
            filters,
            limit,
            acquisition_limit,
            cache_diagnostics,
        )
        candidates: list[RecordSearchCandidate] | None = None
        if candidate_key is not None and not failures:
            try:
                cached_candidates = self._candidate_cache.get(candidate_key)
            except Exception as error:  # noqa: BLE001 - cache is optional
                cached_candidates = None
                cache_diagnostics.append(
                    f"candidate_cache:error:{type(error).__name__}"
                )
            if cached_candidates is not None:
                candidates = list(cached_candidates)
                cache_diagnostics.append("candidate_cache:hit")
            else:
                cache_diagnostics.append("candidate_cache:miss")

        if candidates is None:
            rankings: dict[str, list[RecordHit]] = {}
            if self._policy.vector_candidate_ids is not None:
                stage_results = await _gather_tasks(
                    [
                        asyncio.create_task(
                            _capture_stage(
                                "keyword",
                                lambda: self._acquire_keyword(
                                    query, acquisition_limit, filters
                                ),
                            )
                        )
                        if self._keyword_store is not None
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
                        lambda: self._acquire_vector(
                            cast(tuple[Vector, str, int], embedding_result),
                            acquisition_limit,
                            filters,
                            rankings,
                        ),
                    )
                    vector_value = self._consume_stage(vector_result, failures)
                    if vector_value is not None:
                        rankings["vector"] = cast(list[RecordHit], vector_value)
            else:
                stage_results = await _gather_tasks(
                    [
                        asyncio.create_task(
                            _capture_stage(
                                "keyword",
                                lambda: self._acquire_keyword(
                                    query, acquisition_limit, filters
                                ),
                            )
                        )
                        if self._keyword_store is not None
                        else None,
                        asyncio.create_task(
                            _capture_stage(
                                "vector",
                                lambda: self._acquire_vector(
                                    None,
                                    acquisition_limit,
                                    filters,
                                    rankings,
                                    query=query,
                                ),
                            )
                        )
                        if self._vector_store is not None
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
                    [
                        [hit.storage_key for hit in ranking]
                        for ranking in rankings.values()
                    ],
                    k=self._config.rrf_k,
                )
            base_candidates = self._build_candidates(fused_scores, rankings)
            base_candidates = self._apply_candidate_policy(base_candidates)

            if self._graph_store is not None and base_candidates:
                try:
                    graph_ranking = await self._expand_graph(base_candidates)
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
                                [
                                    [hit.storage_key for hit in ranking]
                                    for ranking in rankings.values()
                                ],
                                k=self._config.rrf_k,
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

            candidates = self._apply_score_adjustments(candidates)
            candidates = self._sort_candidates(candidates)
            if candidate_key is not None:
                try:
                    self._candidate_cache.set(candidate_key, tuple(candidates))
                except Exception as error:  # noqa: BLE001 - cache is optional
                    cache_diagnostics.append(
                        f"candidate_cache:error:{type(error).__name__}"
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
        selected_candidates = candidates[:result_limit]
        batch_hydration = await self._hydrate_candidates(
            selected_candidates,
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

        if self._policy.post_process is not None:
            hydrated = list(self._policy.post_process(hydrated))

        return RecordSearchOutcome(
            results=tuple(hydrated),
            failures=tuple(failures),
            missing_record_ids=tuple(missing_record_ids),
            cache_diagnostics=tuple(cache_diagnostics),
        )

    async def _acquire_keyword(
        self,
        query: str,
        acquisition_limit: int,
        filters: dict[str, object],
    ) -> list[RecordHit]:
        store = self._keyword_store
        if store is None:
            return []
        return _normalize_hits(
            await _call_async(store.search, query, acquisition_limit, filters),
            filters,
        )

    def _candidate_cache_key(
        self,
        query: str,
        filters: dict[str, object],
        requested_limit: int,
        acquisition_limit: int,
        diagnostics: list[str],
    ) -> CandidateCacheKey | None:
        if not self._policy_cacheable():
            diagnostics.append("candidate_cache:bypass:unstable_policy")
            return None
        try:
            return CandidateCacheKey.build(
                query=query,
                filters=filters,
                requested_limit=requested_limit,
                acquisition_limit=acquisition_limit,
                adaptive_limit=(
                    self._config.maximum_limit
                    if self._config.adaptive_enabled
                    else None
                ),
                routing_fingerprint=fingerprint(
                    {
                        "name": self._routing_fingerprint,
                        "candidate_multiplier": self._config.candidate_multiplier,
                        "minimum_candidate_limit": (
                            self._config.minimum_candidate_limit
                        ),
                        "rrf_k": self._config.rrf_k,
                        "graph_fusion": self._config.graph_fusion,
                        "graph_depth": self._config.graph_depth,
                        "max_graph_seeds": self._config.max_graph_seeds,
                        "max_neighbors_per_seed": (
                            self._config.max_neighbors_per_seed
                        ),
                        "adaptive_enabled": self._config.adaptive_enabled,
                        "score_ratio_floor": self._config.score_ratio_floor,
                        "minimum_score": self._config.minimum_score,
                        "maximum_score_gap": self._config.maximum_score_gap,
                    }
                ),
                encoder_namespace=self._encoder_namespace_for_provider(),
                epochs=self._cache_epochs(),
                policy_version=self._policy_version,
            )
        except (UnstableCacheKey, ValueError) as error:
            diagnostics.append(
                f"candidate_cache:bypass:{type(error).__name__}"
            )
            return None

    def _policy_cacheable(self) -> bool:
        if self._policy_version is not None:
            return True
        return not any(
            value is not None
            for value in (
                self._policy.candidate_filter,
                self._policy.vector_candidate_ids,
                self._policy.vector_ranking_order,
                self._policy.score_adjuster,
                self._policy.result_filter,
                self._policy.post_process,
            )
        )

    def _cache_epochs(self) -> SearchEpochs:
        values = {
            lane: _read_lane_epoch(store, lane)
            for lane, store in (
                ("keyword", self._keyword_store),
                ("vector", self._vector_store),
                ("graph", self._graph_store),
            )
        }
        missing = [
            lane
            for lane, store in (
                ("keyword", self._keyword_store),
                ("vector", self._vector_store),
                ("graph", self._graph_store),
            )
            if store is not None and values[lane] is None
        ]
        if missing:
            raise UnstableCacheKey(
                f"missing mutation epoch for {', '.join(missing)} lane"
            )
        return SearchEpochs(
            keyword=values["keyword"] or 0,
            vector=values["vector"] or 0,
            graph=values["graph"] or 0,
        )

    def _encoder_namespace_for_provider(self) -> str | None:
        provider = self._embedding_provider
        if provider is None:
            return self._encoder_namespace
        explicit = self._encoder_namespace
        if explicit:
            return explicit
        for name in ("encoder_namespace", "encoder_fingerprint", "fingerprint"):
            value = getattr(provider, name, None)
            if isinstance(value, str) and value:
                return value
        model_name = self._embedding_model_name or getattr(
            provider, "model_name", None
        )
        dim = self._embedding_dim or getattr(provider, "dim", None)
        if model_name is None:
            return None
        return f"{model_name}|dim={dim}"

    async def _acquire_vector(
        self,
        embedding: tuple[Vector, str, int] | None,
        acquisition_limit: int,
        filters: dict[str, object],
        rankings: Mapping[str, Sequence[RecordHit]],
        *,
        query: str | None = None,
    ) -> list[RecordHit]:
        if embedding is None:
            if query is None:
                raise ValueError("query is required when embedding is absent")
            embedding = await self._query_embedding(query)
        vector, model_name, dim = embedding
        vector_filters = filters
        if self._policy.vector_candidate_ids is not None:
            candidate_ids = self._policy.vector_candidate_ids(
                [
                    (hit.source_id, hit.score)
                    for hit in rankings.get("keyword", ())
                ],
                filters,
            )
            if candidate_ids is not None:
                vector_filters = dict(filters)
                vector_filters["candidate_ids"] = list(candidate_ids)
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
            filters,
            sort=False,
        )
        if self._policy.vector_ranking_order is not None:
            vector_ranking = _normalize_hits(
                self._policy.vector_ranking_order(
                    [(hit.source_id, hit.score) for hit in vector_ranking],
                    filters,
                ),
                filters,
                sort=False,
            )
        return vector_ranking

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
            return cached + loaded

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
        return cached + hydrated

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
        self, candidates: Sequence[RecordSearchCandidate]
    ) -> list[RecordHit]:
        graph_store = self._graph_store
        if graph_store is None:
            return []
        graph_seeds = self._sort_candidates(candidates)[
            : self._config.max_graph_seeds
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
        neighbors_by_seed = await self._load_graph_neighbors(graph_store, graph_seeds)
        for seed_key in seed_scores:
            seed = seed_by_key[seed_key]
            raw_neighbors = neighbors_by_seed.get(seed_key, ())
            sorted_neighbors = sorted(
                (
                    _normalize_graph_neighbor(neighbor, seed.identity)
                    for neighbor in raw_neighbors
                ),
                key=lambda item: (-item[2], item[0], item[1]),
            )
            edges: list[TypedGraphEdge[str, tuple[str, float]]] = []
            for target_id, edge_type, weight in sorted_neighbors:
                edge_key = (edge_type, weight)
                edges.append(TypedGraphEdge(target_id, edge_key))
                discounts[edge_key] = weight
            edges_by_seed[seed_key] = edges

        expanded = expand_bounded_typed_graph(
            seed_scores,
            lambda seed_id: edges_by_seed[seed_id],
            discounts,
            max_seed_count=self._config.max_graph_seeds,
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
    ) -> dict[str, Sequence[GraphNeighbor | tuple[str, str, float]]]:
        identities = [candidate.identity for candidate in graph_seeds]
        neighbors_many = getattr(graph_store, "neighbors_many", None)
        if callable(neighbors_many):
            result = await _call_async(
                neighbors_many,
                identities,
                depth=self._config.graph_depth,
            )
            return dict(
                cast(
                    Mapping[
                        str,
                        Sequence[GraphNeighbor | tuple[str, str, float]],
                    ],
                    result,
                )
            )

        semaphore = asyncio.Semaphore(self._config.max_graph_concurrency)

        async def load(
            seed: RecordSearchCandidate,
        ) -> tuple[str, Sequence[GraphNeighbor | tuple[str, str, float]]]:
            async with semaphore:
                try:
                    neighbors = await _call_async(
                        graph_store.neighbors,
                        seed.identity,
                        depth=self._config.graph_depth,
                    )
                except TypeError:
                    logger.warning(
                        "graph adapter rejected canonical identity; "
                        "using legacy source_id fallback for %s",
                        seed.storage_key,
                    )
                    neighbors = await _call_async(
                        graph_store.neighbors,
                        seed.record_id,
                        depth=self._config.graph_depth,
                    )
                return seed.storage_key, cast(
                    Sequence[GraphNeighbor | tuple[str, str, float]],
                    neighbors,
                )

        loaded = await _gather_tasks(
            [asyncio.create_task(load(seed)) for seed in graph_seeds]
        )
        return dict(cast(list[tuple[str, Sequence[GraphNeighbor | tuple[str, str, float]]]], loaded))

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
                encoder_namespace=self._encoder_namespace_for_provider(),
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
        return await _call_async(
            cast(RecordHydratorCallable, self._hydrator),
            identity.source_id,
        )

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
        if self._config.rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
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


def _normalize_hits(
    results: Sequence[RecordHit | tuple[str, float]],
    filters: Mapping[str, object],
    *,
    sort: bool = True,
) -> list[RecordHit]:
    workspace_id = filters.get("workspace_id")
    if workspace_id is not None and not isinstance(workspace_id, str):
        workspace_id = str(workspace_id)
    source_kind = filters.get("source_kind")
    if not isinstance(source_kind, str):
        source_kinds = filters.get("source_kinds")
        source_kind = (
            source_kinds[0]
            if isinstance(source_kinds, list)
            and len(source_kinds) == 1
            and isinstance(source_kinds[0], str)
            else "legacy"
        )
    normalized: list[RecordHit] = []
    for result in results:
        if isinstance(result, RecordHit):
            normalized.append(result)
        else:
            source_id, score = result
            normalized.append(
                RecordHit(
                    RecordIdentity(workspace_id, source_kind, source_id),
                    score,
                )
            )
    best: dict[str, RecordHit] = {}
    for hit in normalized:
        current = best.get(hit.storage_key)
        if current is None or hit.score > current.score:
            best[hit.storage_key] = hit
    normalized = list(best.values())
    if sort:
        normalized.sort(key=lambda hit: (-hit.score, hit.storage_key))
    return normalized


def _read_lane_epoch(store: object | None, lane: str) -> int | None:
    if store is None:
        return None
    epochs = getattr(store, "epochs", None)
    if callable(epochs):
        try:
            values = epochs()
        except Exception:  # noqa: BLE001 - cache key reads must be best effort
            values = None
        if isinstance(values, Mapping):
            value = values.get(lane)
            if isinstance(value, int):
                return value
    lane_epoch = getattr(store, f"{lane}_epoch", None)
    if callable(lane_epoch):
        try:
            value = lane_epoch()
        except Exception:  # noqa: BLE001 - cache key reads must be best effort
            value = None
        if isinstance(value, int):
            return value
    if lane == "vector":
        epoch = getattr(store, "epoch", None)
        if callable(epoch):
            try:
                value = epoch()
            except Exception:  # noqa: BLE001 - cache key reads must be best effort
                value = None
            if isinstance(value, int):
                return value
    return 0


def _graph_hit(record_id: str, expansion: Any, seed_by_key: Mapping[str, RecordSearchCandidate]) -> RecordHit:
    seed = seed_by_key[expansion.provenance.seed_id]
    if record_id == seed.storage_key:
        identity = seed.identity
    elif record_id.startswith("record:"):
        identity = RecordIdentity.from_storage_key(record_id)
    else:
        identity = RecordIdentity(seed.workspace_id, seed.source_kind, record_id)
    return RecordHit(
        identity,
        expansion.contribution,
    )


def _normalize_graph_neighbor(
    neighbor: GraphNeighbor | tuple[str, str, float],
    seed_identity: RecordIdentity,
) -> tuple[str, str, float]:
    if isinstance(neighbor, GraphNeighbor):
        return neighbor.identity.storage_key, neighbor.edge_type, neighbor.weight
    target_id, edge_type, weight = neighbor
    if target_id.startswith("record:"):
        target_id = RecordIdentity.from_storage_key(target_id).storage_key
    else:
        target_id = RecordIdentity(
            seed_identity.workspace_id,
            seed_identity.source_kind,
            target_id,
        ).storage_key
    return target_id, edge_type, weight


async def _call_async[T](
    function: Callable[..., T | Awaitable[T]],
    *args: Any,
    **kwargs: Any,
) -> T:
    """Run blocking adapter calls away from the async event loop."""
    if inspect.iscoroutinefunction(function):
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


def _find_stage(
    results: Sequence[tuple[FailureStage, Any, Exception | None]],
    stage: FailureStage,
) -> tuple[FailureStage, Any, Exception | None] | None:
    for result_stage, value, error in results:
        if result_stage == stage:
            return result_stage, value, error
    return None


async def _search_vector_store(
    store: VectorStore | AsyncVectorStore,
    vector: Vector,
    k: int,
    *,
    model_name: str,
    dim: int,
    filters: dict[str, object] | None,
) -> Sequence[RecordHit | tuple[str, float]]:
    async_search = getattr(store, "async_search", None)
    if callable(async_search):
        return await _call_async(
            cast(
                Callable[
                    ...,
                    Sequence[RecordHit | tuple[str, float]]
                    | Awaitable[Sequence[RecordHit | tuple[str, float]]],
                ],
                async_search,
            ),
            vector,
            k,
            model_name=model_name,
            dim=dim,
            filters=filters,
        )
    return await _call_async(
        store.search,
        vector,
        k,
        model_name=model_name,
        dim=dim,
        filters=filters,
    )
