"""Source-agnostic retrieval for hydrated records."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, cast

from searchkernel.domain import (
    Record,
    SearchResult,
    SearchResultProvenance,
    Vector,
)
from searchkernel.ports import EmbeddingProvider, GraphStore, KeywordStore, VectorStore
from searchkernel.search.adaptive_limit import resolve_adaptive_result_limit
from searchkernel.search.bounded_graph import (
    TypedGraphEdge,
    expand_bounded_typed_graph,
)
from searchkernel.search.fusion import fuse_reciprocal_rank

RecordHydratorCallable = Callable[[str], Record | None]
QueryEmbeddingCallable = Callable[[str], Vector]
FailureStage = Literal["keyword", "vector", "graph", "hydration"]


class RecordHydrator(Protocol):
    """Hydrate a record without mutating source state."""

    def hydrate_record(self, record_id: str) -> Record | None: ...


class QueryEmbeddingProvider(Protocol):
    """Generate one query embedding for a vector search."""

    @property
    def model_name(self) -> str: ...

    @property
    def dim(self) -> int: ...

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
    score_adjuster: Callable[[RecordSearchCandidate], float] | None = None
    result_filter: Callable[[RecordSearchResult], bool] | None = None
    post_process: (
        Callable[[list[RecordSearchResult]], Sequence[RecordSearchResult]] | None
    ) = None


@dataclass(frozen=True, slots=True)
class RecordSearchCandidate:
    """An unhydrated ranked record candidate."""

    record_id: str
    score: float
    provenance: SearchResultProvenance


@dataclass(frozen=True, slots=True)
class RecordSearchResult:
    """A ranked, hydrated record with reusable kernel provenance."""

    record: Record
    score: float
    provenance: SearchResultProvenance

    @property
    def record_id(self) -> str:
        return self.record.source_id

    def as_search_result(self) -> SearchResult:
        """Adapt this record result to the kernel's generic result model."""
        return SearchResult(
            record_id=self.record_id,
            score=self.score,
            source_kind=self.record.source_kind,
            metadata={"provenance": self.provenance.to_dict()},
        )


@dataclass(frozen=True, slots=True)
class RecordSearchFailure:
    """A source or hydration failure captured in degraded mode."""

    stage: FailureStage
    message: str


@dataclass(frozen=True, slots=True)
class RecordSearchOutcome:
    """Search results plus explicit degradation diagnostics."""

    results: tuple[RecordSearchResult, ...] = ()
    failures: tuple[RecordSearchFailure, ...] = ()
    missing_record_ids: tuple[str, ...] = ()

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
    adaptive_enabled: bool = False
    maximum_limit: int = 100
    score_ratio_floor: float = 0.5
    minimum_score: float = 0.0
    maximum_score_gap: float = 1.0


class RecordSearchPipeline:
    """Acquire, fuse, filter, graph-expand, and hydrate generic records.

    The pipeline is read-only. Domain lifecycle, authorization, supersession,
    and side effects remain injectable application policies.
    """

    def __init__(
        self,
        *,
        hydrator: RecordHydrator | RecordHydratorCallable,
        keyword_store: KeywordStore | None = None,
        vector_store: VectorStore | None = None,
        graph_store: GraphStore | None = None,
        embedding_provider: (
            EmbeddingProvider
            | QueryEmbeddingProvider
            | Callable[[str], Vector]
            | None
        ) = None,
        embedding_model_name: str | None = None,
        embedding_dim: int | None = None,
        policy: RecordSearchPolicy | None = None,
        config: RecordSearchConfig | None = None,
        continue_on_error: bool = False,
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
        self._continue_on_error = continue_on_error
        self._validate_config()

    def search(
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
        filters = dict(filters or {})
        acquisition_limit = max(
            max(limit, 1) * self._config.candidate_multiplier,
            self._config.minimum_candidate_limit,
        )
        rankings: dict[str, list[tuple[str, float]]] = {}

        if self._keyword_store is not None:
            try:
                rankings["keyword"] = _sorted_unique(
                    self._keyword_store.search(query, acquisition_limit, filters)
                )
            except Exception as error:  # noqa: BLE001 - degraded mode captures backend failures
                self._handle_error("keyword", error, failures)

        if self._vector_store is not None:
            try:
                vector, model_name, dim = self._query_embedding(query)
                vector_filters = filters
                if self._policy.vector_candidate_ids is not None:
                    candidate_ids = self._policy.vector_candidate_ids(
                        rankings.get("keyword", ()),
                        filters,
                    )
                    if candidate_ids is not None:
                        vector_filters = dict(filters)
                        vector_filters["candidate_ids"] = list(candidate_ids)
                rankings["vector"] = _sorted_unique(
                    self._vector_store.search(
                        vector,
                        acquisition_limit,
                        model_name=model_name,
                        dim=dim,
                        filters=vector_filters,
                    )
                )
            except Exception as error:  # noqa: BLE001 - degraded mode captures backend failures
                self._handle_error("vector", error, failures)

        fused_scores: dict[str, float] = {}
        if rankings:
            fused_scores = fuse_reciprocal_rank(
                [[record_id for record_id, _ in ranking] for ranking in rankings.values()],
                k=self._config.rrf_k,
            )
        base_candidates = self._build_candidates(fused_scores, rankings)
        base_candidates = self._apply_candidate_policy(base_candidates)

        if self._graph_store is not None and base_candidates:
            try:
                graph_ranking = self._expand_graph(base_candidates)
                if graph_ranking:
                    rankings["graph"] = graph_ranking
                    if self._config.graph_fusion == "max":
                        fused_scores = dict(fused_scores)
                        for record_id, score in graph_ranking:
                            fused_scores[record_id] = max(
                                fused_scores.get(record_id, 0.0),
                                score,
                            )
                    else:
                        fused_scores = fuse_reciprocal_rank(
                            [
                                [record_id for record_id, _ in ranking]
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
        for candidate in candidates[:result_limit]:
            try:
                record = self._hydrate(candidate.record_id)
            except Exception as error:  # noqa: BLE001 - degraded mode captures hydration failures
                self._handle_error("hydration", error, failures)
                continue
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
        )

    def _build_candidates(
        self,
        fused_scores: Mapping[str, float],
        rankings: Mapping[str, Sequence[tuple[str, float]]],
    ) -> list[RecordSearchCandidate]:
        candidates: list[RecordSearchCandidate] = []
        for record_id, score in fused_scores.items():
            provenance = SearchResultProvenance()
            for strategy, ranking in rankings.items():
                for rank, (candidate_id, raw_score) in enumerate(ranking, start=1):
                    if candidate_id == record_id:
                        provenance.add_strategy(strategy, rank, raw_score)
                        break
            candidate = RecordSearchCandidate(record_id, score, provenance)
            candidates.append(candidate)
        return candidates

    def _apply_score_adjustments(
        self, candidates: Sequence[RecordSearchCandidate]
    ) -> list[RecordSearchCandidate]:
        if self._policy.score_adjuster is None:
            return list(candidates)
        return [
            RecordSearchCandidate(
                record_id=candidate.record_id,
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

    def _expand_graph(
        self, candidates: Sequence[RecordSearchCandidate]
    ) -> list[tuple[str, float]]:
        graph_store = self._graph_store
        if graph_store is None:
            return []
        seed_scores = {
            candidate.record_id: candidate.score
            for candidate in self._sort_candidates(candidates)
        }
        edges_by_seed: dict[str, list[TypedGraphEdge[str, tuple[str, float]]]] = {}
        discounts: dict[tuple[str, float], float] = {}
        for seed_id in seed_scores:
            neighbors = graph_store.neighbors(
                seed_id,
                depth=self._config.graph_depth,
            )
            sorted_neighbors = sorted(
                neighbors,
                key=lambda item: (-item[2], item[0], item[1]),
            )
            edges: list[TypedGraphEdge[str, tuple[str, float]]] = []
            for target_id, edge_type, weight in sorted_neighbors:
                edge_key = (edge_type, weight)
                edges.append(TypedGraphEdge(target_id, edge_key))
                discounts[edge_key] = weight
            edges_by_seed[seed_id] = edges

        expanded = expand_bounded_typed_graph(
            seed_scores,
            lambda seed_id: edges_by_seed[seed_id],
            discounts,
            max_seed_count=self._config.max_graph_seeds,
            max_neighbors_per_seed=self._config.max_neighbors_per_seed,
        )
        return sorted(
            (
                (record_id, expansion.contribution)
                for record_id, expansion in expanded.items()
            ),
            key=lambda item: (-item[1], item[0]),
        )

    def _query_embedding(self, query: str) -> tuple[Vector, str, int]:
        provider = self._embedding_provider
        if provider is None:
            raise ValueError("embedding_provider is required for vector search")

        if hasattr(provider, "embed_query"):
            vector = cast(QueryEmbeddingProvider, provider).embed_query(query)
        else:
            embed_method = getattr(provider, "embed", None)
            if callable(embed_method):
                embeddings = cast(
                    Callable[[list[str]], list[Vector]], embed_method
                )([query])
                if len(embeddings) != 1:
                    raise ValueError("embedding provider must return one query vector")
                vector = embeddings[0]
            else:
                vector = cast(QueryEmbeddingCallable, provider)(query)

        model_name = self._embedding_model_name or getattr(
            provider, "model_name", None
        )
        dim = self._embedding_dim or getattr(provider, "dim", None)
        if model_name is None or dim is None:
            raise ValueError(
                "vector search requires embedding model name and dimension"
            )
        if len(vector) != dim:
            raise ValueError(
                f"query embedding has dimension {len(vector)}, expected {dim}"
            )
        return vector, model_name, dim

    def _hydrate(self, record_id: str) -> Record | None:
        if hasattr(self._hydrator, "hydrate_record"):
            return cast(RecordHydrator, self._hydrator).hydrate_record(record_id)
        return cast(RecordHydratorCallable, self._hydrator)(record_id)

    def _handle_error(
        self,
        stage: FailureStage,
        error: Exception,
        failures: list[RecordSearchFailure],
    ) -> None:
        if not self._continue_on_error:
            raise RecordSearchError(stage, error) from error
        failures.append(RecordSearchFailure(stage, str(error)))

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

    @staticmethod
    def _sort_candidates(
        candidates: Sequence[RecordSearchCandidate],
    ) -> list[RecordSearchCandidate]:
        return sorted(candidates, key=lambda item: (-item.score, item.record_id))


def _sorted_unique(
    results: Sequence[tuple[str, float]],
) -> list[tuple[str, float]]:
    best_scores: dict[str, float] = {}
    for record_id, score in results:
        best_scores[record_id] = max(score, best_scores.get(record_id, score))
    return sorted(best_scores.items(), key=lambda item: (-item[1], item[0]))
