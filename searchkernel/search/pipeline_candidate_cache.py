"""Candidate-cache policy for the record search pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, TypeVar

from searchkernel.runtime import (
    CandidateCacheKey,
    CandidateResultCache,
    SearchEpochs,
    UnstableCacheKey,
    fingerprint,
)

if TYPE_CHECKING:
    from searchkernel.search.record_pipeline import (
        RecordSearchConfig,
        RecordSearchPolicy,
    )

CandidateT = TypeVar("CandidateT")


class CandidateCachePolicy[CandidateT]:
    """Build stable candidate keys and mediate optional cache access."""

    def __init__(
        self,
        cache: CandidateResultCache[tuple[CandidateT, ...]],
        *,
        config: RecordSearchConfig,
        policy: RecordSearchPolicy,
        keyword_store: object | None,
        vector_store: object | None,
        graph_store: object | None,
        embedding_provider: object | None,
        embedding_model_name: str | None,
        embedding_dim: int | None,
        encoder_namespace: str | None,
        routing_fingerprint: str,
        policy_version: str | None,
    ) -> None:
        self._cache = cache
        self._config = config
        self._policy = policy
        self._stores = {
            "keyword": keyword_store,
            "vector": vector_store,
            "graph": graph_store,
        }
        self._embedding_provider = embedding_provider
        self._embedding_model_name = embedding_model_name
        self._embedding_dim = embedding_dim
        self._encoder_namespace = encoder_namespace
        self._routing_fingerprint = routing_fingerprint
        self._policy_version = policy_version

    def key(
        self,
        query: str,
        filters: Mapping[str, object],
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
                        "keyword_candidate_multiplier": (
                            self._config.keyword_candidate_multiplier
                        ),
                        "vector_candidate_multiplier": (
                            self._config.vector_candidate_multiplier
                        ),
                        "keyword_candidate_budget": (
                            self._config.keyword_candidate_budget
                        ),
                        "vector_candidate_budget": (
                            self._config.vector_candidate_budget
                        ),
                        "minimum_candidate_limit": (
                            self._config.minimum_candidate_limit
                        ),
                        "rrf_k": self._config.rrf_k,
                        "weighted_rrf_enabled": self._config.weighted_rrf_enabled,
                        "fusion_mode": self._config.fusion_mode,
                        "base_semantic_weight": self._config.base_semantic_weight,
                        "base_keyword_weight": self._config.base_keyword_weight,
                        "base_graph_weight": self._config.base_graph_weight,
                        "graph_fusion": self._config.graph_fusion,
                        "graph_enabled": self._config.graph_enabled,
                        "adaptive_graph_enabled": (
                            self._config.adaptive_graph_enabled
                        ),
                        "adaptive_graph_min_seed_score": (
                            self._config.adaptive_graph_min_seed_score
                        ),
                        "adaptive_graph_min_seed_count": (
                            self._config.adaptive_graph_min_seed_count
                        ),
                        "graph_only_penalty": self._config.graph_only_penalty,
                        "graph_depth": self._config.graph_depth,
                        "max_graph_seeds": self._config.max_graph_seeds,
                        "max_neighbors_per_seed": (
                            self._config.max_neighbors_per_seed
                        ),
                        "adaptive_enabled": self._config.adaptive_enabled,
                        "score_ratio_floor": self._config.score_ratio_floor,
                        "minimum_score": self._config.minimum_score,
                        "maximum_score_gap": self._config.maximum_score_gap,
                        "artifact_confidence_threshold": (
                            self._config.artifact_confidence_threshold
                        ),
                        "expansion_enabled": self._config.expansion_enabled,
                        "expansion_timeout_s": self._config.expansion_timeout_s,
                        "expansion_top_k": self._config.expansion_top_k,
                        "expansion_similarity_threshold": (
                            self._config.expansion_similarity_threshold
                        ),
                        "synonym_expansion_enabled": (
                            self._config.synonym_expansion_enabled
                        ),
                        "synonym_expansion_max_terms": (
                            self._config.synonym_expansion_max_terms
                        ),
                        "query_expander": self._policy.query_expander is not None,
                        "parent_expansion": (
                            self._policy.parent_expander is not None
                        ),
                        "graph_target_resolution": (
                            self._policy.graph_target_resolver is not None
                        ),
                    }
                ),
                encoder_namespace=self.encoder_namespace(),
                epochs=self._cache_epochs(),
                policy_version=self._policy_version,
            )
        except (UnstableCacheKey, ValueError) as error:
            diagnostics.append(
                f"candidate_cache:bypass:{type(error).__name__}"
            )
            return None

    def get(
        self,
        key: CandidateCacheKey | None,
        diagnostics: list[str],
    ) -> list[CandidateT] | None:
        if key is None:
            return None
        try:
            cached = self._cache.get(key)
        except Exception as error:  # noqa: BLE001 - cache is optional
            diagnostics.append(f"candidate_cache:error:{type(error).__name__}")
            return None
        if cached is None:
            diagnostics.append("candidate_cache:miss")
            return None
        diagnostics.append("candidate_cache:hit")
        return list(cached)

    def set(
        self,
        key: CandidateCacheKey | None,
        candidates: list[CandidateT],
        diagnostics: list[str],
    ) -> None:
        if key is None:
            return
        try:
            self._cache.set(key, tuple(candidates))
        except Exception as error:  # noqa: BLE001 - cache is optional
            diagnostics.append(f"candidate_cache:error:{type(error).__name__}")

    async def async_wait_for_miss(
        self,
        key: CandidateCacheKey | None,
        diagnostics: list[str],
    ) -> list[CandidateT] | None:
        if key is None:
            return None
        try:
            leader, candidates = await self._cache.async_wait_for_miss(key)
        except Exception as error:  # noqa: BLE001 - cache is optional
            diagnostics.append(f"candidate_cache:error:{type(error).__name__}")
            return None
        if leader:
            return None
        diagnostics.append("candidate_cache:coalesced")
        return list(candidates or ())

    def encoder_namespace(self) -> str | None:
        provider = self._embedding_provider
        if provider is None:
            return self._encoder_namespace
        if self._encoder_namespace:
            return self._encoder_namespace
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
                self._policy.parent_expander,
            )
        )

    def _cache_epochs(self) -> SearchEpochs:
        values = _read_bulk_epochs(self._stores)
        for lane, store in self._stores.items():
            if lane not in values:
                epoch = _read_lane_epoch(store, lane)
                if epoch is not None:
                    values[lane] = epoch
        missing = [
            lane
            for lane, store in self._stores.items()
            if store is not None and values.get(lane) is None
        ]
        if missing:
            raise UnstableCacheKey(
                f"missing mutation epoch for {', '.join(missing)} lane"
            )
        return SearchEpochs(
            keyword=values.get("keyword") or 0,
            vector=values.get("vector") or 0,
            graph=values.get("graph") or 0,
        )


def _read_bulk_epochs(stores: Mapping[str, object | None]) -> dict[str, int]:
    """Read lane epochs in one backend call when an adapter supports it."""
    values: dict[str, int] = {}
    seen: set[int] = set()
    for store in stores.values():
        if store is None or id(store) in seen:
            continue
        seen.add(id(store))
        epochs = getattr(store, "epochs", None)
        if not callable(epochs):
            continue
        try:
            raw = epochs()
        except Exception:  # noqa: BLE001, S112 - scalar fallback is best effort
            continue
        if isinstance(raw, SearchEpochs):
            candidate = {
                lane: raw.for_lane(lane)
                for lane in ("keyword", "vector", "graph")
            }
        elif isinstance(raw, Mapping):
            candidate = {
                lane: value
                for lane, value in raw.items()
                if lane in ("keyword", "vector", "graph")
                and isinstance(value, int)
                and not isinstance(value, bool)
                and value >= 0
            }
        else:
            continue
        values.update(candidate)
    return values


def _read_lane_epoch(store: object | None, lane: str) -> int | None:
    if store is None:
        return None
    lane_epoch = getattr(store, f"{lane}_epoch", None)
    if callable(lane_epoch):
        try:
            value = lane_epoch()
        except Exception:  # noqa: BLE001 - cache key reads must be best effort
            value = None
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    if lane == "vector":
        epoch = getattr(store, "epoch", None)
        if callable(epoch):
            try:
                value = epoch()
            except Exception:  # noqa: BLE001 - cache key reads must be best effort
                value = None
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                return value
    return None
