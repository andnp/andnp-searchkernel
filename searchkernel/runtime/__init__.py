"""Runtime support for the search kernel: tracing, caching, etc."""

from searchkernel.runtime.canonical_cache import (
    BoundedCacheMetrics,
    CandidateCacheKey,
    CandidateResultCache,
    HydrationCache,
    HydrationCacheKey,
    SearchEpochs,
    UnstableCacheKey,
    fingerprint,
    normalize_cache_query,
    stable_json,
)
from searchkernel.runtime.query_embedding_cache import (
    QueryEmbeddingCache,
    QueryEmbeddingCacheMetrics,
    clear_query_embedding_cache,
    get_or_compute_query_embedding,
    normalize_query,
)

__all__ = [
    "BoundedCacheMetrics",
    "CandidateCacheKey",
    "CandidateResultCache",
    "HydrationCache",
    "HydrationCacheKey",
    "QueryEmbeddingCache",
    "QueryEmbeddingCacheMetrics",
    "SearchEpochs",
    "UnstableCacheKey",
    "clear_query_embedding_cache",
    "fingerprint",
    "get_or_compute_query_embedding",
    "normalize_cache_query",
    "normalize_query",
    "stable_json",
]
