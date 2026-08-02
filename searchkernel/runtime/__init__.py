"""Runtime support for the search kernel: tracing, caching, etc."""

from searchkernel.ports.epochs import SearchEpochProvider
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
    get_query_embedding_cache,
    normalize_query,
)
from searchkernel.runtime.validated_read_cache import (
    ValidatedCacheEntry,
    ValidatedCacheStore,
    ValidatedCacheValue,
    ValidatedReadThroughCache,
    ValidatedReadThroughCacheMetrics,
)

__all__ = [
    "BoundedCacheMetrics",
    "CandidateCacheKey",
    "CandidateResultCache",
    "HydrationCache",
    "HydrationCacheKey",
    "QueryEmbeddingCache",
    "QueryEmbeddingCacheMetrics",
    "SearchEpochProvider",
    "SearchEpochs",
    "UnstableCacheKey",
    "ValidatedCacheEntry",
    "ValidatedCacheStore",
    "ValidatedCacheValue",
    "ValidatedReadThroughCache",
    "ValidatedReadThroughCacheMetrics",
    "clear_query_embedding_cache",
    "fingerprint",
    "get_or_compute_query_embedding",
    "get_query_embedding_cache",
    "normalize_cache_query",
    "normalize_query",
    "stable_json",
]
