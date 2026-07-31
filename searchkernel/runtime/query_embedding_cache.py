"""Bounded, model-safe caches for repeated query embeddings."""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from threading import Event, Lock

from searchkernel.domain import Vector


def normalize_query(query: str) -> str:
    """Normalize only surrounding and repeated whitespace."""
    return " ".join(query.strip().split())


@dataclass(frozen=True, slots=True)
class QueryEmbeddingCacheMetrics:
    """Counters and timing for one query embedding cache."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    coalesced_waiters: int = 0
    compute_time_seconds: float = 0.0


@dataclass(slots=True)
class _CachedEmbedding:
    embedding: tuple[float, ...]
    expires_at: float


@dataclass(slots=True)
class _InFlightEmbedding:
    completed: Event = field(default_factory=Event)
    embedding: tuple[float, ...] | None = None
    error: BaseException | None = None


class QueryEmbeddingCache:
    """Cache ``(encoder_namespace, normalized_query)`` embeddings.

    Synchronous callers use a thread event for miss coalescing. Async callers
    wait for that event in a worker thread, so no blocking wait runs on the
    event loop. The in-flight state is shared by both APIs.
    """

    def __init__(
        self,
        *,
        ttl_seconds: float = 300.0,
        max_entries: int = 128,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be > 0")
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        self._ttl_seconds = ttl_seconds
        self._max_entries = max_entries
        self._clock = clock
        self._cache: OrderedDict[tuple[str, str], _CachedEmbedding] = OrderedDict()
        self._inflight: dict[tuple[str, str], _InFlightEmbedding] = {}
        self._lock = Lock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._coalesced_waiters = 0
        self._compute_time_seconds = 0.0

    def get_or_compute(
        self,
        *,
        model_name: str | None = None,
        encoder_namespace: str | None = None,
        query: str,
        compute: Callable[[], Vector],
    ) -> Vector:
        """Return a cached embedding or synchronously compute one."""
        key = self._key(model_name, encoder_namespace, query)
        cached = self._load(key)
        if cached is not None:
            return cached

        inflight, is_leader = self._start_or_join(key)
        if not is_leader:
            inflight.completed.wait()
            return self._finish_waiter(inflight)

        started = self._clock()
        try:
            embedding = tuple(float(value) for value in compute())
            self._complete_success(key, inflight, embedding)
            return list(embedding)
        except BaseException as error:
            self._complete_failure(key, inflight, error)
            raise
        finally:
            self._record_compute_time(self._clock() - started)

    async def async_get_or_compute(
        self,
        *,
        model_name: str | None = None,
        encoder_namespace: str | None = None,
        query: str,
        compute: Callable[[], Vector | Awaitable[Vector]],
    ) -> Vector:
        """Return a cached embedding without blocking the event loop."""
        key = self._key(model_name, encoder_namespace, query)
        cached = self._load(key)
        if cached is not None:
            return cached

        inflight, is_leader = self._start_or_join(key)
        if not is_leader:
            await asyncio.to_thread(inflight.completed.wait)
            return self._finish_waiter(inflight)

        started = self._clock()
        try:
            result = compute()
            embedding = tuple(float(value) for value in await _maybe_await(result))
            self._complete_success(key, inflight, embedding)
            return list(embedding)
        except BaseException as error:
            self._complete_failure(key, inflight, error)
            raise
        finally:
            self._record_compute_time(self._clock() - started)

    @property
    def metrics(self) -> QueryEmbeddingCacheMetrics:
        """Return a consistent snapshot of cache metrics."""
        with self._lock:
            return QueryEmbeddingCacheMetrics(
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
                coalesced_waiters=self._coalesced_waiters,
                compute_time_seconds=self._compute_time_seconds,
            )

    def clear(self) -> None:
        """Clear completed entries without disturbing active computations."""
        with self._lock:
            self._cache.clear()

    def _key(
        self,
        model_name: str | None,
        encoder_namespace: str | None,
        query: str,
    ) -> tuple[str, str]:
        namespace = encoder_namespace or model_name
        if not namespace:
            raise ValueError("encoder_namespace or model_name is required")
        return namespace, normalize_query(query)

    def _start_or_join(
        self,
        key: tuple[str, str],
    ) -> tuple[_InFlightEmbedding, bool]:
        with self._lock:
            inflight = self._inflight.get(key)
            if inflight is not None:
                self._coalesced_waiters += 1
                return inflight, False
            self._misses += 1
            inflight = _InFlightEmbedding()
            self._inflight[key] = inflight
            return inflight, True

    def _load(self, key: tuple[str, str]) -> Vector | None:
        now = self._clock()
        with self._lock:
            self._evict_expired(now)
            cached = self._cache.get(key)
            if cached is None:
                return None
            self._hits += 1
            self._cache.move_to_end(key)
            return list(cached.embedding)

    def _complete_success(
        self,
        key: tuple[str, str],
        inflight: _InFlightEmbedding,
        embedding: tuple[float, ...],
    ) -> None:
        now = self._clock()
        with self._lock:
            inflight.embedding = embedding
            self._cache[key] = _CachedEmbedding(
                embedding=embedding,
                expires_at=now + self._ttl_seconds,
            )
            self._cache.move_to_end(key)
            while len(self._cache) > self._max_entries:
                self._cache.popitem(last=False)
                self._evictions += 1
            self._inflight.pop(key, None)
            inflight.completed.set()

    def _complete_failure(
        self,
        key: tuple[str, str],
        inflight: _InFlightEmbedding,
        error: BaseException,
    ) -> None:
        with self._lock:
            inflight.error = error
            self._inflight.pop(key, None)
            inflight.completed.set()

    def _finish_waiter(self, inflight: _InFlightEmbedding) -> Vector:
        if inflight.embedding is not None:
            return list(inflight.embedding)
        if inflight.error is not None:
            raise inflight.error
        raise RuntimeError("query embedding coalescing completed without a result")

    def _evict_expired(self, now: float) -> None:
        expired = [
            key
            for key, cached in self._cache.items()
            if cached.expires_at <= now
        ]
        for key in expired:
            self._cache.pop(key, None)
            self._evictions += 1

    def _record_compute_time(self, elapsed: float) -> None:
        with self._lock:
            self._compute_time_seconds += elapsed


async def _maybe_await(value: Vector | Awaitable[Vector]) -> Vector:
    if asyncio.iscoroutine(value) or isinstance(value, Awaitable):
        return await value
    return value


_DEFAULT_QUERY_EMBEDDING_CACHE = QueryEmbeddingCache()


def get_or_compute_query_embedding(
    *,
    model_name: str | None = None,
    encoder_namespace: str | None = None,
    query: str,
    compute: Callable[[], Vector],
) -> Vector:
    """Use the process-wide synchronous query embedding cache."""
    return _DEFAULT_QUERY_EMBEDDING_CACHE.get_or_compute(
        model_name=model_name,
        encoder_namespace=encoder_namespace,
        query=query,
        compute=compute,
    )


def clear_query_embedding_cache() -> None:
    """Clear the process-wide query embedding cache."""
    _DEFAULT_QUERY_EMBEDDING_CACHE.clear()


def get_query_embedding_cache() -> QueryEmbeddingCache:
    """Return the process-wide query embedding cache."""
    return _DEFAULT_QUERY_EMBEDDING_CACHE
