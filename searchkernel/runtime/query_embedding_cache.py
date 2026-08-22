"""Bounded, model-safe caches for repeated query embeddings."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from concurrent.futures import Future
from dataclasses import dataclass, field
from threading import Lock

from searchkernel.domain import Vector
from searchkernel.ports.stores import CacheStore

logger = logging.getLogger(__name__)


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
    completed: Future[None] = field(default_factory=Future)
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
        store: CacheStore | None = None,
        epoch: int = 0,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be > 0")
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        self._ttl_seconds = ttl_seconds
        self._max_entries = max_entries
        self._clock = clock
        self._store = store
        self._epoch = epoch
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
            inflight.completed.result()
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
        inflight, is_leader = self._start_or_join(key, count_miss=False)
        if not is_leader:
            await asyncio.wrap_future(inflight.completed)
            return self._finish_waiter(inflight)

        try:
            cached = await asyncio.to_thread(self._load, key)
        except BaseException as error:
            self._complete_failure(key, inflight, error)
            raise
        if cached is not None:
            self._complete_success(key, inflight, tuple(cached), persist=False)
            return cached

        self._record_miss()
        started = self._clock()
        try:
            result = compute()
            embedding = tuple(float(value) for value in await _maybe_await(result))
            self._complete_success(key, inflight, embedding, persist=False)
        except BaseException as error:
            self._complete_failure(key, inflight, error)
            self._record_compute_time(self._clock() - started)
            raise
        try:
            await asyncio.to_thread(self._save_to_store, key, embedding)
        finally:
            self._record_compute_time(self._clock() - started)
        return list(embedding)

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

    def invalidate_epoch(self, epoch: int) -> None:
        """Discard entries from the supplied epoch or earlier.

        Clears the whole in-memory cache, since entries aren't tracked
        per-epoch locally; the backing store (if any) prunes precisely.
        """
        with self._lock:
            self._cache.clear()
        if self._store is not None:
            try:
                self._store.invalidate_epoch(epoch)
            except Exception:
                logger.warning(
                    "query_embedding_cache_store_invalidate_failed", exc_info=True
                )

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
        *,
        count_miss: bool = True,
    ) -> tuple[_InFlightEmbedding, bool]:
        with self._lock:
            inflight = self._inflight.get(key)
            if inflight is not None:
                self._coalesced_waiters += 1
                return inflight, False
            if count_miss:
                self._misses += 1
            inflight = _InFlightEmbedding()
            self._inflight[key] = inflight
            return inflight, True

    def _record_miss(self) -> None:
        with self._lock:
            self._misses += 1

    def _load(self, key: tuple[str, str]) -> Vector | None:
        now = self._clock()
        with self._lock:
            self._evict_expired(now)
            cached = self._cache.get(key)
            if cached is not None:
                self._hits += 1
                self._cache.move_to_end(key)
                return list(cached.embedding)
        embedding = self._load_from_store(key)
        if embedding is None:
            return None
        with self._lock:
            self._hits += 1
            self._cache[key] = _CachedEmbedding(
                embedding=embedding,
                expires_at=now + self._ttl_seconds,
            )
            self._cache.move_to_end(key)
            while len(self._cache) > self._max_entries:
                self._cache.popitem(last=False)
                self._evictions += 1
        return list(embedding)

    def _complete_success(
        self,
        key: tuple[str, str],
        inflight: _InFlightEmbedding,
        embedding: tuple[float, ...],
        *,
        persist: bool = True,
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
            inflight.completed.set_result(None)
        if persist:
            self._save_to_store(key, embedding)

    def _complete_failure(
        self,
        key: tuple[str, str],
        inflight: _InFlightEmbedding,
        error: BaseException,
    ) -> None:
        with self._lock:
            inflight.error = error
            self._inflight.pop(key, None)
            inflight.completed.set_result(None)

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

    def _store_key(self, key: tuple[str, str]) -> str:
        namespace, normalized_query = key
        return f"query_embedding:{namespace}:{normalized_query}"

    def _load_from_store(self, key: tuple[str, str]) -> tuple[float, ...] | None:
        if self._store is None:
            return None
        try:
            value = self._store.get(self._store_key(key))
        except Exception:
            logger.warning("query_embedding_cache_store_get_failed", exc_info=True)
            return None
        if value is None:
            return None
        try:
            return tuple(float(item) for item in value)
        except (TypeError, ValueError):
            logger.warning("query_embedding_cache_store_value_invalid")
            return None

    def _save_to_store(self, key: tuple[str, str], embedding: tuple[float, ...]) -> None:
        if self._store is None:
            return
        try:
            self._store.set(self._store_key(key), list(embedding), self._epoch)
        except Exception:
            logger.warning("query_embedding_cache_store_set_failed", exc_info=True)


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
