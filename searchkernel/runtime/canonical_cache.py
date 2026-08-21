"""Canonical bounded caches and stable keys for record-oriented search."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import math
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Mapping
from concurrent.futures import Future
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from threading import Lock, RLock
from typing import Protocol, TypeGuard, TypeVar, overload, runtime_checkable

from searchkernel.domain import RecordIdentity, SearchResultProvenance
from searchkernel.ports.epochs import SearchEpochs


class UnstableCacheKey(ValueError):
    """Raised when a value cannot be represented as a stable cache key."""


def normalize_cache_query(query: str) -> str:
    """Normalize only outer and repeated whitespace."""
    return " ".join(query.strip().split())


def stable_json(value: object) -> str:
    """Serialize supported cache-key values without arbitrary ``repr`` calls."""
    return json.dumps(
        _stable_value(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def fingerprint(value: object) -> str:
    """Return a compact fingerprint for a stable configuration value."""
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class CandidateCacheKey:
    """Stable identity for an unhydrated candidate result."""

    query: str
    filters: str
    requested_limit: int
    acquisition_limit: int
    adaptive_limit: int | None
    routing_fingerprint: str
    encoder_namespace: str | None
    keyword_epoch: int
    vector_epoch: int
    graph_epoch: int
    policy_version: str | None

    @classmethod
    def build(
        cls,
        *,
        query: str,
        filters: object,
        requested_limit: int,
        acquisition_limit: int,
        adaptive_limit: int | None,
        routing_fingerprint: str,
        encoder_namespace: str | None,
        epochs: SearchEpochs,
        policy_version: str | None,
    ) -> CandidateCacheKey:
        if requested_limit < 1 or acquisition_limit < 1:
            raise ValueError("cache limits must be positive")
        if not routing_fingerprint:
            raise ValueError("routing_fingerprint must not be empty")
        return cls(
            query=normalize_cache_query(query),
            filters=stable_json(filters),
            requested_limit=requested_limit,
            acquisition_limit=acquisition_limit,
            adaptive_limit=adaptive_limit,
            routing_fingerprint=routing_fingerprint,
            encoder_namespace=encoder_namespace,
            keyword_epoch=epochs.keyword,
            vector_epoch=epochs.vector,
            graph_epoch=epochs.graph,
            policy_version=policy_version,
        )


@dataclass(frozen=True, slots=True)
class HydrationCacheKey:
    """Stable identity for one versioned, optionally authorized record."""

    identity: str
    record_version: str
    policy_version: str | None = None

    @classmethod
    def build(
        cls,
        identity: RecordIdentity,
        *,
        record_version: object,
        policy_version: str | None = None,
    ) -> HydrationCacheKey:
        return cls(
            identity=identity.storage_key,
            record_version=stable_json(record_version),
            policy_version=policy_version,
        )


ValueT = TypeVar("ValueT")
KeyT = TypeVar("KeyT")


@runtime_checkable
class _SearchCandidateContract(Protocol):
    identity: RecordIdentity
    score: float
    provenance: SearchResultProvenance
    priority: int


def _is_search_candidate(
    value: object,
) -> TypeGuard[_SearchCandidateContract]:
    value_type = type(value)
    return (
        isinstance(value, _SearchCandidateContract)
        and value_type.__module__ == "searchkernel.search.record_pipeline"
        and value_type.__qualname__ == "RecordSearchCandidate"
    )


@overload
def _clone_candidate_cache_value(
    value: tuple[_SearchCandidateContract, ...],
) -> tuple[_SearchCandidateContract, ...]: ...


@overload
def _clone_candidate_cache_value[T](value: T) -> T: ...


def _clone_candidate_cache_value(value: object) -> object:
    if isinstance(value, tuple) and value and all(
        _is_search_candidate(candidate) for candidate in value
    ):
        return tuple(
            type(candidate)(
                identity=candidate.identity,
                score=candidate.score,
                provenance=candidate.provenance.clone(),
                priority=candidate.priority,
            )
            for candidate in value
        )
    return copy.deepcopy(value)


@dataclass(frozen=True, slots=True)
class BoundedCacheMetrics:
    """Counters for a bounded cache."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    coalesced_waiters: int = 0


@dataclass(slots=True)
class _InFlight[ValueT]:
    completed: Future[ValueT] = field(default_factory=Future)


class _BoundedCache[ValueT]:
    def __init__(
        self,
        *,
        max_entries: int,
        ttl_seconds: float,
        clock: Callable[[], float] = time.monotonic,
        clone: Callable[[ValueT], ValueT] = copy.deepcopy,
    ) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be > 0")
        self._max_entries = max_entries
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._clone = clone
        self._entries: OrderedDict[object, tuple[float, ValueT]] = OrderedDict()
        self._lock = Lock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._coalesced_waiters = 0

    @property
    def metrics(self) -> BoundedCacheMetrics:
        with self._lock:
            return BoundedCacheMetrics(
                self._hits,
                self._misses,
                self._evictions,
                self._coalesced_waiters,
            )

    def get(self, key: object) -> ValueT | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._misses += 1
                return None
            expires_at, value = entry
            if expires_at <= self._clock():
                self._entries.pop(key, None)
                self._evictions += 1
                self._misses += 1
                return None
            self._entries.move_to_end(key)
            self._hits += 1
            return self._clone(value)

    def set(self, key: object, value: ValueT) -> None:
        with self._lock:
            self._entries[key] = (
                self._clock() + self._ttl_seconds,
                self._clone(value),
            )
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)
                self._evictions += 1


class CandidateResultCache[ValueT]:
    """Small LRU/TTL cache for unhydrated fused candidates."""

    def __init__(
        self,
        *,
        max_entries: int = 128,
        ttl_seconds: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._cache = _BoundedCache[ValueT](
            max_entries=max_entries,
            ttl_seconds=ttl_seconds,
            clock=clock,
            clone=_clone_candidate_cache_value,
        )
        self._inflight: dict[CandidateCacheKey, _InFlight[ValueT]] = {}
        self._flight_lock = Lock()

    @property
    def metrics(self) -> BoundedCacheMetrics:
        return self._cache.metrics

    def get(self, key: CandidateCacheKey) -> ValueT | None:
        return self._cache.get(key)

    def set(self, key: CandidateCacheKey, value: ValueT) -> None:
        self._cache.set(key, value)
        self._complete_inflight(key, value)

    def fail(self, key: CandidateCacheKey, error: BaseException) -> None:
        """Release waiters when the owner cannot populate the cache."""
        with self._flight_lock:
            inflight = self._inflight.pop(key, None)
        if inflight is not None:
            _complete_failure(inflight, error)

    async def async_wait_for_miss(
        self, key: CandidateCacheKey
    ) -> tuple[bool, ValueT | None]:
        """Claim a miss or await the caller already computing this key."""
        inflight, leader = self._start_or_join(key)
        if leader:
            _watch_single_flight_owner(
                self._owner_failed,
                key,
                inflight,
            )
            return True, None
        return False, await _wait_for_inflight(
            inflight, clone=_clone_candidate_cache_value
        )

    async def async_get_or_compute(
        self,
        key: CandidateCacheKey,
        compute: Callable[[], ValueT | Awaitable[ValueT]],
    ) -> ValueT:
        """Read through the cache while coalescing concurrent misses."""
        cached = self.get(key)
        if cached is not None:
            return cached
        inflight, leader = self._start_or_join(key)
        if not leader:
            return await _wait_for_inflight(
                inflight, clone=_clone_candidate_cache_value
            )
        try:
            value = await _maybe_await(compute())
            self.set(key, value)
            return _clone_candidate_cache_value(value)
        except BaseException as error:
            self._fail_inflight(key, inflight, error)
            raise

    def _complete_inflight(self, key: CandidateCacheKey, value: ValueT) -> None:
        with self._flight_lock:
            inflight = self._inflight.pop(key, None)
        if inflight is not None:
            _complete_success(inflight, value, _clone_candidate_cache_value)

    def _fail_inflight(
        self,
        key: CandidateCacheKey,
        inflight: _InFlight[ValueT],
        error: BaseException,
    ) -> None:
        with self._flight_lock:
            if self._inflight.get(key) is not inflight:
                return
            self._inflight.pop(key)
        _complete_failure(inflight, error)

    def _start_or_join(
        self, key: CandidateCacheKey
    ) -> tuple[_InFlight[ValueT], bool]:
        with self._flight_lock:
            inflight = self._inflight.get(key)
            if inflight is not None:
                with self._cache._lock:
                    self._cache._coalesced_waiters += 1
                    self._cache._misses -= 1
                return inflight, False
            inflight = _InFlight()
            self._inflight[key] = inflight
            return inflight, True

    def _owner_failed(
        self,
        key: CandidateCacheKey,
        inflight: _InFlight[ValueT],
        error: BaseException,
    ) -> None:
        self._fail_inflight(key, inflight, error)


class HydrationCache[ValueT]:
    """Optional versioned hydration cache with short-lived missing entries."""

    def __init__(
        self,
        *,
        max_entries: int = 64,
        ttl_seconds: float = 60.0,
        missing_ttl_seconds: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._cache = _BoundedCache[ValueT](
            max_entries=max_entries,
            ttl_seconds=ttl_seconds,
            clock=clock,
        )
        if missing_ttl_seconds <= 0:
            raise ValueError("missing_ttl_seconds must be > 0")
        self._missing_ttl_seconds = missing_ttl_seconds
        self._clock = clock
        self._missing: OrderedDict[HydrationCacheKey, float] = OrderedDict()
        self._inflight: dict[HydrationCacheKey, _InFlight[ValueT | None]] = {}
        self._flight_lock = RLock()

    @property
    def metrics(self) -> BoundedCacheMetrics:
        return self._cache.metrics

    def get(self, key: HydrationCacheKey) -> ValueT | None:
        _hit, value = self.lookup(key)
        return value

    def lookup(self, key: HydrationCacheKey) -> tuple[bool, ValueT | None]:
        with self._flight_lock:
            missing_until = self._missing.get(key)
            if missing_until is not None:
                if missing_until > self._clock():
                    self._missing.move_to_end(key)
                    return True, None
                self._missing.pop(key, None)
            value = self._cache.get(key)
            return value is not None, value

    def set(self, key: HydrationCacheKey, value: ValueT | None) -> None:
        with self._flight_lock:
            if value is None:
                self._missing[key] = self._clock() + self._missing_ttl_seconds
                self._missing.move_to_end(key)
                while len(self._missing) > self._cache._max_entries:
                    self._missing.popitem(last=False)
            else:
                self._cache.set(key, value)
            inflight = self._inflight.pop(key, None)
        if inflight is not None:
            _complete_success(inflight, value)

    def fail(self, key: HydrationCacheKey, error: BaseException) -> None:
        """Release waiters when the owner cannot populate the cache."""
        with self._flight_lock:
            inflight = self._inflight.pop(key, None)
        if inflight is not None:
            _complete_failure(inflight, error)

    async def async_wait_for_miss(
        self, key: HydrationCacheKey
    ) -> tuple[bool, ValueT | None]:
        """Claim a miss or await the caller already computing this key."""
        inflight, leader = self._start_or_join(key)
        if leader:
            _watch_single_flight_owner(
                self._owner_failed,
                key,
                inflight,
            )
            return True, None
        return False, await _wait_for_inflight(inflight)

    async def async_get_or_compute(
        self,
        key: HydrationCacheKey,
        compute: Callable[[], ValueT | None | Awaitable[ValueT | None]],
    ) -> ValueT | None:
        """Read through the cache while coalescing concurrent misses."""
        hit, value = self.lookup(key)
        if hit:
            return value
        inflight, leader = self._start_or_join(key)
        if not leader:
            return await _wait_for_inflight(inflight)
        try:
            value = await _maybe_await(compute())
            self.set(key, value)
            return copy.deepcopy(value)
        except BaseException as error:
            self._fail_inflight(key, inflight, error)
            raise

    def _fail_inflight(
        self,
        key: HydrationCacheKey,
        inflight: _InFlight[ValueT | None],
        error: BaseException,
    ) -> None:
        with self._flight_lock:
            if self._inflight.get(key) is not inflight:
                return
            self._inflight.pop(key)
        _complete_failure(inflight, error)

    def _start_or_join(
        self, key: HydrationCacheKey
    ) -> tuple[_InFlight[ValueT | None], bool]:
        with self._flight_lock:
            inflight = self._inflight.get(key)
            if inflight is not None:
                with self._cache._lock:
                    self._cache._coalesced_waiters += 1
                    self._cache._misses -= 1
                return inflight, False
            inflight = _InFlight()
            self._inflight[key] = inflight
            return inflight, True

    def _owner_failed(
        self,
        key: HydrationCacheKey,
        inflight: _InFlight[ValueT | None],
        error: BaseException,
    ) -> None:
        self._fail_inflight(key, inflight, error)


def _complete_success[T](
    inflight: _InFlight[T],
    value: T,
    clone: Callable[[T], T] = copy.deepcopy,
) -> None:
    inflight.completed.set_result(clone(value))


def _complete_failure[T](inflight: _InFlight[T], error: BaseException) -> None:
    inflight.completed.set_exception(error)


def _watch_single_flight_owner[KeyT, ValueT](
    fail: Callable[[KeyT, _InFlight[ValueT], BaseException], None],
    key: KeyT,
    inflight: _InFlight[ValueT],
) -> None:
    owner = asyncio.current_task()
    if owner is None:
        return

    def owner_done(task: asyncio.Task[object]) -> None:
        try:
            error = task.exception()
        except asyncio.CancelledError as task_error:
            error = task_error
        if error is not None:
            fail(key, inflight, error)

    owner.add_done_callback(owner_done)


async def _wait_for_inflight[T](
    inflight: _InFlight[T],
    clone: Callable[[T], T] = copy.deepcopy,
) -> T:
    await asyncio.shield(asyncio.wrap_future(inflight.completed))
    return clone(inflight.completed.result())


async def _maybe_await[T](value: T | Awaitable[T]) -> T:
    if asyncio.iscoroutine(value) or isinstance(value, Awaitable):
        return await value
    return value


def _stable_value(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise UnstableCacheKey("non-finite floats are not stable cache values")
        return value
    if isinstance(value, Enum):
        return _stable_value(value.value)
    if isinstance(value, RecordIdentity):
        return {
            "workspace_id": value.workspace_id,
            "source_id": value.source_id,
            "source_kind": value.source_kind,
        }
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise UnstableCacheKey("cache mappings require string keys")
        return {
            key: _stable_value(item)
            for key, item in sorted(value.items())
        }
    if isinstance(value, (list, tuple)):
        return [_stable_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        values = [_stable_value(item) for item in value]
        return sorted(values, key=lambda item: stable_json(item))
    raise UnstableCacheKey(
        f"unsupported cache-key value type: {type(value).__name__}"
    )
