"""Validated TTL read-through caching with single-flight coordination."""

from __future__ import annotations

import asyncio
import copy
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from threading import Event, Lock
from typing import Protocol, TypeVar, cast

KeyT = TypeVar("KeyT")
KeyT_contra = TypeVar("KeyT_contra", contravariant=True)
ValueT = TypeVar("ValueT")
TokenT = TypeVar("TokenT")


@dataclass(frozen=True, slots=True)
class ValidatedCacheValue[ValueT, TokenT]:
    """A value returned by an authoritative loader and its validation token."""

    value: ValueT
    validation_token: TokenT


@dataclass(frozen=True, slots=True)
class ValidatedCacheEntry[ValueT, TokenT]:
    """A stored value, token, and timestamp used for TTL evaluation."""

    value: ValueT
    validation_token: TokenT
    cached_at: float


class ValidatedCacheStore(Protocol[KeyT_contra, ValueT, TokenT]):
    """Storage boundary for validated cache entries.

    Storage is deliberately a small injected boundary.  The cache does not
    decide which backend is authoritative or enqueue writes for later replay.
    """

    def get(self, key: KeyT_contra) -> ValidatedCacheEntry[ValueT, TokenT] | None:
        """Return the stored entry, including entries that have passed their TTL."""
        ...

    def set(
        self, key: KeyT_contra, entry: ValidatedCacheEntry[ValueT, TokenT]
    ) -> None:
        """Persist a successfully loaded entry."""
        ...


@dataclass(frozen=True, slots=True)
class ValidatedReadThroughCacheMetrics:
    """Counters and timing for one validated read-through cache."""

    requests: int = 0
    fresh_hits: int = 0
    misses: int = 0
    validation_mismatches: int = 0
    validation_failures: int = 0
    stale_fallbacks: int = 0
    coalesced_waiters: int = 0
    loads: int = 0
    load_failures: int = 0
    writes: int = 0
    write_failures: int = 0
    cache_read_failures: int = 0
    load_time_seconds: float = 0.0


_MISSING = object()


@dataclass(slots=True)
class _InFlight[ValueT]:
    completed: Event = field(default_factory=Event)
    value: object = _MISSING
    error: BaseException | None = None
    used_stale_fallback: bool = False


class ValidatedReadThroughCache[KeyT, ValueT, TokenT]:
    """Read-through cache with TTL, validation, and per-key single-flight.

    A cache hit is accepted only when the entry is within ``ttl_seconds`` and
    ``validate`` confirms its token.  A failed authoritative load may return
    the most recent cached value, including one past its TTL.  The loader and
    validator are supplied by the caller so authority and writeback policy
    remain outside this library primitive.

    The same cache instance supports synchronous and asynchronous callers.
    Async followers wait for a synchronous event in a worker thread, keeping
    the event loop free of blocking coordination.
    """

    def __init__(
        self,
        store: ValidatedCacheStore[KeyT, ValueT, TokenT],
        *,
        ttl_seconds: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be > 0")
        self._store = store
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._inflight: dict[KeyT, _InFlight[ValueT]] = {}
        self._lock = Lock()
        self._metrics = ValidatedReadThroughCacheMetrics()

    @property
    def metrics(self) -> ValidatedReadThroughCacheMetrics:
        """Return a consistent snapshot of cache metrics."""
        with self._lock:
            return self._metrics

    def get_or_load(
        self,
        key: KeyT,
        *,
        validate: Callable[[KeyT, TokenT], bool],
        load: Callable[[], ValidatedCacheValue[ValueT, TokenT]],
    ) -> ValueT:
        """Return a validated cached value or load and cache an authoritative value."""
        self._record(requests=1)
        stale_entry, fresh_entry = self._inspect_cached_entry(
            key, self._read_entry(key), validate
        )
        if fresh_entry is not None:
            return self._copy_value(fresh_entry.value)

        self._record(misses=1)
        inflight, is_leader = self._start_or_join(key)
        if not is_leader:
            inflight.completed.wait()
            return self._finish_waiter(inflight)
        return self._load_as_leader(key, inflight, stale_entry, load)

    async def async_get_or_load(
        self,
        key: KeyT,
        *,
        validate: Callable[[KeyT, TokenT], bool | Awaitable[bool]],
        load: Callable[
            [],
            ValidatedCacheValue[ValueT, TokenT]
            | Awaitable[ValidatedCacheValue[ValueT, TokenT]],
        ],
    ) -> ValueT:
        """Return a validated cached value without blocking the event loop."""
        self._record(requests=1)
        entry = await asyncio.to_thread(self._read_entry, key)
        stale_entry, fresh_entry = await self._inspect_cached_entry_async(
            key, entry, validate
        )
        if fresh_entry is not None:
            return self._copy_value(fresh_entry.value)

        self._record(misses=1)
        inflight, is_leader = self._start_or_join(key)
        if not is_leader:
            await asyncio.to_thread(inflight.completed.wait)
            return self._finish_waiter(inflight)
        return await self._load_as_async_leader(key, inflight, stale_entry, load)

    def _read_entry(self, key: KeyT) -> ValidatedCacheEntry[ValueT, TokenT] | None:
        try:
            entry = self._store.get(key)
            if entry is None:
                return None
            if not isinstance(entry, ValidatedCacheEntry):
                raise TypeError("validated cache store returned an invalid entry")
            return copy.deepcopy(entry)
        except Exception:  # noqa: BLE001 - cache reads are best effort
            self._record(cache_read_failures=1)
            return None

    def _inspect_cached_entry(
        self,
        key: KeyT,
        entry: ValidatedCacheEntry[ValueT, TokenT] | None,
        validate: Callable[[KeyT, TokenT], bool],
    ) -> tuple[
        ValidatedCacheEntry[ValueT, TokenT] | None,
        ValidatedCacheEntry[ValueT, TokenT] | None,
    ]:
        if entry is None:
            return None, None
        if self._is_expired(entry):
            return entry, None
        try:
            valid = validate(key, entry.validation_token)
        except Exception:  # noqa: BLE001 - validation belongs to the caller
            self._record(validation_failures=1)
            return entry, None
        if valid:
            self._record(fresh_hits=1)
            return None, entry
        self._record(validation_mismatches=1)
        return entry, None

    async def _inspect_cached_entry_async(
        self,
        key: KeyT,
        entry: ValidatedCacheEntry[ValueT, TokenT] | None,
        validate: Callable[[KeyT, TokenT], bool | Awaitable[bool]],
    ) -> tuple[
        ValidatedCacheEntry[ValueT, TokenT] | None,
        ValidatedCacheEntry[ValueT, TokenT] | None,
    ]:
        if entry is None:
            return None, None
        if self._is_expired(entry):
            return entry, None
        try:
            valid = await _maybe_await(validate(key, entry.validation_token))
        except Exception:  # noqa: BLE001 - validation belongs to the caller
            self._record(validation_failures=1)
            return entry, None
        if valid:
            self._record(fresh_hits=1)
            return None, entry
        self._record(validation_mismatches=1)
        return entry, None

    def _is_expired(self, entry: ValidatedCacheEntry[ValueT, TokenT]) -> bool:
        return self._clock() - entry.cached_at >= self._ttl_seconds

    def _start_or_join(self, key: KeyT) -> tuple[_InFlight[ValueT], bool]:
        with self._lock:
            inflight = self._inflight.get(key)
            if inflight is not None:
                self._metrics = replace(
                    self._metrics,
                    coalesced_waiters=self._metrics.coalesced_waiters + 1,
                )
                return inflight, False
            inflight = _InFlight()
            self._inflight[key] = inflight
            return inflight, True

    def _load_as_leader(
        self,
        key: KeyT,
        inflight: _InFlight[ValueT],
        stale_entry: ValidatedCacheEntry[ValueT, TokenT] | None,
        load: Callable[[], ValidatedCacheValue[ValueT, TokenT]],
    ) -> ValueT:
        started = self._clock()
        self._record(loads=1)
        try:
            loaded = load()
            if not isinstance(loaded, ValidatedCacheValue):
                raise TypeError("validated cache loader returned an invalid value")
            entry = ValidatedCacheEntry(
                value=copy.deepcopy(loaded.value),
                validation_token=copy.deepcopy(loaded.validation_token),
                cached_at=self._clock(),
            )
            self._write_entry(key, entry)
            value = self._copy_value(loaded.value)
            self._complete_success(key, inflight, value)
            return value
        except Exception as error:  # noqa: BLE001 - stale fallback covers load errors
            return self._handle_load_failure(key, inflight, stale_entry, error)
        except BaseException as error:
            self._complete_failure(key, inflight, error)
            raise
        finally:
            self._record(load_time_seconds=self._clock() - started)

    async def _load_as_async_leader(
        self,
        key: KeyT,
        inflight: _InFlight[ValueT],
        stale_entry: ValidatedCacheEntry[ValueT, TokenT] | None,
        load: Callable[
            [],
            ValidatedCacheValue[ValueT, TokenT]
            | Awaitable[ValidatedCacheValue[ValueT, TokenT]],
        ],
    ) -> ValueT:
        started = self._clock()
        self._record(loads=1)
        try:
            loaded = await _maybe_await(load())
            if not isinstance(loaded, ValidatedCacheValue):
                raise TypeError("validated cache loader returned an invalid value")
            entry = ValidatedCacheEntry(
                value=copy.deepcopy(loaded.value),
                validation_token=copy.deepcopy(loaded.validation_token),
                cached_at=self._clock(),
            )
            await asyncio.to_thread(self._write_entry, key, entry)
            value = self._copy_value(loaded.value)
            self._complete_success(key, inflight, value)
            return value
        except Exception as error:  # noqa: BLE001 - stale fallback covers load errors
            return self._handle_load_failure(key, inflight, stale_entry, error)
        except BaseException as error:
            self._complete_failure(key, inflight, error)
            raise
        finally:
            self._record(load_time_seconds=self._clock() - started)

    def _write_entry(
        self, key: KeyT, entry: ValidatedCacheEntry[ValueT, TokenT]
    ) -> None:
        try:
            self._store.set(key, copy.deepcopy(entry))
        except Exception:  # noqa: BLE001 - cache writes are best effort
            self._record(write_failures=1)
        else:
            self._record(writes=1)

    def _handle_load_failure(
        self,
        key: KeyT,
        inflight: _InFlight[ValueT],
        stale_entry: ValidatedCacheEntry[ValueT, TokenT] | None,
        error: Exception,
    ) -> ValueT:
        self._record(load_failures=1)
        if stale_entry is not None:
            value = self._copy_value(stale_entry.value)
            self._complete_success(
                key, inflight, value, used_stale_fallback=True
            )
            self._record(stale_fallbacks=1)
            return value
        self._complete_failure(key, inflight, error)
        raise error

    def _complete_success(
        self,
        key: KeyT,
        inflight: _InFlight[ValueT],
        value: ValueT,
        *,
        used_stale_fallback: bool = False,
    ) -> None:
        with self._lock:
            inflight.value = copy.deepcopy(value)
            inflight.used_stale_fallback = used_stale_fallback
            self._inflight.pop(key, None)
            inflight.completed.set()

    def _complete_failure(
        self, key: KeyT, inflight: _InFlight[ValueT], error: BaseException
    ) -> None:
        with self._lock:
            inflight.error = error
            self._inflight.pop(key, None)
            inflight.completed.set()

    def _finish_waiter(self, inflight: _InFlight[ValueT]) -> ValueT:
        if inflight.value is not _MISSING:
            if inflight.used_stale_fallback:
                self._record(stale_fallbacks=1)
            return self._copy_value(cast(ValueT, inflight.value))
        if inflight.error is not None:
            raise inflight.error
        raise RuntimeError("validated read-through coalescing completed without a result")

    def _record(self, **increments: float) -> None:
        with self._lock:
            values = {
                field: getattr(self._metrics, field)
                for field in self._metrics.__dataclass_fields__
            }
            for field, amount in increments.items():
                values[field] += amount
            self._metrics = ValidatedReadThroughCacheMetrics(**values)

    @staticmethod
    def _copy_value(value: ValueT) -> ValueT:
        return copy.deepcopy(value)


async def _maybe_await[ResultT](value: ResultT | Awaitable[ResultT]) -> ResultT:
    if asyncio.iscoroutine(value) or isinstance(value, Awaitable):
        return await value
    return value
