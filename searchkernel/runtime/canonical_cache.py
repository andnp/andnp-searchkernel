"""Canonical bounded caches and stable keys for record-oriented search."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import TypeVar

from searchkernel.domain import RecordIdentity
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


@dataclass(frozen=True, slots=True)
class BoundedCacheMetrics:
    """Counters for a bounded cache."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0


class _BoundedCache[ValueT]:
    def __init__(
        self,
        *,
        max_entries: int,
        ttl_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be > 0")
        self._max_entries = max_entries
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._entries: OrderedDict[object, tuple[float, ValueT]] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    @property
    def metrics(self) -> BoundedCacheMetrics:
        return BoundedCacheMetrics(self._hits, self._misses, self._evictions)

    def get(self, key: object) -> ValueT | None:
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
        return copy.deepcopy(value)

    def set(self, key: object, value: ValueT) -> None:
        self._entries[key] = (self._clock() + self._ttl_seconds, copy.deepcopy(value))
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
        )

    @property
    def metrics(self) -> BoundedCacheMetrics:
        return self._cache.metrics

    def get(self, key: CandidateCacheKey) -> ValueT | None:
        return self._cache.get(key)

    def set(self, key: CandidateCacheKey, value: ValueT) -> None:
        self._cache.set(key, value)


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

    @property
    def metrics(self) -> BoundedCacheMetrics:
        return self._cache.metrics

    def get(self, key: HydrationCacheKey) -> ValueT | None:
        _hit, value = self.lookup(key)
        return value

    def lookup(self, key: HydrationCacheKey) -> tuple[bool, ValueT | None]:
        missing_until = self._missing.get(key)
        if missing_until is not None:
            if missing_until > self._clock():
                self._missing.move_to_end(key)
                return True, None
            self._missing.pop(key, None)
        value = self._cache.get(key)
        return value is not None, value

    def set(self, key: HydrationCacheKey, value: ValueT | None) -> None:
        if value is None:
            self._missing[key] = self._clock() + self._missing_ttl_seconds
            self._missing.move_to_end(key)
            while len(self._missing) > self._cache._max_entries:
                self._missing.popitem(last=False)
            return
        self._cache.set(key, value)


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
