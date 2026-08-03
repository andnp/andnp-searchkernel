import threading
from dataclasses import dataclass, field
from typing import Any

import pytest

from searchkernel.runtime import QueryEmbeddingCache


@dataclass
class _FakeCacheStore:
    entries: dict[str, tuple[Any, int]] = field(default_factory=dict)

    def get(self, key: str) -> Any | None:
        entry = self.entries.get(key)
        return None if entry is None else entry[0]

    def set(self, key: str, value: Any, epoch: int) -> None:
        self.entries[key] = (value, epoch)

    def invalidate_epoch(self, epoch: int) -> None:
        self.entries = {
            key: entry for key, entry in self.entries.items() if entry[1] > epoch
        }


class _FailingCacheStore:
    def get(self, key: str) -> Any | None:
        raise RuntimeError("store unavailable")

    def set(self, key: str, value: Any, epoch: int) -> None:
        raise RuntimeError("store unavailable")

    def invalidate_epoch(self, epoch: int) -> None:
        raise RuntimeError("store unavailable")


@dataclass
class _Clock:
    value: float = 0.0

    def __call__(self) -> float:
        return self.value


def test_cache_keys_by_model_and_query_and_returns_copies() -> None:
    cache = QueryEmbeddingCache()
    calls = 0

    def compute() -> list[float]:
        nonlocal calls
        calls += 1
        return [1.0, 2.0]

    first = cache.get_or_compute(model_name="alpha", query="same", compute=compute)
    first.append(3.0)
    second = cache.get_or_compute(model_name="alpha", query="same", compute=compute)
    third = cache.get_or_compute(model_name="beta", query="same", compute=compute)

    assert second == [1.0, 2.0]
    assert third == [1.0, 2.0]
    assert calls == 2


def test_cache_expires_entries() -> None:
    clock = _Clock()
    cache = QueryEmbeddingCache(ttl_seconds=5.0, clock=clock)
    calls = 0

    def compute() -> list[float]:
        nonlocal calls
        calls += 1
        return [float(calls)]

    assert cache.get_or_compute(model_name="model", query="query", compute=compute) == [1.0]
    clock.value = 5.0
    assert cache.get_or_compute(model_name="model", query="query", compute=compute) == [2.0]


def test_cache_evicts_least_recently_used_entries() -> None:
    cache = QueryEmbeddingCache(max_entries=2)
    calls: dict[str, int] = {}

    def compute_for(query: str) -> list[float]:
        calls[query] = calls.get(query, 0) + 1
        return [float(calls[query])]

    cache.get_or_compute(model_name="model", query="a", compute=lambda: compute_for("a"))
    cache.get_or_compute(model_name="model", query="b", compute=lambda: compute_for("b"))
    cache.get_or_compute(model_name="model", query="a", compute=lambda: compute_for("a"))
    cache.get_or_compute(model_name="model", query="c", compute=lambda: compute_for("c"))
    cache.get_or_compute(model_name="model", query="b", compute=lambda: compute_for("b"))

    assert calls == {"a": 1, "b": 2, "c": 1}


def test_cache_coalesces_concurrent_misses() -> None:
    cache = QueryEmbeddingCache()
    started = threading.Event()
    release = threading.Event()
    calls = 0
    results: list[list[float]] = []

    def compute() -> list[float]:
        nonlocal calls
        calls += 1
        started.set()
        assert release.wait(timeout=5.0)
        return [1.0]

    def load() -> None:
        results.append(cache.get_or_compute(model_name="model", query="query", compute=compute))

    leader = threading.Thread(target=load)
    follower = threading.Thread(target=load)
    leader.start()
    assert started.wait(timeout=5.0)
    follower.start()
    release.set()
    leader.join(timeout=5.0)
    follower.join(timeout=5.0)

    assert sorted(results) == [[1.0], [1.0]]
    assert calls == 1


def test_cache_does_not_cache_failures() -> None:
    cache = QueryEmbeddingCache()
    calls = 0

    def fail() -> list[float]:
        nonlocal calls
        calls += 1
        raise RuntimeError("embedding failed")

    with pytest.raises(RuntimeError, match="embedding failed"):
        cache.get_or_compute(model_name="model", query="query", compute=fail)
    with pytest.raises(RuntimeError, match="embedding failed"):
        cache.get_or_compute(model_name="model", query="query", compute=fail)
    assert calls == 2


@pytest.mark.parametrize(
    ("ttl_seconds", "max_entries", "message"),
    [
        (0.0, 128, "ttl_seconds"),
        (300.0, 0, "max_entries"),
    ],
)
def test_cache_validates_bounds(ttl_seconds: float, max_entries: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        QueryEmbeddingCache(ttl_seconds=ttl_seconds, max_entries=max_entries)


def test_cache_writes_through_to_store() -> None:
    store = _FakeCacheStore()
    cache = QueryEmbeddingCache(store=store, epoch=3)

    result = cache.get_or_compute(model_name="model", query="query", compute=lambda: [1.0, 2.0])

    assert result == [1.0, 2.0]
    assert store.entries["query_embedding:model:query"] == ([1.0, 2.0], 3)


def test_cache_reads_through_store_on_a_fresh_instance() -> None:
    store = _FakeCacheStore()
    warm_cache = QueryEmbeddingCache(store=store)
    warm_cache.get_or_compute(model_name="model", query="query", compute=lambda: [9.0])

    cold_cache = QueryEmbeddingCache(store=store)
    calls = 0

    def compute() -> list[float]:
        nonlocal calls
        calls += 1
        return [0.0]

    result = cold_cache.get_or_compute(model_name="model", query="query", compute=compute)

    assert result == [9.0]
    assert calls == 0


def test_cache_tolerates_store_get_and_set_failures() -> None:
    cache = QueryEmbeddingCache(store=_FailingCacheStore())
    calls = 0

    def compute() -> list[float]:
        nonlocal calls
        calls += 1
        return [float(calls)]

    assert cache.get_or_compute(model_name="model", query="query", compute=compute) == [1.0]
    assert cache.get_or_compute(model_name="model", query="query", compute=compute) == [1.0]
    assert calls == 1


def test_cache_invalidate_epoch_clears_memory_and_delegates_to_store() -> None:
    store = _FakeCacheStore()
    cache = QueryEmbeddingCache(store=store, epoch=5)
    cache.get_or_compute(model_name="model", query="query", compute=lambda: [1.0])

    cache.invalidate_epoch(5)

    assert store.entries == {}
    calls = 0

    def compute() -> list[float]:
        nonlocal calls
        calls += 1
        return [2.0]

    assert cache.get_or_compute(model_name="model", query="query", compute=compute) == [2.0]
    assert calls == 1


def test_cache_invalidate_epoch_tolerates_store_failure() -> None:
    cache = QueryEmbeddingCache(store=_FailingCacheStore())
    cache.get_or_compute(model_name="model", query="query", compute=lambda: [1.0])

    cache.invalidate_epoch(0)
