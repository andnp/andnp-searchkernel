import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Event, Lock, Thread

import pytest

from searchkernel.runtime import (
    ValidatedCacheEntry,
    ValidatedCacheValue,
    ValidatedReadThroughCache,
)


@dataclass
class _Clock:
    value: float = 0.0

    def __call__(self) -> float:
        return self.value


class _MemoryStore:
    def __init__(self) -> None:
        self.entries: dict[str, ValidatedCacheEntry[str, str]] = {}
        self._lock = Lock()

    def get(self, key: str) -> ValidatedCacheEntry[str, str] | None:
        with self._lock:
            return self.entries.get(key)

    def set(self, key: str, entry: ValidatedCacheEntry[str, str]) -> None:
        with self._lock:
            self.entries[key] = entry


class _ListStore:
    def __init__(self) -> None:
        self.entries: dict[str, ValidatedCacheEntry[list[str], str]] = {}
        self._lock = Lock()

    def get(self, key: str) -> ValidatedCacheEntry[list[str], str] | None:
        with self._lock:
            return self.entries.get(key)

    def set(self, key: str, entry: ValidatedCacheEntry[list[str], str]) -> None:
        with self._lock:
            self.entries[key] = entry


class _CancellationBarrierStore(_MemoryStore):
    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        super().__init__()
        self._loop = loop
        self.started = asyncio.Event()
        self.cancel_requested = asyncio.Event()
        self.ready_to_publish = asyncio.Event()
        self.allow_publish = Event()
        self.allow_finish = Event()
        self.finished = asyncio.Event()
        self.cancel_task: asyncio.Task[object] | None = None

    def set(self, key: str, entry: ValidatedCacheEntry[str, str]) -> None:
        self._loop.call_soon_threadsafe(self.started.set)
        self._loop.call_soon_threadsafe(self._cancel_leader)
        assert self.allow_publish.wait(timeout=5.0)
        self._loop.call_soon_threadsafe(self.ready_to_publish.set)
        assert self.allow_finish.wait(timeout=5.0)
        super().set(key, entry)
        self._loop.call_soon_threadsafe(self.finished.set)

    def _cancel_leader(self) -> None:
        assert self.cancel_task is not None
        self.cancel_task.cancel()
        self.cancel_requested.set()


class _ObjectStore:
    def __init__(self) -> None:
        self.entries: dict[str, ValidatedCacheEntry[object, str]] = {}

    def get(self, key: str) -> ValidatedCacheEntry[object, str] | None:
        return self.entries.get(key)

    def set(self, key: str, entry: ValidatedCacheEntry[object, str]) -> None:
        self.entries[key] = entry


def test_clone_hook_replaces_default_deepcopy_on_a_load() -> None:
    sentinel = object()
    cache = ValidatedReadThroughCache(_ObjectStore(), clone=lambda _value: sentinel)

    result = cache.get_or_load(
        "key",
        validate=lambda _key, _token: True,
        load=lambda: ValidatedCacheValue(object(), "v1"),
    )

    assert result is sentinel


def test_validated_cache_refreshes_after_ttl_and_defends_values() -> None:
    clock = _Clock()
    store = _MemoryStore()
    cache = ValidatedReadThroughCache(store, ttl_seconds=5.0, clock=clock)
    current_token = "v1"
    calls = 0

    def validate(_key: str, token: str) -> bool:
        return token == current_token

    def load() -> ValidatedCacheValue[str, str]:
        nonlocal calls
        calls += 1
        return ValidatedCacheValue(f"value-{calls}", current_token)

    assert cache.get_or_load("key", validate=validate, load=load) == "value-1"
    assert cache.get_or_load("key", validate=validate, load=load) == "value-1"
    assert cache.metrics.fresh_hits == 1
    assert calls == 1

    clock.value = 5.0
    assert cache.get_or_load("key", validate=validate, load=load) == "value-2"
    assert cache.metrics.misses == 2
    assert cache.metrics.writes == 2
    assert calls == 2


def test_validation_mismatch_refreshes_and_replaces_cached_token() -> None:
    clock = _Clock()
    store = _MemoryStore()
    store.set(
        "key",
        ValidatedCacheEntry(value="old", validation_token="v1", cached_at=0.0),
    )
    cache = ValidatedReadThroughCache(store, ttl_seconds=30.0, clock=clock)

    assert cache.get_or_load(
        "key",
        validate=lambda _key, token: token == "v2",
        load=lambda: ValidatedCacheValue("new", "v2"),
    ) == "new"
    assert store.get("key") == ValidatedCacheEntry(
        value="new", validation_token="v2", cached_at=0.0
    )
    assert cache.metrics.validation_mismatches == 1


def test_authoritative_failure_returns_stale_value() -> None:
    clock = _Clock()
    store = _MemoryStore()
    cache = ValidatedReadThroughCache(store, ttl_seconds=1.0, clock=clock)

    assert cache.get_or_load(
        "key",
        validate=lambda _key, _token: True,
        load=lambda: ValidatedCacheValue("cached", "v1"),
    ) == "cached"
    clock.value = 2.0

    def fail() -> ValidatedCacheValue[str, str]:
        raise TimeoutError("authoritative read failed")

    assert cache.get_or_load("key", validate=lambda _key, _token: True, load=fail) == "cached"
    assert cache.metrics.load_failures == 1
    assert cache.metrics.stale_fallbacks == 1


def test_sync_misses_are_single_flight() -> None:
    store = _MemoryStore()
    cache = ValidatedReadThroughCache(store)
    started = Event()
    release = Event()
    calls = 0
    results: list[str] = []

    def load() -> ValidatedCacheValue[str, str]:
        nonlocal calls
        calls += 1
        started.set()
        assert release.wait(timeout=5.0)
        return ValidatedCacheValue("shared", "v1")

    def read() -> None:
        results.append(
            cache.get_or_load(
                "key", validate=lambda _key, _token: True, load=load
            )
        )

    first = Thread(target=read)
    second = Thread(target=read)
    first.start()
    assert started.wait(timeout=5.0)
    second.start()
    release.set()
    first.join(timeout=5.0)
    second.join(timeout=5.0)

    assert results == ["shared", "shared"]
    assert calls == 1
    assert cache.metrics.coalesced_waiters == 1
    assert cache.metrics.loads == 1


@pytest.mark.asyncio
async def test_async_misses_are_single_flight_and_support_async_validation() -> None:
    store = _MemoryStore()
    cache = ValidatedReadThroughCache(store)
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def validate(_key: str, token: str) -> bool:
        return token == "v1"

    async def load() -> ValidatedCacheValue[str, str]:
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return ValidatedCacheValue("shared", "v1")

    first = asyncio.create_task(
        cache.async_get_or_load("key", validate=validate, load=load)
    )
    await started.wait()
    second = asyncio.create_task(
        cache.async_get_or_load("key", validate=validate, load=load)
    )

    async def wait_for_follower() -> None:
        while cache.metrics.coalesced_waiters == 0:
            await asyncio.sleep(0)

    await asyncio.wait_for(wait_for_follower(), timeout=1.0)
    release.set()

    assert await asyncio.gather(first, second) == ["shared", "shared"]
    assert calls == 1
    assert cache.metrics.coalesced_waiters == 1


@pytest.mark.asyncio
async def test_cancelled_async_leader_drains_started_background_write() -> None:
    """
    Propagate cancellation only after an already-started cache write finishes.

    The caller must not observe cancellation while the worker can still publish
    a cache entry in the background.
    """
    loop = asyncio.get_running_loop()
    store = _CancellationBarrierStore(loop)
    cache = ValidatedReadThroughCache(store)
    task = asyncio.create_task(
        cache.async_get_or_load(
            "key",
            validate=lambda _key, _token: True,
            load=lambda: ValidatedCacheValue("value", "v1"),
        )
    )
    store.cancel_task = task

    await store.started.wait()
    await store.cancel_requested.wait()
    store.allow_publish.set()
    await store.ready_to_publish.wait()
    assert not task.done()
    store.allow_finish.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert store.finished.is_set()
    assert store.get("key") is not None


@pytest.mark.asyncio
async def test_async_followers_do_not_occupy_the_default_executor() -> None:
    """Async single-flight followers leave executor workers available."""
    loop = asyncio.get_running_loop()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
    cache = ValidatedReadThroughCache(_MemoryStore())
    started = Event()
    release = Event()
    results: list[str] = []

    def load() -> ValidatedCacheValue[str, str]:
        started.set()
        assert release.wait(timeout=5.0)
        return ValidatedCacheValue("shared", "v1")

    def read() -> None:
        results.append(
            cache.get_or_load(
                "key", validate=lambda _key, _token: True, load=load
            )
        )

    leader = Thread(target=read)
    leader.start()

    while not started.is_set():
        await asyncio.sleep(0)

    follower = asyncio.create_task(
        cache.async_get_or_load(
            "key", validate=lambda _key, _token: True, load=load
        )
    )

    async def wait_for_follower() -> None:
        while cache.metrics.coalesced_waiters == 0:
            await asyncio.sleep(0)

    await asyncio.wait_for(wait_for_follower(), timeout=1.0)
    probe_started = Event()
    probe = asyncio.create_task(asyncio.to_thread(probe_started.set))

    async def wait_for_probe() -> None:
        while not probe_started.is_set():
            await asyncio.sleep(0)

    try:
        await asyncio.wait_for(wait_for_probe(), timeout=1.0)
    finally:
        release.set()
        leader.join(timeout=5.0)
        await asyncio.gather(follower, probe)

    assert probe_started.is_set()
    assert results == ["shared"]


def test_mutating_a_fresh_hit_value_does_not_corrupt_the_cache() -> None:
    store = _ListStore()
    cache = ValidatedReadThroughCache(store)

    def load() -> ValidatedCacheValue[list[str], str]:
        return ValidatedCacheValue(["a"], "v1")

    validate = lambda _key, _token: True

    cache.get_or_load("key", validate=validate, load=load)
    fresh_hit = cache.get_or_load("key", validate=validate, load=load)
    fresh_hit.append("mutated-by-caller")

    unaffected = cache.get_or_load("key", validate=validate, load=load)
    assert unaffected == ["a"]


def test_mutating_a_freshly_loaded_value_does_not_corrupt_the_cache() -> None:
    store = _ListStore()
    cache = ValidatedReadThroughCache(store)
    validate = lambda _key, _token: True

    loaded = cache.get_or_load(
        "key", validate=validate, load=lambda: ValidatedCacheValue(["a"], "v1")
    )
    loaded.append("mutated-by-caller")

    unaffected = cache.get_or_load(
        "key", validate=validate, load=lambda: ValidatedCacheValue(["a"], "v1")
    )
    assert unaffected == ["a"]


def test_mutating_a_stale_fallback_value_does_not_corrupt_the_cache() -> None:
    clock = _Clock()
    store = _ListStore()
    cache = ValidatedReadThroughCache(store, ttl_seconds=1.0, clock=clock)
    validate = lambda _key, _token: True

    def fail() -> ValidatedCacheValue[list[str], str]:
        raise TimeoutError("authoritative read failed")

    cache.get_or_load(
        "key", validate=validate, load=lambda: ValidatedCacheValue(["cached"], "v1")
    )
    clock.value = 2.0

    fallback = cache.get_or_load("key", validate=validate, load=fail)
    fallback.append("mutated-by-caller")

    unaffected = cache.get_or_load("key", validate=validate, load=fail)
    assert unaffected == ["cached"]


def test_coalesced_readers_receive_independent_objects_not_a_shared_one() -> None:
    store = _ListStore()
    cache = ValidatedReadThroughCache(store)
    started = Event()
    release = Event()
    results: list[list[str]] = []

    def load() -> ValidatedCacheValue[list[str], str]:
        started.set()
        assert release.wait(timeout=5.0)
        return ValidatedCacheValue(["shared"], "v1")

    def read() -> None:
        results.append(
            cache.get_or_load(
                "key", validate=lambda _key, _token: True, load=load
            )
        )

    leader = Thread(target=read)
    follower = Thread(target=read)
    leader.start()
    assert started.wait(timeout=5.0)
    follower.start()
    release.set()
    leader.join(timeout=5.0)
    follower.join(timeout=5.0)

    assert len(results) == 2
    assert results[0] == ["shared"]
    assert results[1] == ["shared"]
    assert results[0] is not results[1]

    results[0].append("mutated-by-one-reader")
    assert results[1] == ["shared"]
