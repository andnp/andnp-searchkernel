import asyncio
import threading
from dataclasses import dataclass, field
from typing import Any

import pytest

from searchkernel.runtime import QueryEmbeddingCache


@dataclass
class _Store:
    entries: dict[str, tuple[Any, int]] = field(default_factory=dict)
    get_started: threading.Event = field(default_factory=threading.Event)
    set_started: threading.Event = field(default_factory=threading.Event)
    release: threading.Event = field(default_factory=threading.Event)
    allow_release: threading.Event = field(default_factory=threading.Event)
    block_get: bool = False
    block_set: bool = False
    released_before_probe: bool = False

    def get(self, key: str) -> Any | None:
        self.get_started.set()
        if self.block_get and not self.release.wait(timeout=2.0):
            raise AssertionError("controlled store read was not released")
        entry = self.entries.get(key)
        return None if entry is None else entry[0]

    def set(self, key: str, value: Any, epoch: int) -> None:
        self.set_started.set()
        if self.block_set and not self.release.wait(timeout=2.0):
            raise AssertionError("controlled store write was not released")
        self.entries[key] = (value, epoch)

    def invalidate_epoch(self, epoch: int) -> None:
        self.entries = {
            key: entry for key, entry in self.entries.items() if entry[1] > epoch
        }

    def release_after_probe(self) -> None:
        if not self.allow_release.wait(timeout=2.0):
            self.released_before_probe = True
        self.release.set()


@pytest.mark.asyncio
async def test_async_store_load_keeps_event_loop_schedulable() -> None:
    """A blocked persistent read must not prevent another coroutine from running."""
    store = _Store(block_get=True)
    cache = QueryEmbeddingCache(store=store)
    watchdog = threading.Thread(target=store.release_after_probe, daemon=True)
    watchdog.start()

    task = asyncio.create_task(
        cache.async_get_or_compute(
            model_name="model", query="query", compute=lambda: [1.0]
        )
    )
    await asyncio.to_thread(store.get_started.wait, 1.0)
    probe_ran = asyncio.Event()

    async def probe() -> None:
        probe_ran.set()
        store.allow_release.set()

    await asyncio.wait_for(probe(), timeout=1.0)
    assert probe_ran.is_set()
    store.release.set()
    assert await task == [1.0]
    watchdog.join(timeout=1.0)
    assert not store.released_before_probe


@pytest.mark.asyncio
async def test_async_store_save_keeps_event_loop_schedulable() -> None:
    """A blocked persistent write must not prevent another coroutine from running."""
    store = _Store(block_set=True)
    cache = QueryEmbeddingCache(store=store)
    watchdog = threading.Thread(target=store.release_after_probe, daemon=True)
    watchdog.start()

    task = asyncio.create_task(
        cache.async_get_or_compute(
            model_name="model", query="query", compute=lambda: [1.0]
        )
    )
    await asyncio.to_thread(store.set_started.wait, 1.0)
    probe_ran = asyncio.Event()

    async def probe() -> None:
        probe_ran.set()
        store.allow_release.set()

    await asyncio.wait_for(probe(), timeout=1.0)
    assert probe_ran.is_set()
    store.release.set()
    assert await task == [1.0]
    watchdog.join(timeout=1.0)
    assert not store.released_before_probe


@pytest.mark.asyncio
async def test_async_store_hit_avoids_compute_after_miss_warms_store() -> None:
    """A persistent miss populates the store, and a fresh cache reads that value."""
    store = _Store(release=threading.Event())
    store.release.set()
    first = QueryEmbeddingCache(store=store)
    second = QueryEmbeddingCache(store=store)
    calls = 0

    assert await first.async_get_or_compute(
        model_name="model", query=" query ", compute=lambda: [2.0]
    ) == [2.0]

    def unexpected_compute() -> list[float]:
        nonlocal calls
        calls += 1
        return [3.0]

    assert await second.async_get_or_compute(
        model_name="model", query="query", compute=unexpected_compute
    ) == [2.0]
    assert calls == 0


@pytest.mark.asyncio
async def test_async_store_failures_remain_best_effort() -> None:
    """Persistent read and write failures must not change the computed result."""

    class FailingStore(_Store):
        def get(self, key: str) -> Any | None:
            raise RuntimeError("read failed")

        def set(self, key: str, value: Any, epoch: int) -> None:
            raise RuntimeError("write failed")

    cache = QueryEmbeddingCache(store=FailingStore())
    assert await cache.async_get_or_compute(
        model_name="model", query="query", compute=lambda: [4.0]
    ) == [4.0]


@pytest.mark.asyncio
async def test_async_misses_still_use_single_flight() -> None:
    """Concurrent async misses compute once and share the embedding."""
    cache = QueryEmbeddingCache()
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def compute() -> list[float]:
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return [5.0]

    leader = asyncio.create_task(
        cache.async_get_or_compute(model_name="model", query="query", compute=compute)
    )
    await started.wait()
    follower = asyncio.create_task(
        cache.async_get_or_compute(model_name="model", query="query", compute=compute)
    )
    release.set()

    assert await asyncio.gather(leader, follower) == [[5.0], [5.0]]
    assert calls == 1
