"""Private coordination for versioned record hydration."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

from searchkernel.domain import Record, RecordIdentity
from searchkernel.ports.search_results import (
    FailureStage,
    RecordSearchFailure,
)
from searchkernel.runtime.canonical_cache import (
    HydrationCache,
    HydrationCacheKey,
)
from searchkernel.search.pipeline_candidate_acquisition import _is_async_callable

if TYPE_CHECKING:
    from searchkernel.search.record_pipeline import RecordSearchCandidate


HydrateRecord = Callable[
    [RecordIdentity], Awaitable[Record | None]
]
HydrationErrorHandler = Callable[
    [FailureStage, Exception, list[RecordSearchFailure]], None
]


class RecordHydrationCoordinator:
    """Coordinate cache-aware hydration without owning pipeline state."""

    def __init__(
        self,
        *,
        hydrator: object,
        hydrate_record: HydrateRecord,
        hydration_cache: HydrationCache[Record | None] | None,
        policy_version: str | None,
        hydration_version: object | None,
        hydration_version_provider: Callable[
            [RecordIdentity], object | Awaitable[object]
        ]
        | None,
        max_batch_size: int,
        max_concurrency: int,
        handle_error: HydrationErrorHandler,
    ) -> None:
        self._hydrator = hydrator
        self._hydrate_record = hydrate_record
        self._hydration_cache = hydration_cache
        self._policy_version = policy_version
        self._hydration_version = hydration_version
        self._hydration_version_provider = hydration_version_provider
        self._max_batch_size = max_batch_size
        self._max_concurrency = max_concurrency
        self._handle_error = handle_error

    async def hydrate_candidates(
        self,
        candidates: Sequence[RecordSearchCandidate],
        failures: list[RecordSearchFailure],
        diagnostics: list[str],
    ) -> list[tuple[RecordSearchCandidate, Record | None]]:
        if not candidates:
            return []
        versioned: list[
            tuple[RecordSearchCandidate, HydrationCacheKey]
        ] = []
        versioned_keys: dict[str, HydrationCacheKey] = {}
        cached: list[tuple[RecordSearchCandidate, Record | None]] = []
        misses: list[RecordSearchCandidate] = []
        if self._hydration_cache is not None and self._policy_version is not None:
            hydration_versions = await self._hydration_versions_for(
                [candidate.identity for candidate in candidates],
                diagnostics,
            )
            for candidate in candidates:
                try:
                    if candidate.storage_key in hydration_versions:
                        version = hydration_versions[candidate.storage_key]
                    else:
                        version = await self._hydration_version_for(
                            candidate.identity
                        )
                except Exception as error:  # noqa: BLE001 - cache is optional
                    misses.append(candidate)
                    diagnostics.append(
                        f"hydration_cache:bypass:{type(error).__name__}"
                    )
                    continue
                if version is None:
                    misses.append(candidate)
                    continue
                try:
                    key = HydrationCacheKey.build(
                        candidate.identity,
                        record_version=version,
                        policy_version=self._policy_version,
                    )
                    hit, record = self._hydration_cache.lookup(key)
                    versioned.append((candidate, key))
                    versioned_keys[candidate.storage_key] = key
                except Exception as error:  # noqa: BLE001 - cache is optional
                    misses.append(candidate)
                    diagnostics.append(
                        f"hydration_cache:bypass:{type(error).__name__}"
                    )
                    continue
                if hit:
                    cached.append((candidate, record))
                    diagnostics.append("hydration_cache:hit")
                else:
                    try:
                        leader, shared = await self._hydration_cache.async_wait_for_miss(
                            key
                        )
                    except Exception as error:  # noqa: BLE001 - cache is optional
                        leader, shared = True, None
                        diagnostics.append(
                            f"hydration_cache:bypass:{type(error).__name__}"
                        )
                    if leader:
                        misses.append(candidate)
                        diagnostics.append("hydration_cache:miss")
                    else:
                        cached.append((candidate, shared))
                        diagnostics.append("hydration_cache:coalesced")
        else:
            misses = list(candidates)
            if self._hydration_cache is not None:
                diagnostics.append("hydration_cache:bypass:missing_policy_version")

        if not misses:
            return cached
        hydration_cache = self._hydration_cache
        hydrate_records = getattr(self._hydrator, "hydrate_records", None)
        if callable(hydrate_records):
            loaded: list[tuple[RecordSearchCandidate, Record | None]] = []
            for offset in range(0, len(misses), self._max_batch_size):
                hydration_batch = misses[offset : offset + self._max_batch_size]
                result = await _capture_stage(
                    "hydration",
                    lambda batch=hydration_batch: _call_async(
                        hydrate_records,
                        [candidate.identity for candidate in batch],
                    ),
                )
                if result[2] is not None:
                    for candidate in hydration_batch:
                        key = versioned_keys.get(candidate.storage_key)
                        if key is not None and hydration_cache is not None:
                            hydration_cache.fail(key, result[2])
                records = self._consume_stage(result, failures)
                if records is None:
                    continue
                records_by_key = cast(Mapping[str, Record | None], records)
                loaded.extend(
                    (candidate, records_by_key.get(candidate.storage_key))
                    for candidate in hydration_batch
                )
            self._store_hydration_cache(versioned, loaded, diagnostics)
            hydrated_by_key = {
                candidate.storage_key: record
                for candidate, record in [*cached, *loaded]
            }
            return [
                (candidate, hydrated_by_key[candidate.storage_key])
                for candidate in candidates
                if candidate.storage_key in hydrated_by_key
            ]

        semaphore = asyncio.Semaphore(self._max_concurrency)

        async def hydrate(
            candidate: RecordSearchCandidate,
        ) -> tuple[RecordSearchCandidate, Record | None, Exception | None]:
            async with semaphore:
                try:
                    return candidate, await self._hydrate_record(candidate.identity), None
                except Exception as error:  # noqa: BLE001 - captured per candidate
                    return candidate, None, error

        loaded = await _gather_tasks(
            [asyncio.create_task(hydrate(candidate)) for candidate in misses]
        )
        hydrated: list[tuple[RecordSearchCandidate, Record | None]] = []
        for candidate, record, error in cast(
            list[tuple["RecordSearchCandidate", Record | None, Exception | None]],
            loaded,
        ):
            if error is not None:
                key = versioned_keys.get(candidate.storage_key)
                if key is not None and hydration_cache is not None:
                    hydration_cache.fail(key, error)
                self._handle_error("hydration", error, failures)
                continue
            hydrated.append((candidate, record))
        self._store_hydration_cache(versioned, hydrated, diagnostics)
        hydrated_by_key = {
            candidate.storage_key: record
            for candidate, record in [*cached, *hydrated]
        }
        return [
            (candidate, hydrated_by_key[candidate.storage_key])
            for candidate in candidates
            if candidate.storage_key in hydrated_by_key
        ]

    def _consume_stage(
        self,
        result: tuple[FailureStage, Any, Exception | None],
        failures: list[RecordSearchFailure],
    ) -> Any | None:
        stage, value, error = result
        if error is not None:
            self._handle_error(stage, error, failures)
            return None
        return value

    async def _hydration_versions_for(
        self,
        identities: Sequence[RecordIdentity],
        diagnostics: list[str],
    ) -> Mapping[str, object | None]:
        provider = self._hydration_version_provider
        if provider is None or self._hydration_version is not None:
            return {}
        batch_provider = getattr(provider, "hydration_versions", None)
        if not callable(batch_provider):
            return {}
        try:
            versions = await _call_async(batch_provider, identities)
            if not isinstance(versions, Mapping):
                raise TypeError("hydration_versions must return a mapping")
            return cast(Mapping[str, object | None], versions)
        except Exception as error:  # noqa: BLE001 - scalar fallback preserves compatibility
            diagnostics.append(
                f"hydration_cache:batch_version_fallback:{type(error).__name__}"
            )
            return {}

    async def _hydration_version_for(
        self,
        identity: RecordIdentity,
    ) -> object | None:
        if self._hydration_version is not None:
            return self._hydration_version
        provider = self._hydration_version_provider
        if provider is not None:
            return await _call_async(provider, identity)
        for name in ("record_epoch", "hydration_epoch"):
            value = getattr(self._hydrator, name, None)
            if callable(value):
                return await _call_async(value)
            if value is not None:
                return value
        return None

    def _store_hydration_cache(
        self,
        versioned: Sequence[tuple[RecordSearchCandidate, HydrationCacheKey]],
        loaded: Sequence[tuple[RecordSearchCandidate, Record | None]],
        diagnostics: list[str],
    ) -> None:
        if self._hydration_cache is None:
            return
        keys = {candidate.storage_key: key for candidate, key in versioned}
        for candidate, record in loaded:
            key = keys.get(candidate.storage_key)
            if key is None:
                continue
            try:
                self._hydration_cache.set(key, record)
            except Exception as error:  # noqa: BLE001 - cache is optional
                self._hydration_cache.fail(key, error)
                diagnostics.append(
                    f"hydration_cache:error:{type(error).__name__}"
                )


async def _call_async[T](
    function: Callable[..., T | Awaitable[T]],
    *args: Any,
    **kwargs: Any,
) -> T:
    if _is_async_callable(function):
        value = function(*args, **kwargs)
    else:
        value = await asyncio.to_thread(function, *args, **kwargs)
    if inspect.isawaitable(value):
        return await value
    return value


async def _capture_stage(
    stage: FailureStage,
    operation: Callable[[], Awaitable[Any]],
) -> tuple[FailureStage, Any, Exception | None]:
    try:
        return stage, await operation(), None
    except Exception as error:  # noqa: BLE001 - stage errors are handled by the caller
        return stage, None, error


async def _gather_tasks(
    tasks: Sequence[asyncio.Task[Any] | None],
) -> list[Any]:
    pending = [task for task in tasks if task is not None]
    if not pending:
        return []
    try:
        return list(await asyncio.gather(*pending))
    except BaseException:
        for task in pending:
            if not task.done():
                task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        raise
