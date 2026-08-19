"""Evidence benchmark for record-search candidate caching.

The workload uses deterministic ``RecordSearchCandidate`` payloads with
workspace/source identity and keyword/vector/graph provenance. It measures
cache miss/set, warm hits, async single-flight waiters, and repeated local
keyword retrieval for 10, 50, and 100 candidates. The local workload counts
source acquisitions so cache effectiveness is visible alongside latency.

Interpretation: isolated clone timings are diagnostic; the repeated local
retrieval workload is the decision signal. Production cache cloning is only
justified after a repeatable material end-to-end improvement (at least 15%
warm p50 under a realistic workload) with zero mutation leaks. This benchmark
has no machine-specific CI threshold.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import functools
import json
import statistics
import tempfile
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from searchkernel.domain import Record, RecordIdentity, SearchResultProvenance
from searchkernel.indices import LocalRecordBackend
from searchkernel.runtime import CandidateCacheKey, CandidateResultCache, SearchEpochs
from searchkernel.search.record_pipeline import RecordSearchCandidate

COUNTS = (10, 50, 100)
DEFAULT_REPETITIONS = 200


def _percentile(samples: list[float], percentile: float) -> float:
    ordered = sorted(samples)
    index = min(len(ordered) - 1, max(0, int(len(ordered) * percentile) - 1))
    return ordered[index]


def _summary(samples: list[float]) -> dict[str, float]:
    return {
        "p50_ms": statistics.median(samples),
        "p95_ms": _percentile(samples, 0.95),
        "p99_ms": _percentile(samples, 0.99),
    }


def _key(count: int, query: str = "cache evidence") -> CandidateCacheKey:
    return CandidateCacheKey.build(
        query=query,
        filters={"workspace_id": "workspace-0", "source_kind": "note"},
        requested_limit=count,
        acquisition_limit=count,
        adaptive_limit=None,
        routing_fingerprint="benchmark-routing-v1",
        encoder_namespace="benchmark-encoder|dim=3",
        epochs=SearchEpochs(keyword=1, vector=1, graph=1),
        policy_version="benchmark-policy-v1",
    )


def _candidates(count: int) -> tuple[RecordSearchCandidate, ...]:
    result: list[RecordSearchCandidate] = []
    for index in range(count):
        identity = RecordIdentity(
            workspace_id=f"workspace-{index % 4}",
            source_kind="note" if index % 2 == 0 else "commit",
            source_id=f"candidate-{index}",
        )
        provenance = SearchResultProvenance(record_identity=identity)
        provenance.add_strategy("keyword", index + 1, 1.0 / (index + 1))
        provenance.add_strategy("vector", index + 1, 0.9 / (index + 1))
        provenance.add_strategy("graph", index + 1, 0.8 / (index + 1))
        result.append(
            RecordSearchCandidate(
                identity=identity,
                score=1.0 / (index + 1),
                provenance=provenance,
                priority=index % 3,
            )
        )
    return tuple(result)


def _typed_clone(value: tuple[RecordSearchCandidate, ...]) -> tuple[RecordSearchCandidate, ...]:
    return tuple(
        RecordSearchCandidate(
            identity=candidate.identity,
            score=candidate.score,
            provenance=candidate.provenance.clone(),
            priority=candidate.priority,
        )
        for candidate in value
    )


def _measure_operation(
    operation: Callable[[], object], repetitions: int
) -> dict[str, float]:
    samples: list[float] = []
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        operation()
        samples.append((time.perf_counter_ns() - started) / 1_000_000)
    return _summary(samples)


def _cache_miss_set(
    cache: CandidateResultCache[tuple[RecordSearchCandidate, ...]],
    key: CandidateCacheKey,
    payload: tuple[RecordSearchCandidate, ...],
) -> None:
    cache.get(key)
    cache.set(key, payload)


def _cache_hit(
    cache: CandidateResultCache[tuple[RecordSearchCandidate, ...]],
    key: CandidateCacheKey,
) -> None:
    cache.get(key)


def _mutation_isolated(
    cache: CandidateResultCache[tuple[RecordSearchCandidate, ...]],
    key: CandidateCacheKey,
    payload: tuple[RecordSearchCandidate, ...],
) -> bool:
    cache.set(key, payload)
    payload[0].provenance.add_strategy("producer-mutation", 99, 0.1)
    returned = cache.get(key)
    if returned is None:
        return False
    returned[0].provenance.strategy_details.clear()
    stored = cache.get(key)
    return stored is not None and stored[0].provenance.strategies == (
        "keyword",
        "vector",
        "graph",
    )


async def _measure_coalesced(
    payload: tuple[RecordSearchCandidate, ...], repetitions: int
) -> dict[str, Any]:
    cache: CandidateResultCache[tuple[RecordSearchCandidate, ...]] = CandidateResultCache()
    key = _key(len(payload), "coalesced")
    started = asyncio.Event()
    release = asyncio.Event()

    async def compute() -> tuple[RecordSearchCandidate, ...]:
        started.set()
        await release.wait()
        return payload

    samples: list[float] = []
    for _ in range(repetitions):
        cache = CandidateResultCache()
        started.clear()
        release.clear()
        leader = asyncio.create_task(cache.async_get_or_compute(key, compute))
        await started.wait()
        waiter = asyncio.create_task(cache.async_get_or_compute(key, compute))
        measured = time.perf_counter_ns()
        release.set()
        await asyncio.gather(leader, waiter)
        samples.append((time.perf_counter_ns() - measured) / 1_000_000)
    return {
        **_summary(samples),
        "coalesced_waiters": cache.metrics.coalesced_waiters,
        "compute_calls": 1,
    }


def _records(count: int) -> list[Record]:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    return [
        Record(
            source_kind="note",
            source_id=f"candidate-{index}",
            title=f"Cache evidence {index}",
            body="cache evidence benchmark retrieval payload",
            created_at=timestamp,
            updated_at=timestamp,
            workspace_id=f"workspace-{index % 4}",
            uri=f"workspace-{index % 4}/notes/{index}.md",
            metadata={"source": "benchmark", "ordinal": index},
        )
        for index in range(count)
    ]


def _local_retrieval(count: int, repetitions: int) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="searchkernel-candidate-cache-") as directory:
        backend = LocalRecordBackend(Path(directory) / "records.db")
        backend.index(_records(count))
        cache: CandidateResultCache[tuple[RecordSearchCandidate, ...]] = CandidateResultCache()
        key = _key(count, "cache evidence benchmark retrieval")
        source_acquisitions = 0

        def retrieve() -> tuple[RecordSearchCandidate, ...]:
            nonlocal source_acquisitions
            candidates = cache.get(key)
            if candidates is not None:
                return candidates
            source_acquisitions += 1
            hits = backend.search_keyword("cache evidence benchmark retrieval", count)
            candidates = tuple(
                RecordSearchCandidate(
                    identity=hit.identity,
                    score=hit.score,
                    provenance=SearchResultProvenance(
                        strategies=("keyword",),
                        record_identity=hit.identity,
                    ),
                )
                for hit in hits
            )
            cache.set(key, candidates)
            return candidates

        retrieve()
        samples: list[float] = []
        for _ in range(repetitions):
            started = time.perf_counter_ns()
            result = retrieve()
            if len(result) != count:
                raise RuntimeError(f"expected {count} candidates, got {len(result)}")
            samples.append((time.perf_counter_ns() - started) / 1_000_000)
        metrics = cache.metrics
        backend.close()
    total_requests = repetitions + 1
    return {
        **_summary(samples),
        "requests": total_requests,
        "cache_hit_ratio": metrics.hits / total_requests,
        "cache_hits": metrics.hits,
        "cache_misses": metrics.misses,
        "source_acquisitions": source_acquisitions,
    }


async def run_benchmark(repetitions: int = DEFAULT_REPETITIONS) -> dict[str, Any]:
    """Run deterministic cache evidence measurements for each candidate count."""
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    results: dict[str, Any] = {}
    for count in COUNTS:
        payload = _candidates(count)
        key = _key(count)
        cache: CandidateResultCache[tuple[RecordSearchCandidate, ...]] = CandidateResultCache()
        miss_cache = CandidateResultCache[tuple[RecordSearchCandidate, ...]]()
        miss_samples: list[float] = []
        for _ in range(repetitions):
            miss_cache = CandidateResultCache()
            miss_samples.append(
                _measure_operation(
                    functools.partial(_cache_miss_set, miss_cache, key, payload), 1
                )["p50_ms"]
            )
        cache.set(key, payload)
        warm_hit = _measure_operation(functools.partial(_cache_hit, cache, key), repetitions)
        clone_deepcopy = _measure_operation(functools.partial(copy.deepcopy, payload), repetitions)
        clone_typed = _measure_operation(functools.partial(_typed_clone, payload), repetitions)
        results[str(count)] = {
            "candidate_count": count,
            "provenance_size_bytes": sum(
                len(json.dumps(candidate.provenance.to_dict(), sort_keys=True))
                for candidate in payload
            ),
            "cache_miss_set": _summary(miss_samples),
            "warm_cache_hit": warm_hit,
            "async_single_flight": await _measure_coalesced(payload, max(3, repetitions // 10)),
            "clone_comparison": {
                "current_deepcopy": clone_deepcopy,
                "benchmark_only_typed_clone": clone_typed,
            },
            "mutation_isolated": _mutation_isolated(cache, key, payload),
            "local_retrieval": _local_retrieval(count, repetitions),
        }
    return {"schema_version": 1, "repetitions": repetitions, "results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    print(json.dumps(asyncio.run(run_benchmark(args.repetitions)), indent=2))


if __name__ == "__main__":
    main()
