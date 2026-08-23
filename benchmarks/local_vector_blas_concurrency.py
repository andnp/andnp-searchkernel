"""Measure BLAS and application-worker concurrency for local vector search.

The benchmark keeps snapshot construction separate from warmed query timing.
Thread limits are applied only inside ``threadpoolctl`` contexts when that
optional package is installed; the process environment is never changed.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from typing import Any

from benchmarks.local_vector_search_hotspots import _build_corpus

try:
    from threadpoolctl import threadpool_info, threadpool_limits
except ImportError:  # pragma: no cover - depends on the local environment
    threadpool_info = None
    threadpool_limits = None


_MODEL_NAME = "benchmark-local-vector-hotspots-v1"


def _percentile(samples: list[float], percentile: float) -> float:
    ordered = sorted(samples)
    return ordered[min(len(ordered) - 1, int(len(ordered) * percentile))]


def _summary(samples: list[float], query_count: int, elapsed: float) -> dict[str, float]:
    return {
        "p50_ms": statistics.median(samples),
        "p95_ms": _percentile(samples, 0.95),
        "p99_ms": _percentile(samples, 0.99),
        "throughput_qps": query_count / elapsed,
    }


def _limit_context(blas_threads: int | None):
    if blas_threads is None or threadpool_limits is None:
        return nullcontext()
    return threadpool_limits(limits=blas_threads)


def _search(
    backend: Any,
    query: list[float],
    dim: int,
) -> tuple[str, ...]:
    hits = backend.search_vector(query, 10, model_name=_MODEL_NAME, dim=dim)
    return tuple(hit.storage_key for hit in hits)


def _cold_samples(
    backend: Any,
    query: list[float],
    dim: int,
    repetitions: int,
) -> list[float]:
    samples: list[float] = []
    for _ in range(repetitions):
        backend._vector_snapshot_engine._vector_snapshots.clear()
        started = time.perf_counter()
        _search(backend, query, dim)
        samples.append((time.perf_counter() - started) * 1_000)
    return samples


def _warm_concurrent(
    backend: Any,
    query: list[float],
    dim: int,
    workers: int,
    warmups: int,
    repetitions: int,
) -> tuple[dict[str, float], bool]:
    expected = _search(backend, query, dim)
    queries = [query] * workers
    samples: list[float] = []
    parity = True
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for _ in range(warmups):
            list(executor.map(lambda item: _search(backend, item, dim), queries))
        started = time.perf_counter()
        for _ in range(repetitions):
            batch_started = time.perf_counter()
            results = list(executor.map(lambda item: _search(backend, item, dim), queries))
            samples.append((time.perf_counter() - batch_started) * 1_000)
            parity &= all(result == expected for result in results)
    return _summary(samples, repetitions * workers, time.perf_counter() - started), parity


def _run_case(
    records: int,
    dim: int,
    *,
    seed: int,
    workers: Iterable[int],
    blas_threads: Iterable[int | None],
    repetitions: int,
    warmups: int,
) -> dict[str, Any]:
    backend, _, query = _build_corpus(records, dim, seed)
    query_list = query.tolist()
    cold: dict[str, Any] = {}
    try:
        for setting in blas_threads:
            label = "default" if setting is None else str(setting)
            with _limit_context(setting):
                samples = _cold_samples(backend, query_list, dim, repetitions)
            cold[label] = _summary(samples, repetitions, sum(samples) / 1_000)

        warm: list[dict[str, Any]] = []
        for setting in blas_threads:
            label = "default" if setting is None else str(setting)
            with _limit_context(setting):
                for worker_count in workers:
                    metrics, parity = _warm_concurrent(
                        backend,
                        query_list,
                        dim,
                        worker_count,
                        warmups,
                        repetitions,
                    )
                    warm.append(
                        {
                            "blas_threads": label,
                            "workers": worker_count,
                            "parity": parity,
                            **metrics,
                        }
                    )
    finally:
        backend.close()
    return {"records": records, "dim": dim, "cold_serial": cold, "warm": warm}


def run_benchmark(
    records: Iterable[int],
    *,
    dim: int = 384,
    seed: int = 0,
    workers: Iterable[int] = (1, 2, 4, 8, 16),
    blas_threads: Iterable[int | None] = (None, 1, 2, 4, 8),
    warmups: int = 2,
    repetitions: int = 5,
) -> dict[str, Any]:
    """Return cold and warmed local vector concurrency measurements."""
    return {
        "threadpoolctl_available": threadpool_limits is not None,
        "blas": threadpool_info() if threadpool_info is not None else [],
        "dim": dim,
        "seed": seed,
        "warmups": warmups,
        "repetitions": repetitions,
        "cases": [
            _run_case(
                record_count,
                dim,
                seed=seed,
                workers=workers,
                blas_threads=blas_threads,
                warmups=warmups,
                repetitions=repetitions,
            )
            for record_count in records
        ],
        "runtime_policy": "deferred",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=int, nargs="+", default=[20_000])
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--blas-threads", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()
    if (
        any(value < 1 for value in args.records)
        or args.dim < 1
        or any(value < 1 for value in args.workers)
        or any(value < 1 for value in args.blas_threads)
        or args.warmups < 0
        or args.repetitions < 1
    ):
        parser.error("records, dim, workers, blas threads, and repetitions must be positive")
    print(
        json.dumps(
            run_benchmark(
                args.records,
                dim=args.dim,
                seed=args.seed,
                workers=args.workers,
                blas_threads=(None, *args.blas_threads),
                warmups=args.warmups,
                repetitions=args.repetitions,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
