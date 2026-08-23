"""Compare scalar local-vector scoring with a minimal shared-mask GEMM.

This is intentionally benchmark-only: it does not exercise or alter a local
store API.  It models the exact normalized float32 score and deterministic
top-k contract so the matrix operation can be tested independently of
snapshot, routing, cache, and block-search policy.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

import numpy as np

from searchkernel.domain import Record
from searchkernel.indices import LocalRecordBackend


def _library_entry_parity() -> bool:
    """Verify parity through the concrete local backend entry point."""
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    records = [
        Record(
            workspace_id="workspace",
            source_kind="note",
            source_id=f"record-{index}",
            title=f"record-{index}",
            body=f"record-{index}",
            created_at=timestamp,
            updated_at=timestamp,
            embedding=vector.tolist(),
        )
        for index, vector in enumerate(
            np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        )
    ]
    backend = LocalRecordBackend()
    backend.upsert(records, "benchmark", 2)
    queries = [[1.0, 0.0], [0.0, 1.0]]
    batch = backend.search_vector_batch(queries, 2, model_name="benchmark", dim=2)
    scalar = [
        backend.search_vector(query, 2, model_name="benchmark", dim=2)
        for query in queries
    ]
    return [
        [hit.storage_key for hit in result] for result in batch
    ] == [[hit.storage_key for hit in result] for result in scalar]


def _top_k(scores: np.ndarray, keys: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    candidates = np.argpartition(-scores, min(k, len(scores)) - 1)[:k]
    ordered = sorted(
        (int(index) for index in candidates),
        key=lambda index: (-float(scores[index]), str(keys[index])),
    )
    indexes = np.asarray(ordered[:k], dtype=np.intp)
    return keys[indexes], scores[indexes]


def _scalar(matrix: np.ndarray, queries: np.ndarray, eligible: np.ndarray, k: int) -> list[tuple[np.ndarray, np.ndarray]]:
    selected = matrix[eligible]
    return [_top_k(selected @ query, np.flatnonzero(eligible), k) for query in queries]


def _gemm(matrix: np.ndarray, queries: np.ndarray, eligible: np.ndarray, k: int) -> list[tuple[np.ndarray, np.ndarray]]:
    positions = np.flatnonzero(eligible)
    scores = matrix[positions] @ queries.T
    return [_top_k(scores[:, column], positions, k) for column in range(scores.shape[1])]


def _parity(left: list[tuple[np.ndarray, np.ndarray]], right: list[tuple[np.ndarray, np.ndarray]]) -> bool:
    return all(
        np.array_equal(left_result[0], right_result[0])
        and np.allclose(left_result[1], right_result[1], rtol=0.0, atol=2e-6)
        for left_result, right_result in zip(left, right, strict=True)
    )


def _time(
    fn: Any,
    *,
    warmups: int,
    repetitions: int,
) -> dict[str, float]:
    for _ in range(warmups):
        fn()
    samples = []
    for _ in range(repetitions):
        started = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - started) * 1_000)
    library_entry_parity = _library_entry_parity()
    return {
        "p50_ms": statistics.median(samples),
        "p95_ms": sorted(samples)[min(len(samples) - 1, int(len(samples) * 0.95))],
    }


def run_case(
    records: int,
    query_count: int,
    *,
    dimension: int,
    seed: int,
    warmups: int,
    repetitions: int,
    filtered: bool,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed + records + query_count + int(filtered))
    matrix = rng.standard_normal((records, dimension)).astype(np.float32)
    matrix /= np.linalg.norm(matrix.astype(np.float64), axis=1, keepdims=True).astype(np.float32)
    queries = rng.standard_normal((query_count, dimension)).astype(np.float32)
    queries /= np.linalg.norm(queries.astype(np.float64), axis=1, keepdims=True).astype(np.float32)
    eligible = np.ones(records, dtype=bool) if not filtered else (np.arange(records) % 8 == 2)
    scalar = lambda: _scalar(matrix, queries, eligible, 10)
    gemm = lambda: _gemm(matrix, queries, eligible, 10)
    scalar_result = scalar()
    gemm_result = gemm()
    scalar_timing = _time(scalar, warmups=warmups, repetitions=repetitions)
    gemm_timing = _time(gemm, warmups=warmups, repetitions=repetitions)
    speedup = scalar_timing["p50_ms"] / gemm_timing["p50_ms"]
    return {
        "records": records,
        "dimension": dimension,
        "queries": query_count,
        "filter": "shared_scalar" if filtered else "unfiltered",
        "eligible_rows": int(eligible.sum()),
        "scalar": scalar_timing,
        "minimal_gemm": gemm_timing,
        "speedup": speedup,
        "parity": _parity(scalar_result, gemm_result),
    }


def run_benchmark(
    records: Iterable[int] = (20_000, 100_000),
    queries: Iterable[int] = (8, 32, 64),
    *,
    dimension: int = 384,
    seed: int = 0,
    warmups: int = 2,
    repetitions: int = 15,
) -> dict[str, Any]:
    results = [
        run_case(
            record_count,
            query_count,
            dimension=dimension,
            seed=seed,
            warmups=warmups,
            repetitions=repetitions,
            filtered=filtered,
        )
        for record_count in records
        for query_count in queries
        for filtered in (False, True)
    ]
    library_entry_parity = _library_entry_parity()
    return {
        "workload": "minimal shared-mask GEMM versus scalar local-vector scoring",
        "warmups": warmups,
        "repetitions": repetitions,
        "results": results,
        "library_entry_parity": library_entry_parity,
        "passed": all(result["parity"] for result in results)
        and library_entry_parity,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=int, nargs="+", default=[20_000, 100_000])
    parser.add_argument("--queries", type=int, nargs="+", default=[8, 32, 64])
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=15)
    args = parser.parse_args()
    if any(value < 1 for value in (*args.records, *args.queries, args.warmups, args.repetitions)):
        parser.error("records, queries, warmups, and repetitions must be positive")
    report = run_benchmark(
        args.records,
        args.queries,
        warmups=args.warmups,
        repetitions=args.repetitions,
    )
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
