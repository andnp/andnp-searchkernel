"""Regression benchmark for local vector search hotspots.

Builds a deterministic seeded corpus and reports p50/p95 latency, in
milliseconds, for four measured cases against the local record backend's
exact-search path:

  * ``snapshot_build``          -- ``VectorSnapshot.from_rows`` via a forced
                                    cache miss (``_get_vector_snapshot``)
  * ``search_unfiltered``       -- ``search_vector`` with no filters
  * ``search_candidate_bounded``-- ``search_vector`` restricted to a small
                                    ``candidate_storage_keys`` set
  * ``search_metadata_filtered``-- ``search_vector`` with a metadata filter
                                    that cannot be prefiltered by scalar
                                    fields alone

Each case warms twice and measures repeated calls. Run this script before and
after a change to ``searchkernel/indices/local_vectors.py`` or the
``_VectorEngine`` region of ``searchkernel/indices/local.py``; it
intentionally has no CI latency gate.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import numpy as np

from searchkernel.domain import Record
from searchkernel.indices import LocalRecordBackend
from searchkernel.indices.local_vectors import PackedVectorCodec

_TIMESTAMP = datetime(2026, 1, 1, tzinfo=UTC)
_MODEL_NAME = "benchmark-local-vector-hotspots-v1"
_UPSERT_BATCH_SIZE = 2_000


def _percentile(samples: list[float], percentile: float) -> float:
    ordered = sorted(samples)
    position = min(len(ordered) - 1, int(len(ordered) * percentile))
    return ordered[position]


def _timeit(
    fn: Callable[[], object],
    *,
    warmups: int,
    repetitions: int,
) -> dict[str, float]:
    for _ in range(warmups):
        fn()
    samples: list[float] = []
    for _ in range(repetitions):
        started = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - started) * 1_000)
    return {
        "latency_p50_ms": statistics.median(samples),
        "latency_p95_ms": _percentile(samples, 0.95),
    }


def _build_corpus(
    record_count: int,
    dim: int,
    seed: int,
) -> tuple[LocalRecordBackend, list[Record], np.ndarray]:
    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((record_count, dim)).astype(np.float32)
    records = [
        Record(
            workspace_id=f"workspace-{index % 8}",
            source_kind="note" if index % 2 == 0 else "commit",
            source_id=f"record-{index:07d}",
            title=f"title {index}",
            body=f"benchmark body text for record {index}",
            created_at=_TIMESTAMP,
            updated_at=_TIMESTAMP,
            metadata={"project_id": f"p{index % 11}"},
            embedding=vectors[index].tolist(),
        )
        for index in range(record_count)
    ]
    backend = LocalRecordBackend()
    for start in range(0, record_count, _UPSERT_BATCH_SIZE):
        backend.upsert(records[start : start + _UPSERT_BATCH_SIZE], _MODEL_NAME, dim)
    query = PackedVectorCodec.normalize(rng.standard_normal(dim).tolist(), dim)
    return backend, records, query


def run_benchmark(
    record_count: int,
    dim: int,
    *,
    seed: int = 0,
    warmups: int = 2,
    repetitions: int = 7,
) -> dict[str, Any]:
    """Measure the four hot paths and return their latency percentiles."""
    backend, records, query = _build_corpus(record_count, dim, seed)
    query_list = query.tolist()
    engine = backend._vector_snapshot_engine

    candidate_count = max(1, record_count // 200)
    candidate_keys = [record.storage_key for record in records[:candidate_count]]

    def build_snapshot() -> None:
        engine._vector_snapshots.clear()
        engine._get_vector_snapshot(_MODEL_NAME, dim)

    cases: dict[str, Callable[[], object]] = {
        "snapshot_build": build_snapshot,
        "search_unfiltered": lambda: backend.search_vector(
            query_list, 10, model_name=_MODEL_NAME, dim=dim
        ),
        "search_candidate_bounded": lambda: backend.search_vector(
            query_list,
            10,
            model_name=_MODEL_NAME,
            dim=dim,
            filters={"candidate_storage_keys": candidate_keys},
        ),
        "search_metadata_filtered": lambda: backend.search_vector(
            query_list,
            10,
            model_name=_MODEL_NAME,
            dim=dim,
            filters={"project_ids": ["p3"]},
        ),
    }

    results = {
        name: _timeit(fn, warmups=warmups, repetitions=repetitions)
        for name, fn in cases.items()
    }
    return {
        "record_count": record_count,
        "dim": dim,
        "seed": seed,
        "candidate_count": candidate_count,
        "warmups": warmups,
        "repetitions": repetitions,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=int, default=20_000)
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repetitions", type=int, default=7)
    args = parser.parse_args()
    if args.records < 1 or args.dim < 1 or args.repetitions < 1:
        parser.error("--records, --dim, and --repetitions must be positive")
    print(
        json.dumps(
            run_benchmark(
                args.records,
                args.dim,
                seed=args.seed,
                repetitions=args.repetitions,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
