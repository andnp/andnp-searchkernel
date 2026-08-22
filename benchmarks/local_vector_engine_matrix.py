"""Compare local exact-vector engines across corpus sizes and dimensions.

The matrix measures warmed SQLite exact search, warmed FAISS exact search, and
the first plus steady-state calls through ``engine="auto"``. The automatic
path calibrates both engines on its first call, then reuses the winner for the
current vector epoch and filter shape.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import time
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from typing import Any

import numpy as np

from searchkernel.domain import Record
from searchkernel.indices import (
    FAISSLocalVectorStore,
    LocalRecordBackend,
    LocalVectorStore,
)
from searchkernel.indices.local_vectors import PackedVectorCodec

_TIMESTAMP = datetime(2026, 1, 1, tzinfo=UTC)
_MODEL_NAME = "benchmark-local-vector-routing-v1"
_UPSERT_BATCH_SIZE = 2_000


def _time_search(
    search: Callable[[], object],
    *,
    warmups: int,
    repetitions: int,
) -> float:
    for _ in range(warmups):
        search()
    samples: list[float] = []
    for _ in range(repetitions):
        started = time.perf_counter()
        search()
        samples.append((time.perf_counter() - started) * 1_000)
    return statistics.median(samples)


def _build_corpus(
    record_count: int,
    dim: int,
    seed: int,
) -> tuple[LocalRecordBackend, list[float]]:
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
    backend = LocalRecordBackend(faiss_threshold=1)
    for start in range(0, record_count, _UPSERT_BATCH_SIZE):
        backend.upsert(records[start : start + _UPSERT_BATCH_SIZE], _MODEL_NAME, dim)
    query = PackedVectorCodec.normalize(rng.standard_normal(dim).tolist(), dim)
    return backend, query.tolist()


def _run_case(
    record_count: int,
    dim: int,
    *,
    seed: int,
    warmups: int,
    repetitions: int,
) -> dict[str, Any]:
    backend, query = _build_corpus(record_count, dim, seed)
    exact_ms = _time_search(
        lambda: backend.search_vector(
            query, 10, model_name=_MODEL_NAME, dim=dim
        ),
        warmups=warmups,
        repetitions=repetitions,
    )
    if importlib.util.find_spec("faiss") is None:
        faiss_result: dict[str, object] = {"status": "unavailable"}
    else:
        faiss = FAISSLocalVectorStore(backend)
        exact_hits = backend.search_vector(
            query, 10, model_name=_MODEL_NAME, dim=dim
        )
        faiss_hits = faiss.search(
            query, 10, model_name=_MODEL_NAME, dim=dim
        )
        faiss_ms = _time_search(
            lambda: faiss.search(
                query, 10, model_name=_MODEL_NAME, dim=dim
            ),
            warmups=warmups,
            repetitions=repetitions,
        )
        faiss_result = {
            "status": "available",
            "latency_p50_ms": faiss_ms,
            "same_results": [hit.storage_key for hit in exact_hits]
            == [hit.storage_key for hit in faiss_hits],
        }

    adaptive = LocalVectorStore(backend, engine="auto")
    started = time.perf_counter()
    adaptive.search(query, 10, model_name=_MODEL_NAME, dim=dim)
    calibration_ms = (time.perf_counter() - started) * 1_000
    steady_state_ms = _time_search(
        lambda: adaptive.search(query, 10, model_name=_MODEL_NAME, dim=dim),
        warmups=warmups,
        repetitions=repetitions,
    )
    measurement = adaptive.last_routing_measurement
    return {
        "record_count": record_count,
        "dimension": dim,
        "seed": seed,
        "sqlite_exact_p50_ms": exact_ms,
        "faiss_exact": faiss_result,
        "adaptive": {
            "calibration_p50_ms": calibration_ms,
            "steady_state_p50_ms": steady_state_ms,
            "selected_engine": adaptive.engine_name,
            "measurement": (
                None
                if measurement is None
                else {
                    "sqlite_ms": measurement.sqlite_ms,
                    "faiss_ms": measurement.faiss_ms,
                    "selected": measurement.selected,
                }
            ),
        },
    }


def run_matrix(
    record_counts: Iterable[int],
    dimensions: Iterable[int],
    *,
    seed: int = 0,
    warmups: int = 2,
    repetitions: int = 5,
) -> list[dict[str, Any]]:
    """Run every requested corpus-size and dimension combination."""
    return [
        _run_case(
            record_count,
            dim,
            seed=seed,
            warmups=warmups,
            repetitions=repetitions,
        )
        for record_count in record_counts
        for dim in dimensions
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records",
        type=int,
        nargs="+",
        default=[10_000, 100_000],
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        nargs="+",
        default=[32, 384],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()
    if (
        any(value < 1 for value in args.records)
        or any(value < 1 for value in args.dimensions)
        or args.warmups < 0
        or args.repetitions < 1
    ):
        parser.error("records, dimensions, and repetitions must be positive")
    print(
        json.dumps(
            run_matrix(
                args.records,
                args.dimensions,
                seed=args.seed,
                warmups=args.warmups,
                repetitions=args.repetitions,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
