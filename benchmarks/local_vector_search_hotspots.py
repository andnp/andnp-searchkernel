"""Regression benchmark for local vector search hotspots.

Builds a deterministic seeded corpus and reports p50/p95/p99 latency, in
milliseconds, for four measured cases against the local record backend's
exact-search path:

  * ``snapshot_build``          -- ``VectorSnapshot.from_rows`` via a forced
                                    cache miss (``_get_vector_snapshot``)
  * ``search_unfiltered``       -- ``search_vector`` with no filters
  * ``search_candidate_small`` -- ``search_vector`` restricted to a small
                                    ``candidate_storage_keys`` set
  * ``search_candidate_broad`` -- ``search_vector`` restricted to a broader
                                    candidate set
  * ``search_scalar_filtered`` -- scalar workspace/source/status filters
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
from searchkernel.domain.vector_filters import compile_vector_filters
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
        "latency_p99_ms": _percentile(samples, 0.99),
    }


def _stage_timeit(
    fn: Callable[[], object],
    *,
    warmups: int,
    repetitions: int,
) -> dict[str, float]:
    """Measure one benchmark stage using the hotspot timing contract."""
    return _timeit(fn, warmups=warmups, repetitions=repetitions)


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
            metadata={
                "project_id": f"p{index % 11}",
                "file_path": f"src/file-{index % 101}.py",
                "doc_id": f"doc-{index % 97}",
                "acl": ["allowed"] if index % 3 else ["blocked"],
            },
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

    candidate_counts = {
        "candidate_small": max(1, record_count // 200),
        "candidate_broad": max(1, record_count // 20),
    }
    candidate_keys = {
        name: [record.storage_key for record in records[:count]]
        for name, count in candidate_counts.items()
    }

    def build_snapshot() -> None:
        engine._vector_snapshots.clear()
        engine._get_vector_snapshot(_MODEL_NAME, dim)

    cases: dict[str, Callable[[], object]] = {
        "snapshot_build": build_snapshot,
        "search_unfiltered": lambda: backend.search_vector(
            query_list, 10, model_name=_MODEL_NAME, dim=dim
        ),
        "search_candidate_small": lambda: backend.search_vector(
            query_list,
            10,
            model_name=_MODEL_NAME,
            dim=dim,
            filters={"candidate_storage_keys": candidate_keys["candidate_small"]},
        ),
        "search_candidate_broad": lambda: backend.search_vector(
            query_list,
            10,
            model_name=_MODEL_NAME,
            dim=dim,
            filters={"candidate_storage_keys": candidate_keys["candidate_broad"]},
        ),
        "search_scalar_filtered": lambda: backend.search_vector(
            query_list,
            10,
            model_name=_MODEL_NAME,
            dim=dim,
            filters={
                "workspace_id": "workspace-2",
                "source_kind": "note",
                "statuses": ["active"],
            },
        ),
        "search_project_filtered": lambda: backend.search_vector(
            query_list,
            10,
            model_name=_MODEL_NAME,
            dim=dim,
            filters={"project_ids": ["p3"]},
        ),
        "search_path_filtered": lambda: backend.search_vector(
            query_list,
            10,
            model_name=_MODEL_NAME,
            dim=dim,
            filters={"paths": ["file-3.py"]},
        ),
        "search_document_filtered": lambda: backend.search_vector(
            query_list,
            10,
            model_name=_MODEL_NAME,
            dim=dim,
            filters={"document_ids": ["doc-3"]},
        ),
        "search_source_scoped_filtered": lambda: backend.search_vector(
            query_list,
            10,
            model_name=_MODEL_NAME,
            dim=dim,
            filters={
                "source_scoped_filters": {
                    "note": {
                        "metadata_contains_any": {"acl": ["allowed"]}
                    }
                }
            },
        ),
    }

    results: dict[str, Any] = {
        name: _timeit(fn, warmups=warmups, repetitions=repetitions)
        for name, fn in cases.items()
    }
    typed_filters = {
        "project": {"project_ids": ["p3"]},
        "path": {"paths": ["file-3.py"]},
        "document": {"document_ids": ["doc-3"]},
        "source_scoped": {
            "source_scoped_filters": {
                "note": {"metadata_contains_any": {"acl": ["allowed"]}}
            }
        },
    }

    def stage_timings(filters: dict[str, Any]) -> dict[str, Any]:
        predicate = compile_vector_filters(filters)

        def select_eligible_keys() -> None:
            engine._eligible_storage_keys(
                filters, model_name=_MODEL_NAME, dim=dim
            )

        def materialize_metadata() -> None:
            engine._vector_snapshots.pop((_MODEL_NAME, dim), None)
            engine._get_vector_snapshot(
                _MODEL_NAME, dim, materialize_metadata=True
            )

        def evaluate_predicate() -> None:
            snapshot = engine._get_vector_snapshot(
                _MODEL_NAME, dim, materialize_metadata=True
            )
            snapshot.filter_positions(
                filters,
                status_values=engine._status_values(filters),
                filter_values=engine._filter_values,
                compiled_filter=predicate,
            )

        prepared_snapshot = engine._get_vector_snapshot(
            _MODEL_NAME, dim, materialize_metadata=True
        )
        prepared_positions = prepared_snapshot.filter_positions(
            filters,
            status_values=engine._status_values(filters),
            filter_values=engine._filter_values,
            compiled_filter=predicate,
        )

        def score_and_select() -> None:
            scores = prepared_snapshot.matrix[prepared_positions] @ query
            engine._select_top_positions(
                prepared_positions, scores, prepared_snapshot.storage_keys, 10
            )

        stage_results: dict[str, dict[str, float]] = {
            "eligible_key_selection": _stage_timeit(
                select_eligible_keys, warmups=warmups, repetitions=repetitions
            ),
            "snapshot_metadata_materialization": _stage_timeit(
                materialize_metadata, warmups=warmups, repetitions=repetitions
            ),
            "python_predicate_evaluation": _stage_timeit(
                evaluate_predicate, warmups=warmups, repetitions=repetitions
            ),
            "vector_scoring_top_k": _stage_timeit(
                score_and_select, warmups=warmups, repetitions=repetitions
            ),
        }
        report: dict[str, Any] = dict(stage_results)
        report["total_p50_ms"] = sum(
            timing["latency_p50_ms"] for timing in stage_results.values()
        )
        return report

    for name, filters in typed_filters.items():
        results[f"search_{name}_filtered"]["stage_timings_ms"] = stage_timings(
            filters
        )
    return {
        "record_count": record_count,
        "dim": dim,
        "seed": seed,
        "candidate_count": candidate_counts["candidate_small"],
        "candidate_counts": candidate_counts,
        "warmups": warmups,
        "repetitions": repetitions,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=int, default=20_000)
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--dimensions", type=int, nargs="+")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repetitions", type=int, default=7)
    args = parser.parse_args()
    dimensions = args.dimensions or [args.dim]
    if (
        args.records < 1
        or any(dim < 1 for dim in dimensions)
        or args.repetitions < 1
    ):
        parser.error("--records, --dimensions, and --repetitions must be positive")
    results = [
        run_benchmark(
            args.records,
            dim,
            seed=args.seed,
            repetitions=args.repetitions,
        )
        for dim in dimensions
    ]
    print(
        json.dumps(results[0] if args.dimensions is None else results, indent=2)
    )


if __name__ == "__main__":
    main()
