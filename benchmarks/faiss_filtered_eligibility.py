"""Measure the canonical filtered FAISS eligibility-mask optimization.

The benchmark compares the optimized exact path with the pre-mask validation
path for a repeated ``workspace_id`` plus ``source_kind`` filter. It reports
the first filtered call separately from warmed calls because the mask is lazy.
Metadata composites and approximate filtering remain control paths.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import numpy as np

from searchkernel.domain import Record
from searchkernel.domain.vector_filters import CompiledVectorFilter
from searchkernel.indices import FAISSLocalVectorStore, LocalRecordBackend
from searchkernel.indices.faiss_local import _FAISSState
from searchkernel.indices.local_vectors import PackedVectorCodec

_TIMESTAMP = datetime(2026, 1, 1, tzinfo=UTC)
_MODEL_NAME = "benchmark-faiss-filtered-eligibility-v1"
_UPSERT_BATCH_SIZE = 2_000
_FILTERS = {"workspace_id": "workspace-2", "source_kind": "note"}


class _GenericValidationStore(FAISSLocalVectorStore):
    """Reference store that disables only the eligibility-mask fast path."""

    def _eligibility_mask(
        self,
        state: _FAISSState,
        predicate: CompiledVectorFilter,
    ) -> None:
        return None


def _percentile(samples: list[float], percentile: float) -> float:
    ordered = sorted(samples)
    return ordered[min(len(ordered) - 1, int(len(ordered) * percentile))]


def _time_search(
    search: Callable[[], object],
    *,
    warmups: int,
    repetitions: int,
) -> dict[str, float]:
    for _ in range(warmups):
        search()
    samples: list[float] = []
    for _ in range(repetitions):
        started = time.perf_counter()
        search()
        samples.append((time.perf_counter() - started) * 1_000)
    return {
        "p50_ms": statistics.median(samples),
        "p95_ms": _percentile(samples, 0.95),
        "p99_ms": _percentile(samples, 0.99),
    }


def _build_corpus(
    record_count: int,
    dim: int,
    seed: int,
) -> tuple[LocalRecordBackend, list[float], int]:
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
    eligible_count = sum(
        record.workspace_id == _FILTERS["workspace_id"]
        and record.source_kind == _FILTERS["source_kind"]
        for record in records
    )
    return backend, query.tolist(), eligible_count


def _measure_exact_variant(
    store_type: type[FAISSLocalVectorStore],
    backend: LocalRecordBackend,
    query: list[float],
    dim: int,
    *,
    repetitions: int,
) -> dict[str, Any]:
    store = store_type(backend)
    store.search(query, 10, model_name=_MODEL_NAME, dim=dim)
    first_started = time.perf_counter()
    first_hits = store.search(
        query,
        10,
        model_name=_MODEL_NAME,
        dim=dim,
        filters=_FILTERS,
    )
    first_filter_ms = (time.perf_counter() - first_started) * 1_000
    warmed = _time_search(
        lambda: store.search(
            query,
            10,
            model_name=_MODEL_NAME,
            dim=dim,
            filters=_FILTERS,
        ),
        warmups=3,
        repetitions=repetitions,
    )
    state = next(iter(store._states.values()))
    mask = state.eligibility_masks.get(("workspace-2", "note"))
    return {
        "first_filter_ms": first_filter_ms,
        "warmed": warmed,
        "returned": len(first_hits),
        "returned_keys": [hit.storage_key for hit in first_hits],
        "mask_entries": None if mask is None else len(mask),
        "mask_storage_bytes_estimate": (
            None if mask is None else sys.getsizeof(mask) + len(mask) * 8
        ),
    }


def run_benchmark(
    record_count: int,
    dim: int,
    *,
    seed: int = 17,
    repetitions: int = 10,
) -> dict[str, Any]:
    """Measure exact mask benefit plus composite and approximate controls."""
    backend, query, eligible_count = _build_corpus(record_count, dim, seed)
    expected = backend.search_vector(
        query, 10, model_name=_MODEL_NAME, dim=dim, filters=_FILTERS
    )
    masked = _measure_exact_variant(
        FAISSLocalVectorStore,
        backend,
        query,
        dim,
        repetitions=repetitions,
    )
    generic = _measure_exact_variant(
        _GenericValidationStore,
        backend,
        query,
        dim,
        repetitions=repetitions,
    )
    expected_keys = [hit.storage_key for hit in expected]
    masked["same_results"] = masked["returned_keys"] == expected_keys
    generic["same_results"] = generic["returned_keys"] == expected_keys
    generic_p50 = generic["warmed"]["p50_ms"]
    masked_p50 = masked["warmed"]["p50_ms"]
    approximate = FAISSLocalVectorStore(
        backend,
        search_strategy="approximate",
        max_scan_candidates=100_000,
    )
    approximate.search(
        query,
        10,
        model_name=_MODEL_NAME,
        dim=dim,
        filters=_FILTERS,
    )
    diagnostics = approximate.last_search_diagnostics
    composite_filters = {
        **_FILTERS,
        "metadata_equals": {"project_id": "p3"},
    }
    composite_expected = backend.search_vector(
        query,
        10,
        model_name=_MODEL_NAME,
        dim=dim,
        filters=composite_filters,
    )
    composite_store = FAISSLocalVectorStore(backend)
    composite_store.search(query, 10, model_name=_MODEL_NAME, dim=dim)
    composite_warmed = _time_search(
        lambda: composite_store.search(
            query,
            10,
            model_name=_MODEL_NAME,
            dim=dim,
            filters=composite_filters,
        ),
        warmups=3,
        repetitions=repetitions,
    )
    composite_hits = composite_store.search(
        query,
        10,
        model_name=_MODEL_NAME,
        dim=dim,
        filters=composite_filters,
    )
    return {
        "record_count": record_count,
        "dimension": dim,
        "seed": seed,
        "eligible_count": eligible_count,
        "eligible_fraction": eligible_count / record_count,
        "filters": _FILTERS,
        "generic_validation": generic,
        "masked_validation": masked,
        "composite_control": {
            "warmed": composite_warmed,
            "same_results": [hit.storage_key for hit in composite_hits]
            == [hit.storage_key for hit in composite_expected],
        },
        "warmed_p50_improvement_percent": (
            (generic_p50 - masked_p50) / generic_p50 * 100.0
        ),
        "approximate_control": {
            "candidate_budget": diagnostics.get("candidate_budget"),
            "scan_limit": diagnostics.get("scan_limit"),
            "under_returned": diagnostics.get("under_returned"),
            "fallback": diagnostics.get("fallback"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=int, nargs="+", default=[20_000, 100_000])
    parser.add_argument("--dimensions", type=int, nargs="+", default=[32])
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--repetitions", type=int, default=10)
    args = parser.parse_args()
    if (
        any(value < 1 for value in args.records)
        or any(value < 1 for value in args.dimensions)
        or args.repetitions < 1
    ):
        parser.error("records, dimensions, and repetitions must be positive")
    results = [
        run_benchmark(
            record_count,
            dim,
            seed=args.seed,
            repetitions=args.repetitions,
        )
        for record_count in args.records
        for dim in args.dimensions
    ]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
