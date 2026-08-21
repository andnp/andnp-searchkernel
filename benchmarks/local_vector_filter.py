"""Measure local vector snapshot filtering with a generic parity oracle.

The deterministic workload builds 100,000 two-dimensional rows with four
status values per eight workspaces and two source kinds. Each measured case
warms twice, reports p50 and p95 over repeated masks, and verifies that the
snapshot result keys match the generic predicate oracle. Run this script before
and after an implementation change; it intentionally has no CI latency gate.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from typing import Any

import numpy as np

from searchkernel.domain import RecordIdentity
from searchkernel.domain.vector_filters import compile_vector_filters
from searchkernel.indices.local_vectors import PackedVectorCodec, VectorSnapshot


def _percentile(samples: list[float], percentile: float) -> float:
    ordered = sorted(samples)
    position = min(len(ordered) - 1, int(len(ordered) * percentile))
    return ordered[position]


def _snapshot(record_count: int) -> VectorSnapshot:
    payload = PackedVectorCodec.encode([1.0, 0.0], 2)
    rows = []
    for index in range(record_count):
        workspace_id = f"workspace-{index % 8}"
        source_kind = "note" if index % 2 == 0 else "commit"
        status = "archived" if index % 4 == 0 else "active"
        storage_key = RecordIdentity(
            workspace_id,
            source_kind,
            f"record-{index}",
        ).storage_key
        rows.append(
            {
                "storage_key": storage_key,
                "source_id": f"record-{index}",
                "workspace_id": workspace_id,
                "source_kind": source_kind,
                "status": status,
                "metadata": {},
                "uri": None,
                "embedding": payload,
                "format_version": 2,
                "normalization_policy": "l2",
            }
        )
    return VectorSnapshot.from_rows(
        rows,
        encoder_namespace="benchmark",
        dim=2,
        epoch=1,
    )


def _generic_mask(
    snapshot: VectorSnapshot,
    filters: dict[str, object],
) -> np.ndarray:
    predicate = compile_vector_filters(filters)
    metadata = snapshot.metadata or ({} for _ in snapshot.storage_keys)
    uris = snapshot.uris or (None for _ in snapshot.storage_keys)
    return np.asarray(
        [
            predicate.matches(
                storage_key=storage_key,
                source_id=str(source_id),
                workspace_id=(
                    str(workspace_id) if workspace_id is not None else None
                ),
                source_kind=str(source_kind),
                status=str(status),
                metadata=metadata,
                uri=uri,
            )
            for storage_key, source_id, workspace_id, source_kind, status, metadata, uri in zip(
                snapshot.storage_keys,
                snapshot.source_ids,
                snapshot.workspace_ids,
                snapshot.source_kinds,
                snapshot.statuses,
                metadata,
                uris,
                strict=True,
            )
        ],
        dtype=bool,
    )


def _measure(
    snapshot: VectorSnapshot,
    filters: dict[str, object],
    *,
    repetitions: int,
    generic: bool,
) -> tuple[dict[str, float], tuple[str, ...]]:
    if generic:
        def measure() -> np.ndarray:
            return _generic_mask(snapshot, filters)
    else:
        def measure() -> np.ndarray:
            return snapshot.filter_mask(
                filters,
                status_values=set(),
                filter_values=None,
            )
    for _ in range(2):
        measure()
    samples: list[float] = []
    mask = np.empty(0, dtype=bool)
    for _ in range(repetitions):
        started = time.perf_counter()
        mask = measure()
        samples.append((time.perf_counter() - started) * 1_000)
    keys = tuple(
        storage_key
        for storage_key, eligible in zip(snapshot.storage_keys, mask, strict=True)
        if eligible
    )
    return {
        "latency_p50_ms": statistics.median(samples),
        "latency_p95_ms": _percentile(samples, 0.95),
    }, keys


def run_benchmark(record_count: int, repetitions: int) -> dict[str, Any]:
    """Return scalar filter timings and generic result-key parity."""
    snapshot = _snapshot(record_count)
    filters_by_name = {
        "default_active": {},
        "scalar_combination": {
            "statuses": ["active"],
            "workspace_id": "workspace-2",
            "source_kind": "note",
        },
    }
    results: dict[str, Any] = {}
    for name, filters in filters_by_name.items():
        fast, fast_keys = _measure(
            snapshot, filters, repetitions=repetitions, generic=False
        )
        generic, generic_keys = _measure(
            snapshot, filters, repetitions=repetitions, generic=True
        )
        results[name] = {
            "fast": fast,
            "generic_reference": generic,
            "eligible_count": len(fast_keys),
            "result_key_parity": fast_keys == generic_keys,
        }
        if fast_keys != generic_keys:
            raise RuntimeError(f"result key parity failed for {name}")
    return {
        "record_count": record_count,
        "warmup_count": 2,
        "repetitions": repetitions,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=int, default=100_000)
    parser.add_argument("--repetitions", type=int, default=10)
    args = parser.parse_args()
    if args.records < 1 or args.repetitions < 1:
        parser.error("--records and --repetitions must be positive")
    print(json.dumps(run_benchmark(args.records, args.repetitions), indent=2))


if __name__ == "__main__":
    main()
