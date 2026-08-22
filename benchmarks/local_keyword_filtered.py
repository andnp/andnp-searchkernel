"""Measure SQLite keyword retrieval with a selective project filter."""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from searchkernel.domain import Record
from searchkernel.indices import LocalRecordBackend


def _records(count: int) -> list[Record]:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    return [
        Record(
            source_kind="note",
            source_id=f"record-{index}",
            title=f"Filtered retrieval {index}",
            body="benchmark keyword retrieval body",
            created_at=timestamp,
            updated_at=timestamp,
            uri=f"projects/{index % 10}/guide.md",
            metadata={"project_id": f"project-{index % 10}"},
        )
        for index in range(count)
    ]


def _percentile(samples: list[float], percentile: float) -> float:
    ordered = sorted(samples)
    position = min(len(ordered) - 1, int(len(ordered) * percentile))
    return ordered[position]


def _measure(
    backend: LocalRecordBackend,
    *,
    repetitions: int,
    filters: dict[str, object] | None,
) -> dict[str, Any]:
    for _ in range(2):
        backend.search_keyword("benchmark retrieval", 10, filters)
    samples: list[float] = []
    hits = []
    for _ in range(repetitions):
        started = time.perf_counter()
        hits = backend.search_keyword("benchmark retrieval", 10, filters)
        samples.append((time.perf_counter() - started) * 1_000)
    return {
        "hit_count": len(hits),
        "latency_p50_ms": statistics.median(samples),
        "latency_p95_ms": _percentile(samples, 0.95),
        "latency_p99_ms": _percentile(samples, 0.99),
    }


def run_benchmark(record_count: int, repetitions: int) -> dict[str, Any]:
    """Build a deterministic corpus and report filtered keyword timings."""
    records = _records(record_count)
    with tempfile.TemporaryDirectory(prefix="searchkernel-keyword-") as directory:
        backend = LocalRecordBackend(Path(directory) / "records.db")
        backend.index(records)
        filters_by_name = {
            "unfiltered": None,
            "workspace_scalar": {"workspace_id": "workspace-1"},
            "source_scalar": {"source_kind": "note"},
            "metadata": {"project_filter": ["project-1"]},
            "scalar_and_metadata": {
                "workspace_id": "workspace-1",
                "project_filter": ["project-1"],
            },
        }
        cases = {
            name: _measure(backend, repetitions=repetitions, filters=filters)
            for name, filters in filters_by_name.items()
        }
        backend.close()
    expected_filtered = sum(
        record.metadata["project_id"] == "project-1" for record in records
    )
    if cases["metadata"]["hit_count"] != min(10, expected_filtered):
        raise RuntimeError("filtered keyword retrieval returned an incorrect hit count")
    return {
        "record_count": record_count,
        "eligible_count": expected_filtered,
        "filtered": cases["metadata"],
        "unfiltered": cases["unfiltered"],
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=int, default=10_000)
    parser.add_argument("--repetitions", type=int, default=10)
    args = parser.parse_args()
    if args.records < 1 or args.repetitions < 1:
        parser.error("--records and --repetitions must be positive")
    print(json.dumps(run_benchmark(args.records, args.repetitions), indent=2))


if __name__ == "__main__":
    main()
