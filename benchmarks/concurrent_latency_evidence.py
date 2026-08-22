"""Measure serial and concurrent local retrieval without a pass/fail threshold."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from benchmarks.evaluate_labeled_retrieval import DEFAULT_FIXTURE, load_labeled_fixture
from searchkernel.domain import Record
from searchkernel.eval import BenchmarkConfig, EvalReport, SearchExecution, run_eval
from searchkernel.eval.golden import GoldenSet
from searchkernel.indices import LocalRecordBackend


def _make_search(
    records: list[Record], golden_set: GoldenSet, db_path: Path | None, k: int
):
    backend = LocalRecordBackend(db_path)
    backend.index(records)
    entries = {entry.query: entry for entry in golden_set}

    def search(query: str) -> SearchExecution:
        entry = entries[query]
        filters = {"workspace_id": entry.workspace_id} if entry.workspace_id else None
        hits = backend.search_keyword(query, k, filters)
        return SearchExecution(
            ids=tuple(hit.source_id for hit in hits),
            source_kinds={hit.source_id: hit.source_kind for hit in hits},
        )

    return search, backend


def _run(
    golden_set: GoldenSet,
    search,
    *,
    k: int,
    concurrency: int,
    repetitions: int,
    warmup_count: int,
) -> EvalReport:
    return run_eval(
        golden_set,
        search,
        k=k,
        config=BenchmarkConfig(
            warmup_count=warmup_count,
            measured_repetitions=repetitions,
            concurrency=concurrency,
            corpus_version="searchkernel-labeled-v1",
            backend="sqlite-fts5",
            metadata={"evidence": "serial-vs-concurrent"},
        ),
    )


def _quality_rows(report: EvalReport) -> list[tuple[Any, ...]]:
    return [
        (
            metric.query,
            metric.repetition,
            metric.recall_at_k,
            metric.ndcg_at_k,
            metric.mrr,
            metric.ap,
            metric.empty_result,
        )
        for metric in report.metrics
    ]


def measure_concurrent_latency(
    fixture: Path = DEFAULT_FIXTURE,
    *,
    k: int = 3,
    repetitions: int = 3,
    concurrent_workers: int = 4,
    warmup_count: int = 2,
) -> dict[str, Any]:
    """Return comparable serial/concurrent reports and observed deltas."""
    records, golden_set = load_labeled_fixture(fixture)

    def measure_backend(db_path: Path | None, backend_name: str) -> dict[str, Any]:
        search, backend = _make_search(records, golden_set, db_path, k)
        try:
            serial = _run(
                golden_set,
                search,
                k=k,
                concurrency=1,
                repetitions=repetitions,
                warmup_count=warmup_count,
            )
            concurrent = _run(
                golden_set,
                search,
                k=k,
                concurrency=concurrent_workers,
                repetitions=repetitions,
                warmup_count=warmup_count,
            )
        finally:
            backend.close()
        return {
            "backend": backend_name,
            "serial": serial.to_dict(),
            "concurrent": concurrent.to_dict(),
            "quality_equivalent": _quality_rows(serial) == _quality_rows(concurrent),
            "latency_p95_delta_ms": (
                concurrent.latency_p95_ms - serial.latency_p95_ms
                if concurrent.latency_p95_ms is not None
                and serial.latency_p95_ms is not None
                else None
            ),
            "latency_p99_delta_ms": (
                concurrent.latency_p99_ms - serial.latency_p99_ms
                if concurrent.latency_p99_ms is not None
                and serial.latency_p99_ms is not None
                else None
            ),
        }

    with tempfile.TemporaryDirectory(prefix="searchkernel-latency-") as directory:
        file_backed = measure_backend(Path(directory) / "records.db", "file-backed")
    in_memory = measure_backend(None, "in-memory")

    return {
        "schema_version": 2,
        "fixture": str(fixture),
        "serial": file_backed["serial"],
        "concurrent": file_backed["concurrent"],
        "quality_equivalent": file_backed["quality_equivalent"],
        "latency_p95_delta_ms": file_backed["latency_p95_delta_ms"],
        "backends": {"file_backed": file_backed, "in_memory": in_memory},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmup-count", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    json.dump(
        measure_concurrent_latency(
            args.fixture,
            k=args.k,
            repetitions=args.repetitions,
            concurrent_workers=args.workers,
            warmup_count=args.warmup_count,
        ),
        sys.stdout,
        indent=2,
    )
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
