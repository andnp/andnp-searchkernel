"""Run the repository's deterministic performance acceptance matrix.

The matrix composes existing benchmark harnesses so one artifact contains the
local scale, cache, and filter-parity evidence used for regression review.
Latency and resource values are observations; only existing structural and
quality gates are evaluated here.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from benchmarks.candidate_cache_evidence import run_benchmark as run_cache_benchmark
from benchmarks.local_scale_gate import load_config, run_gate
from benchmarks.local_vector_filter import run_benchmark as run_filter_benchmark


def _scale_acceptance(report: dict[str, Any]) -> dict[str, Any]:
    results = report["results"]
    persistence = {
        result["size"]: {
            "index_size_bytes": result["benchmark"]["metadata"].get(
                "index_size_bytes"
            ),
            "build_time_ms": result["benchmark"]["metadata"].get(
                "index_build_time_ms"
            ),
            "load_time_ms": result["benchmark"]["metadata"].get(
                "index_load_time_ms"
            ),
        }
        for result in results
    }
    return {
        "gate": report["gate"],
        "cold_warm_latency": {
            result["size"]: {
                "cold": result["benchmark"]["cold"],
                "warm": result["benchmark"]["warm"],
            }
            for result in results
        },
        "resource_scaling": {
            result["size"]: {
                key: result["benchmark"]["metadata"].get(key)
                for key in (
                    "rss_before_index_load_bytes",
                    "rss_after_index_load_bytes",
                    "rss_peak_query_bytes",
                    "index_size_bytes",
                )
            }
            for result in results
        },
        "recall": {
            result["size"]: result["ann"] for result in results
        },
        "persistence": persistence,
    }


def _cache_acceptance(report: dict[str, Any]) -> dict[str, Any]:
    results = report["results"]
    return {
        str(count): {
            "warm_cache_hit": result["warm_cache_hit"],
            "local_retrieval": result["local_retrieval"],
            "mutation_isolated": result["mutation_isolated"],
        }
        for count, result in results.items()
    }


def _filter_acceptance(report: dict[str, Any]) -> dict[str, Any]:
    return {
        name: {
            "eligible_count": result["eligible_count"],
            "result_key_parity": result["result_key_parity"],
            "fast": result["fast"],
            "generic_reference": result["generic_reference"],
        }
        for name, result in report["results"].items()
    }


async def run_matrix(
    *,
    scale_config: Path,
    scale_work_dir: Path,
    cache_repetitions: int,
    filter_records: int,
    filter_repetitions: int,
) -> dict[str, Any]:
    """Run each acceptance workload with deterministic input settings."""
    if cache_repetitions < 1 or filter_records < 1 or filter_repetitions < 1:
        raise ValueError("matrix repetitions and record count must be positive")
    scale_report = run_gate(load_config(scale_config), scale_work_dir)
    cache_report = await run_cache_benchmark(cache_repetitions)
    filter_report = run_filter_benchmark(filter_records, filter_repetitions)
    return {
        "schema_version": 1,
        "threshold_policy": {
            "latency": "observation_only",
            "resources": "observation_only",
            "scale_gate": "existing_local_scale_gate",
            "ann_recall": "existing_local_scale_gate_threshold",
            "filter_parity": "required",
            "cache_isolation": "required",
        },
        "scale": _scale_acceptance(scale_report),
        "cache": _cache_acceptance(cache_report),
        "filter_parity": _filter_acceptance(filter_report),
    }


def _validate_matrix(report: dict[str, Any]) -> None:
    if not report["scale"]["gate"]["passed"]:
        raise RuntimeError("local scale acceptance gate failed")
    if not all(
        result["mutation_isolated"]
        for result in report["cache"].values()
    ):
        raise RuntimeError("cache mutation isolation evidence failed")
    if not all(
        result["result_key_parity"]
        for result in report["filter_parity"].values()
    ):
        raise RuntimeError("filter result-key parity evidence failed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scale-config", type=Path, default=Path(__file__).with_name("local_scale_gate.json")
    )
    parser.add_argument("--scale-work-dir", type=Path, default=Path(".performance-matrix-work"))
    parser.add_argument("--cache-repetitions", type=int, default=200)
    parser.add_argument("--filter-records", type=int, default=100_000)
    parser.add_argument("--filter-repetitions", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = asyncio.run(
        run_matrix(
            scale_config=args.scale_config,
            scale_work_dir=args.scale_work_dir,
            cache_repetitions=args.cache_repetitions,
            filter_records=args.filter_records,
            filter_repetitions=args.filter_repetitions,
        )
    )
    _validate_matrix(report)
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(f"{payload}\n")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
