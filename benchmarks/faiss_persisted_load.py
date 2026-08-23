"""Measure staged, post-copy, and warm loading of a persisted FAISS artifact.

The supplied database and artifact are copied to a temporary directory before
the benchmark opens them. The result reports that staging time separately from
the first process/page-cache load after staging; it does not claim cold-storage
latency. A stale or corrupt artifact can therefore exercise SearchKernel's
rebuild fallback without modifying the caller's files.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import tempfile
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from searchkernel.indices import FAISSLocalVectorStore, LocalRecordBackend


def _rss_bytes() -> int | None:
    """Return current resident memory when the host exposes it."""
    status_path = Path("/proc/self/status")
    if not status_path.exists():
        return None
    for line in status_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    return None


def _copy_database(source: Path, target: Path) -> None:
    """Copy a SQLite database and any sidecars needed for a consistent read."""
    shutil.copy2(source, target)
    for suffix in ("-wal", "-shm"):
        sidecar = source.with_name(source.name + suffix)
        if sidecar.exists():
            shutil.copy2(sidecar, target.with_name(target.name + suffix))


def _copy_artifact(source: Path, target: Path) -> None:
    """Copy a FAISS file artifact or directory artifact to a temporary path."""
    if source.is_dir():
        shutil.copytree(source, target)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    metadata = source.with_suffix(".json")
    if metadata.exists():
        shutil.copy2(metadata, target.with_suffix(".json"))


def _percentile(samples: list[float], percentile: float) -> float:
    ordered = sorted(samples)
    index = min(len(ordered) - 1, int(len(ordered) * percentile))
    return ordered[index]


def _measure(
    call: Callable[[], object],
    repetitions: int,
) -> dict[str, float] | None:
    """Measure optional repeated calls without imposing a latency threshold."""
    if repetitions < 1:
        return None
    samples: list[float] = []
    for _ in range(repetitions):
        started = time.perf_counter()
        call()
        samples.append((time.perf_counter() - started) * 1_000)
    return {
        "count": float(len(samples)),
        "p50_ms": statistics.median(samples),
        "p95_ms": _percentile(samples, 0.95),
        "p99_ms": _percentile(samples, 0.99),
    }


def _query_vector(raw: str | None, dim: int) -> list[float]:
    if raw is None:
        return [1.0] + [0.0] * (dim - 1)
    values = [float(value) for value in raw.split(",") if value.strip()]
    if len(values) != dim:
        raise ValueError(f"query must contain exactly {dim} values")
    return values


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--query", help="Comma-separated query vector; defaults to e1")
    parser.add_argument("--search-strategy", choices=("exact", "approximate"), default="approximate")
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--hnsw-ef-construction", type=int, default=40)
    parser.add_argument("--hnsw-ef-search", type=int, default=64)
    parser.add_argument("--warm-repetitions", type=int, default=0)
    parser.add_argument("--filtered-repetitions", type=int, default=0)
    parser.add_argument("--filters-json", help="JSON object for optional filtered measurements")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run one isolated benchmark and return JSON-compatible measurements."""
    if args.dim < 1 or args.k < 1:
        raise ValueError("dim and k must be positive")
    if not args.artifact.exists() or not args.database.is_file():
        raise FileNotFoundError("artifact and database paths must exist")
    filters: Mapping[str, Any] | None = None
    if args.filters_json is not None:
        decoded = json.loads(args.filters_json)
        if not isinstance(decoded, dict):
            raise ValueError("filters-json must be a JSON object")
        filters = decoded
    query = _query_vector(args.query, args.dim)
    with tempfile.TemporaryDirectory(prefix="searchkernel-faiss-load-") as directory:
        root = Path(directory)
        database = root / args.database.name
        artifact = root / args.artifact.name
        copy_started = time.perf_counter()
        _copy_database(args.database, database)
        _copy_artifact(args.artifact, artifact)
        copy_ms = (time.perf_counter() - copy_started) * 1_000
        backend = LocalRecordBackend(database)
        store = FAISSLocalVectorStore(
            backend,
            index_path=artifact,
            search_strategy=args.search_strategy,
            hnsw_m=args.hnsw_m,
            hnsw_ef_construction=args.hnsw_ef_construction,
            hnsw_ef_search=args.hnsw_ef_search,
        )
        rss_before = _rss_bytes()
        started = time.perf_counter()
        hits = store.search(
            query,
            args.k,
            model_name=args.model_name,
            dim=args.dim,
        )
        post_copy_process_load_ms = (time.perf_counter() - started) * 1_000
        rss_after = _rss_bytes()
        result: dict[str, Any] = {
            "copy_ms": copy_ms,
            "post_copy_process_load_ms": post_copy_process_load_ms,
            "rss_before_bytes": rss_before,
            "rss_after_bytes": rss_after,
            "diagnostics": store.last_search_diagnostics,
            "returned": len(hits),
            "top_storage_keys": [hit.storage_key for hit in hits[: args.k]],
            "warm": _measure(
                lambda: store.search(
                    query, args.k, model_name=args.model_name, dim=args.dim
                ),
                args.warm_repetitions,
            ),
            "filtered": None,
        }
        if filters is not None and args.filtered_repetitions > 0:
            result["filtered"] = _measure(
                lambda: store.search(
                    query,
                    args.k,
                    model_name=args.model_name,
                    dim=args.dim,
                    filters=filters,
                ),
                args.filtered_repetitions,
            )
        return result


def main() -> None:
    args = _parser().parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
