"""Run the reproducible local vector scale gate."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from searchkernel.domain import Record
from searchkernel.eval import (
    BenchmarkConfig,
    BenchmarkHooks,
    SearchExecution,
    make_synthetic_corpus,
    run_benchmark,
)
from searchkernel.indices import FAISSLocalVectorStore, LocalRecordBackend
from searchkernel.runtime.trace import QueryTrace

_TOPICS = (
    "api",
    "cache",
    "config",
    "database",
    "embedding",
    "graph",
    "index",
    "latency",
    "metadata",
    "pipeline",
    "query",
    "ranking",
    "relevance",
    "storage",
    "trace",
    "vector",
)
_EXPECTED_COUNTS = {"1k": 1_000, "10k": 10_000, "100k": 100_000}


@dataclass(frozen=True)
class AnnConfig:
    """Optional ANN acceptance settings."""

    enabled: bool = True
    required: bool = False
    min_recall_at_k: float = 0.9


@dataclass(frozen=True)
class ScaleGateConfig:
    """Reproducible settings for the local scale gate."""

    sizes: tuple[str, ...] = ("1k", "10k", "100k")
    seed: int = 0
    dimension: int = 32
    model_name: str = "synthetic-local-v1"
    k: int = 10
    warmup_count: int = 1
    measured_repetitions: int = 3
    concurrency: int = 1
    ann: AnnConfig = AnnConfig()

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> ScaleGateConfig:
        """Load and validate JSON configuration."""
        ann_values = values.get("ann", {})
        if not isinstance(ann_values, dict):
            raise TypeError("ann must be an object")
        ann = AnnConfig(
            enabled=bool(ann_values.get("enabled", True)),
            required=bool(ann_values.get("required", False)),
            min_recall_at_k=float(ann_values.get("min_recall_at_k", 0.9)),
        )
        sizes = tuple(str(size) for size in values.get("sizes", cls.sizes))
        config = cls(
            sizes=sizes,
            seed=int(values.get("seed", 0)),
            dimension=int(values.get("dimension", 32)),
            model_name=str(values.get("model_name", cls.model_name)),
            k=int(values.get("k", 10)),
            warmup_count=int(values.get("warmup_count", 1)),
            measured_repetitions=int(values.get("measured_repetitions", 3)),
            concurrency=int(values.get("concurrency", 1)),
            ann=ann,
        )
        if not config.sizes:
            raise ValueError("sizes must not be empty")
        if config.dimension <= len(_TOPICS):
            raise ValueError(f"dimension must be greater than {len(_TOPICS)}")
        if config.k < 1:
            raise ValueError("k must be positive")
        if not 0.0 <= config.ann.min_recall_at_k <= 1.0:
            raise ValueError("ann.min_recall_at_k must be between 0 and 1")
        return config


def load_config(path: Path) -> ScaleGateConfig:
    """Load a scale gate configuration from JSON."""
    values = json.loads(path.read_text())
    if not isinstance(values, dict):
        raise TypeError("scale gate configuration must be a JSON object")
    return ScaleGateConfig.from_dict(values)


def _topic_vector(topic: str, dimension: int) -> list[float]:
    """Return the deterministic query vector for one synthetic topic."""
    try:
        topic_index = _TOPICS.index(topic)
    except ValueError as error:
        raise ValueError(f"unknown synthetic topic: {topic}") from error
    vector = [0.0] * dimension
    vector[topic_index] = 1.0
    vector[-1] = -0.01
    return vector


def _with_embeddings(records: list[Record], dimension: int) -> list[Record]:
    """Attach deterministic one-hot topic embeddings to synthetic records."""
    for record in records:
        topic = str(record.metadata["topic"])
        vector = _topic_vector(topic, dimension)
        ordinal = int(record.metadata["ordinal"])
        vector[-1] = 0.01 * math.log1p(ordinal // len(_TOPICS))
        record.embedding = vector
    return records


def _query_topic(query: str) -> str:
    """Extract a synthetic topic query."""
    prefix, separator, topic = query.partition(":")
    if prefix != "topic" or not separator:
        raise ValueError(f"unexpected synthetic query: {query}")
    return topic


def _file_size(path: Path) -> int:
    """Return SQLite database size including WAL sidecars."""
    total = 0
    for candidate in (
        path,
        path.with_name(f"{path.name}-wal"),
        path.with_name(f"{path.name}-shm"),
    ):
        if candidate.exists():
            total += candidate.stat().st_size
    return total


def _run_size(
    config: ScaleGateConfig,
    size: str,
    work_dir: Path,
) -> dict[str, Any]:
    """Build and measure one local exact-vector corpus."""
    corpus = make_synthetic_corpus(size, seed=config.seed)
    records = _with_embeddings(corpus.records, config.dimension)
    db_path = work_dir / f"{size}.sqlite"
    faiss_path = work_dir / f"{size}-faiss"
    state: dict[str, Any] = {"backend": None}

    def build_index() -> None:
        backend = LocalRecordBackend(db_path)
        backend.upsert(records, config.model_name, config.dimension)
        state["backend"] = backend

    def load_index() -> None:
        previous = state["backend"]
        if previous is not None:
            close = getattr(previous.db_manager, "close", None)
            if callable(close):
                close()
        state["backend"] = LocalRecordBackend(db_path)

    def search(query: str, *, trace: QueryTrace | None = None) -> SearchExecution:
        backend: LocalRecordBackend = state["backend"]
        vector = _topic_vector(_query_topic(query), config.dimension)
        hits = backend.search_vector(
            vector,
            config.k,
            model_name=config.model_name,
            dim=config.dimension,
        )
        return SearchExecution(
            tuple(hit.source_id for hit in hits),
            trace=trace,
        )

    benchmark = run_benchmark(
        corpus.golden_set,
        search,
        k=config.k,
        config=BenchmarkConfig(
            warmup_count=config.warmup_count,
            measured_repetitions=config.measured_repetitions,
            concurrency=config.concurrency,
            corpus_version=corpus.version,
            backend="sqlite-exact",
            model_fingerprint=config.model_name,
            metadata={
                "corpus_size": corpus.size,
                "dimension": config.dimension,
                "seed": config.seed,
            },
        ),
        hooks=BenchmarkHooks(
            build_index=build_index,
            load_index=load_index,
            index_size_bytes=lambda: _file_size(db_path),
        ),
    )

    backend: LocalRecordBackend = state["backend"]
    backend.db_manager.get_connection().execute("PRAGMA wal_checkpoint(TRUNCATE)")
    ann_result: dict[str, Any] = {"status": "disabled"}
    if config.ann.enabled:
        try:
            importlib.import_module("faiss")
        except ImportError:
            ann_result = {
                "status": "unavailable",
                "reason": "faiss is not installed",
            }
        else:
            ann = FAISSLocalVectorStore(backend, index_path=faiss_path)
            recalls = [
                ann.verify_recall(
                    _topic_vector(topic, config.dimension),
                    config.k,
                    model_name=config.model_name,
                    dim=config.dimension,
                )
                for topic in _TOPICS[:4]
            ]
            ann_result = {
                "status": "available",
                "recall_at_k": min(recalls),
                "recall_samples": recalls,
                "index_size_bytes": sum(
                    path.stat().st_size
                    for path in faiss_path.glob("*")
                    if path.is_file()
                ),
            }

    result = {
        "size": size,
        "record_count": corpus.size,
        "corpus_version": corpus.version,
        "dimension": config.dimension,
        "backend": "sqlite-exact",
        "index_size_bytes": _file_size(db_path),
        "benchmark": benchmark.to_dict(),
        "ann": ann_result,
    }
    close = getattr(backend.db_manager, "close", None)
    if callable(close):
        close()
    shutil.rmtree(faiss_path, ignore_errors=True)
    db_path.unlink(missing_ok=True)
    db_path.with_name(f"{db_path.name}-wal").unlink(missing_ok=True)
    db_path.with_name(f"{db_path.name}-shm").unlink(missing_ok=True)
    return result


def evaluate_gate(
    results: list[dict[str, Any]],
    config: ScaleGateConfig,
) -> dict[str, Any]:
    """Validate required evidence without inventing performance claims."""
    failures: list[str] = []
    warnings: list[str] = []
    by_size = {result["size"]: result for result in results}
    for size in config.sizes:
        result = by_size.get(size)
        if result is None:
            failures.append(f"{size}: result is missing")
            continue
        expected_count = _EXPECTED_COUNTS.get(size)
        if expected_count is not None and result.get("record_count") != expected_count:
            failures.append(
                f"{size}: record_count={result.get('record_count')} "
                f"does not match {expected_count}"
            )
        benchmark = result["benchmark"]
        metadata = benchmark["metadata"]
        warm = benchmark["warm"]
        for field in (
            "rss_before_index_load_bytes",
            "rss_after_index_load_bytes",
            "rss_peak_query_bytes",
            "index_size_bytes",
        ):
            value = metadata.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                failures.append(f"{size}: {field} evidence is unavailable")
            elif value <= 0:
                failures.append(f"{size}: {field} must be positive")
        for field in ("latency_p50_ms", "latency_p95_ms", "latency_p99_ms", "qps"):
            value = warm.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                failures.append(f"{size}: warm.{field} evidence is unavailable")
            elif value <= 0:
                failures.append(f"{size}: warm.{field} must be positive")
        ann = result["ann"]
        if ann["status"] == "disabled":
            if config.ann.required:
                failures.append(f"{size}: optional FAISS ANN is disabled")
        elif ann["status"] == "unavailable":
            warning = f"{size}: optional FAISS ANN unavailable ({ann['reason']})"
            warnings.append(warning)
            if config.ann.required:
                failures.append(warning)
        elif ann["status"] == "available":
            recall = ann["recall_at_k"]
            if not isinstance(recall, (int, float)) or not math.isfinite(recall):
                failures.append(f"{size}: ANN recall evidence is unavailable")
            elif recall < config.ann.min_recall_at_k:
                failures.append(
                    f"{size}: ANN recall@{config.k}={recall:.3f} "
                    f"below {config.ann.min_recall_at_k:.3f}"
                )
    return {
        "passed": not failures,
        "failures": failures,
        "warnings": warnings,
    }


def run_gate(config: ScaleGateConfig, work_dir: Path) -> dict[str, Any]:
    """Run every configured corpus size and evaluate the evidence gate."""
    work_dir.mkdir(parents=True, exist_ok=True)
    results = [_run_size(config, size, work_dir) for size in config.sizes]
    return {
        "schema_version": 1,
        "config": {
            "sizes": list(config.sizes),
            "seed": config.seed,
            "dimension": config.dimension,
            "model_name": config.model_name,
            "k": config.k,
            "warmup_count": config.warmup_count,
            "measured_repetitions": config.measured_repetitions,
            "concurrency": config.concurrency,
            "ann": {
                "enabled": config.ann.enabled,
                "required": config.ann.required,
                "min_recall_at_k": config.ann.min_recall_at_k,
            },
        },
        "results": results,
        "gate": evaluate_gate(results, config),
    }


def main() -> int:
    """Run the CLI gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("local_scale_gate.json"),
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path(__file__).with_name(".local-scale-work"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = run_gate(load_config(args.config), args.work_dir)
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(f"{payload}\n")
    print(payload)
    return 0 if report["gate"]["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
