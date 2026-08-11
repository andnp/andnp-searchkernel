"""Evaluation and benchmark runners for ranked retrieval quality."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import platform
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Any

from searchkernel.eval.golden import GoldenEntry, GoldenSet
from searchkernel.eval.metrics import average_precision, mrr, ndcg_at_k, recall_at_k
from searchkernel.runtime.trace import QueryTrace


@dataclass
class BenchmarkConfig:
    """Reproducible execution settings for evaluation and benchmark runs."""

    warmup_count: int = 0
    measured_repetitions: int = 1
    concurrency: int = 1
    capture_trace: bool = False
    corpus_version: str | None = None
    split: str | None = None
    backend: str | None = None
    model_fingerprint: str | None = None
    vector_dimension: int | None = None
    indexing_fingerprint: str | None = None
    ann_build_fingerprint: str | None = None
    ann_query_policy_fingerprint: str | None = None
    routing_fingerprint: str | None = None
    fusion_fingerprint: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    result_source_fn: Callable[[str], str | None] | None = None
    relevant_source_fn: Callable[[str], str | None] | None = None

    def __post_init__(self) -> None:
        """Reject settings that cannot produce a meaningful report."""
        if self.warmup_count < 0:
            raise ValueError("warmup_count must be non-negative")
        if self.measured_repetitions < 1:
            raise ValueError("measured_repetitions must be positive")
        if self.concurrency < 1:
            raise ValueError("concurrency must be positive")


def _current_rss_bytes() -> int | None:
    """Read the current process RSS without adding a dependency."""
    try:
        with open("/proc/self/statm") as statm:
            resident_pages = int(statm.read().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE")
    except (FileNotFoundError, IndexError, OSError, ValueError):
        return None


@dataclass
class BenchmarkHooks:
    """Optional lifecycle and resource hooks used by :func:`run_benchmark`."""

    before_cold: Callable[[], object] | None = None
    before_warm: Callable[[], object] | None = None
    build_index: Callable[[], object] | None = None
    load_index: Callable[[], object] | None = None
    index_size_bytes: Callable[[], int | None] | None = None
    rss_bytes: Callable[[], int | None] = _current_rss_bytes


@dataclass(frozen=True)
class SearchExecution:
    """Search output with optional source and trace metadata."""

    ids: tuple[str, ...]
    source_kinds: dict[str, str] = field(default_factory=dict)
    trace: QueryTrace | None = None
    diagnostics_complete: bool | None = None
    degraded: bool | None = None
    semantic_abstained: bool | None = None


@dataclass
class MetricSnapshot:
    """Per-query metric snapshot for one measured execution."""

    query: str
    recall_at_k: float
    ndcg_at_k: float
    mrr: float
    ap: float
    latency_ms: float | None
    repetition: int = 0
    query_type: str | None = None
    tags: list[str] = field(default_factory=list)
    source_kinds: list[str] = field(default_factory=list)
    empty_result: bool = False
    source_coverage: float | None = None
    stage_timings_ms: dict[str, float] = field(default_factory=dict)
    duplicate_result_ids: list[str] = field(default_factory=list)
    query_class: str | None = None
    diagnostics_complete: bool | None = None
    degraded: bool | None = None
    semantic_abstained: bool | None = None


SearchObservation = MetricSnapshot


@dataclass
class SliceReport:
    """Aggregate metrics for one source, query-type, or tag slice."""

    count: int
    mean_recall_at_k: float
    mean_ndcg_at_k: float
    mean_mrr: float
    mean_ap: float
    empty_result_rate: float
    mean_source_coverage: float | None
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    latency_p99_ms: float | None
    degradation_rate: float | None = None
    duplicate_result_rate: float | None = None
    semantic_abstention_rate: float | None = None
    diagnostics_complete: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the slice aggregate."""
        return {
            "count": self.count,
            "mean_recall_at_k": self.mean_recall_at_k,
            "mean_ndcg_at_k": self.mean_ndcg_at_k,
            "mean_mrr": self.mean_mrr,
            "mean_ap": self.mean_ap,
            "empty_result_rate": self.empty_result_rate,
            "mean_source_coverage": self.mean_source_coverage,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "latency_p99_ms": self.latency_p99_ms,
            "degradation_rate": self.degradation_rate,
            "duplicate_result_rate": self.duplicate_result_rate,
            "semantic_abstention_rate": self.semantic_abstention_rate,
            "diagnostics_complete": self.diagnostics_complete,
        }


@dataclass
class EvalReport:
    """Evaluation report for one measured execution set."""

    golden_set_size: int
    k: int
    metrics: list[MetricSnapshot] = field(default_factory=list)
    mean_recall_at_k: float | None = None
    mean_ndcg_at_k: float | None = None
    mean_mrr: float | None = None
    mean_ap: float | None = None
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None
    latency_p99_ms: float | None = None
    latency_mean_ms: float | None = None
    latency_min_ms: float | None = None
    latency_max_ms: float | None = None
    stage_latency_p50_ms: dict[str, float] = field(default_factory=dict)
    stage_latency_p95_ms: dict[str, float] = field(default_factory=dict)
    stage_latency_p99_ms: dict[str, float] = field(default_factory=dict)
    qps: float | None = None
    empty_result_rate: float | None = None
    mean_source_coverage: float | None = None
    per_source_recall: dict[str, float] = field(default_factory=dict)
    slices: dict[str, SliceReport] = field(default_factory=dict)
    mode: str = "warm"
    warmup_count: int = 0
    measured_repetitions: int = 1
    concurrency: int = 1
    wall_time_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    diagnostics_complete: bool | None = None
    degradation_rate: float | None = None
    duplicate_result_count: int = 0
    duplicate_result_rate: float | None = None
    semantic_abstention_rate: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report, including per-query distributions."""
        return {
            "golden_set_size": self.golden_set_size,
            "k": self.k,
            "mode": self.mode,
            "warmup_count": self.warmup_count,
            "measured_repetitions": self.measured_repetitions,
            "concurrency": self.concurrency,
            "wall_time_ms": self.wall_time_ms,
            "mean_recall_at_k": self.mean_recall_at_k,
            "mean_ndcg_at_k": self.mean_ndcg_at_k,
            "mean_mrr": self.mean_mrr,
            "mean_ap": self.mean_ap,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "latency_p99_ms": self.latency_p99_ms,
            "latency_mean_ms": self.latency_mean_ms,
            "latency_min_ms": self.latency_min_ms,
            "latency_max_ms": self.latency_max_ms,
            "stage_latency_p50_ms": self.stage_latency_p50_ms,
            "stage_latency_p95_ms": self.stage_latency_p95_ms,
            "stage_latency_p99_ms": self.stage_latency_p99_ms,
            "qps": self.qps,
            "empty_result_rate": self.empty_result_rate,
            "mean_source_coverage": self.mean_source_coverage,
            "per_source_recall": self.per_source_recall,
            "slices": {name: value.to_dict() for name, value in self.slices.items()},
            "metadata": self.metadata,
            "diagnostics_complete": self.diagnostics_complete,
            "degradation_rate": self.degradation_rate,
            "duplicate_result_count": self.duplicate_result_count,
            "duplicate_result_rate": self.duplicate_result_rate,
            "semantic_abstention_rate": self.semantic_abstention_rate,
            "per_query_metrics": [
                {
                    "query": metric.query,
                    "recall_at_k": metric.recall_at_k,
                    "ndcg_at_k": metric.ndcg_at_k,
                    "mrr": metric.mrr,
                    "ap": metric.ap,
                    "latency_ms": metric.latency_ms,
                    "repetition": metric.repetition,
                    "query_type": metric.query_type,
                    "tags": metric.tags,
                    "source_kinds": metric.source_kinds,
                    "empty_result": metric.empty_result,
                    "source_coverage": metric.source_coverage,
                    "stage_timings_ms": metric.stage_timings_ms,
                    "duplicate_result_ids": metric.duplicate_result_ids,
                    "query_class": metric.query_class,
                    "diagnostics_complete": metric.diagnostics_complete,
                    "degraded": metric.degraded,
                    "semantic_abstained": metric.semantic_abstained,
                }
                for metric in self.metrics
            ],
        }


@dataclass
class BenchmarkReport:
    """Cold-start and warm-cache reports plus reproducibility metadata."""

    cold: EvalReport
    warm: EvalReport
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize both benchmark phases."""
        return {
            "cold": self.cold.to_dict(),
            "warm": self.warm.to_dict(),
            "metadata": self.metadata,
        }


@dataclass
class AbReport:
    """A/B comparison report between two search functions."""

    report_a: EvalReport
    report_b: EvalReport
    recall_at_k_delta: float | None = None
    ndcg_at_k_delta: float | None = None
    mrr_delta: float | None = None
    ap_delta: float | None = None
    latency_p50_delta_ms: float | None = None
    latency_p95_delta_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    degradation_rate_delta: float | None = None
    duplicate_result_count_delta: int | None = None
    duplicate_result_rate_delta: float | None = None
    semantic_abstention_rate_delta: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the A/B reports and their deltas."""
        return {
            "report_a": self.report_a.to_dict(),
            "report_b": self.report_b.to_dict(),
            "deltas": {
                "recall_at_k": self.recall_at_k_delta,
                "ndcg_at_k": self.ndcg_at_k_delta,
                "mrr": self.mrr_delta,
                "ap": self.ap_delta,
                "latency_p50_ms": self.latency_p50_delta_ms,
                "latency_p95_ms": self.latency_p95_delta_ms,
                "degradation_rate": self.degradation_rate_delta,
                "duplicate_result_count": self.duplicate_result_count_delta,
                "duplicate_result_rate": self.duplicate_result_rate_delta,
                "semantic_abstention_rate": self.semantic_abstention_rate_delta,
            },
            "metadata": self.metadata,
        }


def _percentile(values: list[float], p: float) -> float:
    """Compute a percentile with the Hyndman-Fan type 7 method.

    The percentile position is ``(n - 1) * p / 100`` and adjacent sorted
    observations are linearly interpolated.
    """
    if not values:
        return 0.0
    if not 0 <= p <= 100:
        raise ValueError("p must be between 0 and 100")

    sorted_vals = sorted(values)
    position = (len(sorted_vals) - 1) * (p / 100.0)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_vals[lower]
    fraction = position - lower
    return sorted_vals[lower] + (sorted_vals[upper] - sorted_vals[lower]) * fraction


def _fingerprint(value: Any) -> str:
    """Hash canonical JSON metadata for reproducible comparisons."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def _environment_metadata() -> dict[str, Any]:
    """Return stable runtime details useful for comparing benchmark runs."""
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }


def _config_metadata(config: BenchmarkConfig, mode: str) -> dict[str, Any]:
    """Build fingerprints without serializing callback objects."""
    config_values = {
        "warmup_count": config.warmup_count,
        "measured_repetitions": config.measured_repetitions,
        "concurrency": config.concurrency,
        "capture_trace": config.capture_trace,
        "corpus_version": config.corpus_version,
        "split": config.split,
        "backend": config.backend,
        "model_fingerprint": config.model_fingerprint,
        "vector_dimension": config.vector_dimension,
        "indexing_fingerprint": config.indexing_fingerprint,
        "ann_build_fingerprint": config.ann_build_fingerprint,
        "ann_query_policy_fingerprint": config.ann_query_policy_fingerprint,
        "routing_fingerprint": config.routing_fingerprint,
        "fusion_fingerprint": config.fusion_fingerprint,
        "metadata": config.metadata,
    }
    environment = _environment_metadata()
    return {
        "mode": mode,
        "config": config_values,
        "config_fingerprint": _fingerprint(config_values),
        "corpus_version": config.corpus_version,
        "split": config.split,
        "backend": config.backend,
        "model_fingerprint": config.model_fingerprint,
        "vector_dimension": config.vector_dimension,
        "indexing_fingerprint": config.indexing_fingerprint,
        "ann_build_fingerprint": config.ann_build_fingerprint,
        "ann_query_policy_fingerprint": config.ann_query_policy_fingerprint,
        "routing_fingerprint": config.routing_fingerprint,
        "fusion_fingerprint": config.fusion_fingerprint,
        "environment": environment,
        "environment_fingerprint": _fingerprint(environment),
        **config.metadata,
    }


def _search_accepts_trace(search_fn: Callable[..., object]) -> bool:
    """Check whether a search callable exposes the optional trace argument."""
    try:
        parameters = inspect.signature(search_fn).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.name == "trace" or parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def _normalize_search_result(result: object) -> SearchExecution:
    """Adapt common ranked-result shapes to the benchmark representation."""
    if isinstance(result, SearchExecution):
        return result

    returned_trace: QueryTrace | None = None
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], QueryTrace):
        result, returned_trace = result

    raw_results = getattr(result, "results", result)
    source_kinds: dict[str, str] = {}
    ids: list[str] = []
    for item in raw_results:  # type: ignore[union-attr]
        if isinstance(item, str):
            result_id = item
            source_kind = None
        elif isinstance(item, tuple) and item and isinstance(item[0], str):
            result_id = item[0]
            source_kind = None
        else:
            result_id = (
                getattr(item, "storage_key", None)
                or getattr(item, "record_id", None)
                or getattr(item, "id", None)
            )
            source_kind = getattr(item, "source_kind", None)
            if not isinstance(result_id, str):
                raise TypeError("search results must expose a string result ID")
        ids.append(result_id)
        if isinstance(source_kind, str):
            source_kinds[result_id] = source_kind
    return SearchExecution(
        tuple(ids),
        source_kinds,
        returned_trace,
        _optional_bool(getattr(result, "diagnostics_complete", None)),
        _optional_bool(getattr(result, "degraded", None)),
        _optional_bool(getattr(result, "semantic_abstained", None)),
    )


def _optional_bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _run_search(
    search_fn: Callable[..., object],
    query: str,
    config: BenchmarkConfig,
) -> tuple[SearchExecution, QueryTrace]:
    """Execute one search while preserving optional implementation tracing."""
    trace = QueryTrace(query)
    accepts_trace = config.capture_trace and _search_accepts_trace(search_fn)
    start = time.perf_counter()
    with trace.span("search"):
        result = (
            search_fn(query, trace=trace)
            if accepts_trace
            else search_fn(query)
        )
    execution = _normalize_search_result(result)
    effective_trace = execution.trace or trace
    if effective_trace.end_time is None:
        effective_trace.close()
    if effective_trace.total_duration_ms is None:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return execution, replace(effective_trace, end_time=start + elapsed_ms / 1000)
    return execution, effective_trace


def _source_for_result(
    result_id: str,
    source_kinds: dict[str, str],
    config: BenchmarkConfig,
) -> str | None:
    """Resolve source metadata from the result or configured callback."""
    if result_id in source_kinds:
        return source_kinds[result_id]
    if config.result_source_fn is not None:
        return config.result_source_fn(result_id)
    return None


def _snapshot(
    entry: GoldenEntry,
    execution: SearchExecution,
    trace: QueryTrace,
    k: int,
    repetition: int,
    config: BenchmarkConfig,
) -> MetricSnapshot:
    """Compute relevance and slice metadata for one measured query."""
    ranked_ids = list(execution.ids)
    seen_ids: set[str] = set()
    duplicate_result_ids: list[str] = []
    for result_id in ranked_ids:
        if result_id in seen_ids and result_id not in duplicate_result_ids:
            duplicate_result_ids.append(result_id)
        seen_ids.add(result_id)
    relevant_ids = entry.relevant_ids
    source_map = {
        result_id: source
        for result_id in ranked_ids
        if (source := _source_for_result(result_id, execution.source_kinds, config))
        is not None
    }
    expected_sources = set(entry.source_kinds)
    observed_sources = {
        source_map[result_id]
        for result_id in ranked_ids[:k]
        if result_id in source_map
    }
    source_coverage = (
        len(expected_sources & observed_sources) / len(expected_sources)
        if expected_sources
        else None
    )
    return MetricSnapshot(
        query=entry.query,
        recall_at_k=recall_at_k(ranked_ids, relevant_ids, k),
        ndcg_at_k=ndcg_at_k(ranked_ids, relevant_ids, k, gains=entry.relevance),
        mrr=mrr(ranked_ids, relevant_ids),
        ap=average_precision(ranked_ids, relevant_ids),
        latency_ms=trace.total_duration_ms,
        repetition=repetition,
        query_type=entry.query_type,
        tags=list(entry.tags),
        query_class=entry.query_class or entry.query_type,
        source_kinds=sorted(observed_sources),
        empty_result=not ranked_ids,
        source_coverage=source_coverage,
        stage_timings_ms={
            name: span.duration_ms
            for name, span in trace.spans.items()
            if span.duration_ms is not None and name != "search"
        },
        duplicate_result_ids=duplicate_result_ids,
        diagnostics_complete=execution.diagnostics_complete,
        degraded=execution.degraded,
        semantic_abstained=execution.semantic_abstained,
    )


@dataclass
class _Measurement:
    """Internal measured query result used for report aggregation."""

    entry: GoldenEntry
    snapshot: MetricSnapshot
    execution: SearchExecution


def _measure_one(
    entry: GoldenEntry,
    search_fn: Callable[..., object],
    k: int,
    repetition: int,
    config: BenchmarkConfig,
) -> _Measurement:
    execution, trace = _run_search(search_fn, entry.query, config)
    return _Measurement(
        entry=entry,
        snapshot=_snapshot(entry, execution, trace, k, repetition, config),
        execution=execution,
    )


def _run_warmups(
    golden_set: GoldenSet,
    search_fn: Callable[..., object],
    config: BenchmarkConfig,
) -> None:
    """Execute warmups without adding them to the measured report."""
    if config.warmup_count == 0:
        return
    queries = [entry.query for entry in golden_set for _ in range(config.warmup_count)]
    if config.concurrency == 1:
        for query in queries:
            _run_warmup_one(search_fn, query, config)
        return
    with ThreadPoolExecutor(max_workers=config.concurrency) as executor:
        list(executor.map(lambda query: _run_warmup_one(search_fn, query, config), queries))


def _measure(
    golden_set: GoldenSet,
    search_fn: Callable[..., object],
    k: int,
    config: BenchmarkConfig,
) -> tuple[list[_Measurement], float]:
    """Measure all query repetitions, preserving input order under concurrency."""
    tasks = [
        (entry, repetition)
        for repetition in range(config.measured_repetitions)
        for entry in golden_set
    ]
    start = time.perf_counter()
    if config.concurrency == 1:
        measurements = [
            _measure_one(entry, search_fn, k, repetition, config)
            for entry, repetition in tasks
        ]
    else:
        with ThreadPoolExecutor(max_workers=config.concurrency) as executor:
            measurements = list(
                executor.map(
                    lambda task: _measure_one(task[0], search_fn, k, task[1], config),
                    tasks,
                )
            )
    return measurements, (time.perf_counter() - start) * 1000


def _slice_report(metrics: list[MetricSnapshot]) -> SliceReport:
    """Aggregate a group of query snapshots."""
    latencies = [metric.latency_ms for metric in metrics if metric.latency_ms is not None]
    coverage = [
        metric.source_coverage
        for metric in metrics
        if metric.source_coverage is not None
    ]
    degradation = _known_rate(metric.degraded for metric in metrics)
    abstention = _known_rate(metric.semantic_abstained for metric in metrics)
    return SliceReport(
        count=len(metrics),
        mean_recall_at_k=sum(metric.recall_at_k for metric in metrics) / len(metrics),
        mean_ndcg_at_k=sum(metric.ndcg_at_k for metric in metrics) / len(metrics),
        mean_mrr=sum(metric.mrr for metric in metrics) / len(metrics),
        mean_ap=sum(metric.ap for metric in metrics) / len(metrics),
        empty_result_rate=sum(metric.empty_result for metric in metrics) / len(metrics),
        mean_source_coverage=sum(coverage) / len(coverage) if coverage else None,
        latency_p50_ms=_percentile(latencies, 50) if latencies else None,
        latency_p95_ms=_percentile(latencies, 95) if latencies else None,
        latency_p99_ms=_percentile(latencies, 99) if latencies else None,
        degradation_rate=degradation,
        duplicate_result_rate=sum(
            bool(metric.duplicate_result_ids) for metric in metrics
        )
        / len(metrics),
        semantic_abstention_rate=abstention,
        diagnostics_complete=_diagnostics_complete(metrics),
    )


def _known_rate(values: Iterable[bool | None]) -> float | None:
    known = [value for value in values if isinstance(value, bool)]
    return sum(value is True for value in known) / len(known) if known else None


def _diagnostics_complete(metrics: list[MetricSnapshot]) -> bool | None:
    values = [metric.diagnostics_complete for metric in metrics]
    known = [value for value in values if isinstance(value, bool)]
    if not known:
        return None
    return len(known) == len(metrics) and all(known)


def _stage_latency_percentiles(
    metrics: list[MetricSnapshot],
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Aggregate measured per-stage timings using the existing percentile method."""
    timings: dict[str, list[float]] = defaultdict(list)
    for metric in metrics:
        for stage, duration_ms in metric.stage_timings_ms.items():
            timings[stage].append(duration_ms)
    percentiles = [
        {
            stage: _percentile(durations, percentile)
            for stage, durations in sorted(timings.items())
        }
        for percentile in (50, 95, 99)
    ]
    return percentiles[0], percentiles[1], percentiles[2]


def _run_warmup_one(
    search_fn: Callable[..., object],
    query: str,
    config: BenchmarkConfig,
) -> None:
    """Run one warmup while honoring the search callable's optional trace hook."""
    if config.capture_trace and _search_accepts_trace(search_fn):
        trace = QueryTrace(query)
        search_fn(query, trace=trace)
        trace.close()
    else:
        search_fn(query)


def _build_report(
    golden_set: GoldenSet,
    measurements: list[_Measurement],
    k: int,
    config: BenchmarkConfig,
    wall_time_ms: float,
    mode: str,
) -> EvalReport:
    """Build aggregate, slice, and per-source metrics from measurements."""
    metrics = [measurement.snapshot for measurement in measurements]
    report = EvalReport(
        golden_set_size=len(golden_set),
        k=k,
        metrics=metrics,
        mode=mode,
        warmup_count=config.warmup_count,
        measured_repetitions=config.measured_repetitions,
        concurrency=config.concurrency,
        wall_time_ms=wall_time_ms,
        metadata=_config_metadata(config, mode),
    )
    if not metrics:
        return report

    latencies = [metric.latency_ms for metric in metrics if metric.latency_ms is not None]
    report.mean_recall_at_k = sum(metric.recall_at_k for metric in metrics) / len(metrics)
    report.mean_ndcg_at_k = sum(metric.ndcg_at_k for metric in metrics) / len(metrics)
    report.mean_mrr = sum(metric.mrr for metric in metrics) / len(metrics)
    report.mean_ap = sum(metric.ap for metric in metrics) / len(metrics)
    report.empty_result_rate = sum(metric.empty_result for metric in metrics) / len(metrics)
    report.diagnostics_complete = _diagnostics_complete(metrics)
    report.degradation_rate = _known_rate(metric.degraded for metric in metrics)
    report.duplicate_result_count = sum(
        bool(metric.duplicate_result_ids) for metric in metrics
    )
    report.duplicate_result_rate = report.duplicate_result_count / len(metrics)
    report.semantic_abstention_rate = _known_rate(
        metric.semantic_abstained for metric in metrics
    )
    coverage = [
        metric.source_coverage
        for metric in metrics
        if metric.source_coverage is not None
    ]
    report.mean_source_coverage = sum(coverage) / len(coverage) if coverage else None
    if latencies:
        report.latency_p50_ms = _percentile(latencies, 50)
        report.latency_p95_ms = _percentile(latencies, 95)
        report.latency_p99_ms = _percentile(latencies, 99)
        report.latency_mean_ms = sum(latencies) / len(latencies)
        report.latency_min_ms = min(latencies)
        report.latency_max_ms = max(latencies)
    (
        report.stage_latency_p50_ms,
        report.stage_latency_p95_ms,
        report.stage_latency_p99_ms,
    ) = _stage_latency_percentiles(metrics)
    if wall_time_ms > 0:
        report.qps = len(metrics) / (wall_time_ms / 1000)

    slices: dict[str, list[MetricSnapshot]] = defaultdict(list)
    for metric in metrics:
        if metric.query_type is not None:
            slices[f"query_type:{metric.query_type}"].append(metric)
        if metric.query_class is not None:
            slices[f"query_class:{metric.query_class}"].append(metric)
        for tag in metric.tags:
            slices[f"tag:{tag}"].append(metric)
        for source_kind in metric.source_kinds:
            slices[f"source:{source_kind}"].append(metric)
    report.slices = {
        name: _slice_report(slice_metrics)
        for name, slice_metrics in sorted(slices.items())
    }

    per_source_hits: dict[str, int] = defaultdict(int)
    per_source_total: dict[str, int] = defaultdict(int)
    for measurement in measurements:
        entry = measurement.entry
        relevant_by_source: dict[str, set[str]] = defaultdict(set)
        for result_id in entry.relevant_ids:
            source_kind = config.relevant_source_fn(result_id) if config.relevant_source_fn else None
            if source_kind is None and len(entry.source_kinds) == 1:
                source_kind = entry.source_kinds[0]
            if source_kind is not None:
                relevant_by_source[source_kind].add(result_id)
        for source_kind, source_ids in relevant_by_source.items():
            per_source_total[source_kind] += len(source_ids)
            top_ids = set(measurement.execution.ids[:k])
            per_source_hits[source_kind] += len(top_ids & source_ids)
    report.per_source_recall = {
        source_kind: per_source_hits[source_kind] / total
        for source_kind, total in sorted(per_source_total.items())
        if total
    }
    return report


def run_eval(
    golden_set: GoldenSet,
    search_fn: Callable[..., object],
    k: int = 10,
    *,
    config: BenchmarkConfig | None = None,
    warmup_count: int | None = None,
    measured_repetitions: int | None = None,
    concurrency: int | None = None,
    capture_trace: bool | None = None,
    mode: str = "warm",
) -> EvalReport:
    """Run measured retrieval evaluation with optional warmups and concurrency."""
    effective_config = config or BenchmarkConfig()
    overrides: dict[str, Any] = {}
    if warmup_count is not None:
        overrides["warmup_count"] = warmup_count
    if measured_repetitions is not None:
        overrides["measured_repetitions"] = measured_repetitions
    if concurrency is not None:
        overrides["concurrency"] = concurrency
    if capture_trace is not None:
        overrides["capture_trace"] = capture_trace
    if overrides:
        effective_config = replace(effective_config, **overrides)
    _run_warmups(golden_set, search_fn, effective_config)
    measurements, wall_time_ms = _measure(golden_set, search_fn, k, effective_config)
    return _build_report(golden_set, measurements, k, effective_config, wall_time_ms, mode)


def run_benchmark(
    golden_set: GoldenSet,
    search_fn: Callable[..., object],
    k: int = 10,
    *,
    config: BenchmarkConfig | None = None,
    hooks: BenchmarkHooks | None = None,
) -> BenchmarkReport:
    """Run cold-start and warm-cache evaluations with resource metadata."""
    effective_config = config or BenchmarkConfig()
    effective_hooks = hooks or BenchmarkHooks()
    metadata = _config_metadata(effective_config, "benchmark")

    rss_before = effective_hooks.rss_bytes()
    build_start = time.perf_counter()
    if effective_hooks.build_index is not None:
        effective_hooks.build_index()
    metadata["index_build_time_ms"] = (
        (time.perf_counter() - build_start) * 1000
        if effective_hooks.build_index is not None
        else None
    )

    load_start = time.perf_counter()
    if effective_hooks.load_index is not None:
        effective_hooks.load_index()
    metadata["index_load_time_ms"] = (
        (time.perf_counter() - load_start) * 1000
        if effective_hooks.load_index is not None
        else None
    )
    metadata["rss_before_index_load_bytes"] = rss_before
    metadata["rss_after_index_load_bytes"] = effective_hooks.rss_bytes()
    metadata["index_size_bytes"] = (
        effective_hooks.index_size_bytes()
        if effective_hooks.index_size_bytes is not None
        else None
    )

    if effective_hooks.before_cold is not None:
        effective_hooks.before_cold()
    cold_config = replace(effective_config, warmup_count=0)
    cold = run_eval(golden_set, search_fn, k, config=cold_config, mode="cold")
    metadata["rss_peak_query_bytes"] = effective_hooks.rss_bytes()

    if effective_hooks.before_warm is not None:
        effective_hooks.before_warm()
    warm = run_eval(golden_set, search_fn, k, config=effective_config, mode="warm")
    cold.metadata.update(metadata)
    warm.metadata.update(metadata)
    return BenchmarkReport(cold=cold, warm=warm, metadata=metadata)


def ab_eval(
    golden_set: GoldenSet,
    search_fn_a: Callable[..., object],
    search_fn_b: Callable[..., object],
    k: int = 10,
    *,
    config: BenchmarkConfig | None = None,
) -> AbReport:
    """Run A/B evaluation comparing two search functions."""
    report_a = run_eval(golden_set, search_fn_a, k, config=config)
    report_b = run_eval(golden_set, search_fn_b, k, config=config)

    ab_report = AbReport(
        report_a=report_a,
        report_b=report_b,
        metadata=_config_metadata(config or BenchmarkConfig(), "ab"),
    )
    if report_a.mean_recall_at_k is not None and report_b.mean_recall_at_k is not None:
        ab_report.recall_at_k_delta = report_b.mean_recall_at_k - report_a.mean_recall_at_k
    if report_a.mean_ndcg_at_k is not None and report_b.mean_ndcg_at_k is not None:
        ab_report.ndcg_at_k_delta = report_b.mean_ndcg_at_k - report_a.mean_ndcg_at_k
    if report_a.mean_mrr is not None and report_b.mean_mrr is not None:
        ab_report.mrr_delta = report_b.mean_mrr - report_a.mean_mrr
    if report_a.mean_ap is not None and report_b.mean_ap is not None:
        ab_report.ap_delta = report_b.mean_ap - report_a.mean_ap
    if report_a.latency_p50_ms is not None and report_b.latency_p50_ms is not None:
        ab_report.latency_p50_delta_ms = report_b.latency_p50_ms - report_a.latency_p50_ms
    if report_a.latency_p95_ms is not None and report_b.latency_p95_ms is not None:
        ab_report.latency_p95_delta_ms = report_b.latency_p95_ms - report_a.latency_p95_ms
    if report_a.degradation_rate is not None and report_b.degradation_rate is not None:
        ab_report.degradation_rate_delta = (
            report_b.degradation_rate - report_a.degradation_rate
        )
    ab_report.duplicate_result_count_delta = (
        report_b.duplicate_result_count - report_a.duplicate_result_count
    )
    if (
        report_a.duplicate_result_rate is not None
        and report_b.duplicate_result_rate is not None
    ):
        ab_report.duplicate_result_rate_delta = (
            report_b.duplicate_result_rate - report_a.duplicate_result_rate
        )
    if (
        report_a.semantic_abstention_rate is not None
        and report_b.semantic_abstention_rate is not None
    ):
        ab_report.semantic_abstention_rate_delta = (
            report_b.semantic_abstention_rate - report_a.semantic_abstention_rate
        )
    return ab_report
