"""Unit tests for evaluation runner."""

import json

import pytest

from searchkernel.eval.golden import GoldenEntry, GoldenSet
from searchkernel.eval.runner import (
    BenchmarkConfig,
    BenchmarkHooks,
    MetricSnapshot,
    SearchExecution,
    _percentile,
    _stage_latency_percentiles,
    ab_eval,
    paired_metric_scores,
    per_query_metric_scores,
    run_benchmark,
    run_eval,
)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("split", "test"),
        ("vector_dimension", 384),
        ("indexing_fingerprint", "index-v1"),
        ("ann_build_fingerprint", "ann-build-v1"),
        ("ann_query_policy_fingerprint", "ann-query-v1"),
        ("routing_fingerprint", "routing-v1"),
        ("fusion_fingerprint", "fusion-v1"),
    ],
)
def test_run_eval_config_metadata_fields_are_fingerprinted(field_name, value):
    """Schema metadata is serialized and changes the config fingerprint."""
    golden_set = GoldenSet(entries=[GoldenEntry(query="q", relevant_ids=["a"])])

    baseline = run_eval(golden_set, lambda query: ["a"])
    changed = run_eval(
        golden_set,
        lambda query: ["a"],
        config=BenchmarkConfig(**{field_name: value}),
    )

    assert changed.metadata[field_name] == value
    assert changed.to_dict()["metadata"][field_name] == value
    assert changed.metadata["config_fingerprint"] != baseline.metadata["config_fingerprint"]


def test_run_eval_perfect_search():
    """Test eval with a perfect search function."""
    golden_set = GoldenSet(
        entries=[
            GoldenEntry(query="query1", relevant_ids=["a", "b"]),
            GoldenEntry(query="query2", relevant_ids=["c"]),
        ]
    )

    def perfect_search(query: str) -> list[str]:
        """Returns all relevant IDs in order."""
        if query == "query1":
            return ["a", "b", "x", "y"]
        elif query == "query2":
            return ["c", "x", "y"]
        else:
            return []

    report = run_eval(golden_set, perfect_search, k=2)

    assert report.golden_set_size == 2
    assert report.k == 2
    assert len(report.metrics) == 2

    # Query 1: ["a", "b"], relevant: {a, b}
    # Recall@2 = 2/2 = 1.0
    # nDCG@2 = 1.0
    # MRR = 1.0 (first relevant at position 1)
    # AP = 1.0
    assert report.metrics[0].recall_at_k == 1.0
    assert report.metrics[0].ndcg_at_k == 1.0
    assert report.metrics[0].mrr == 1.0
    assert report.metrics[0].ap == 1.0

    # Query 2: ["c", "x", "y"], relevant: {c}
    # Recall@2 = 1/1 = 1.0
    # nDCG@2 = 1.0
    # MRR = 1.0
    # AP = 1.0
    assert report.metrics[1].recall_at_k == 1.0
    assert report.metrics[1].ndcg_at_k == 1.0
    assert report.metrics[1].mrr == 1.0
    assert report.metrics[1].ap == 1.0

    # Aggregates
    assert report.mean_recall_at_k == 1.0
    assert report.mean_ndcg_at_k == 1.0
    assert report.mean_mrr == 1.0
    assert report.mean_ap == 1.0


def test_run_eval_partial_search():
    """Test eval with partial matches."""
    golden_set = GoldenSet(
        entries=[
            GoldenEntry(query="q1", relevant_ids=["a", "b", "c"]),
            GoldenEntry(query="q2", relevant_ids=["x", "y"]),
        ]
    )

    def partial_search(query: str) -> list[str]:
        """Returns some relevant items."""
        if query == "q1":
            return ["a", "z", "b"]  # Has a, b but missing c
        elif query == "q2":
            return ["z", "w"]  # No relevant items
        else:
            return []

    report = run_eval(golden_set, partial_search, k=3)

    # Query 1: ["a", "z", "b"], relevant: {a, b, c}
    # Recall@3 = 2/3 = 0.667
    assert abs(report.metrics[0].recall_at_k - 2.0 / 3.0) < 0.001

    # Query 2: ["z", "w"], relevant: {x, y}
    # Recall@3 = 0/2 = 0.0
    assert report.metrics[1].recall_at_k == 0.0

    # Aggregate recall = (2/3 + 0) / 2 = 0.333
    mean_recall = report.mean_recall_at_k
    assert mean_recall is not None
    assert abs(mean_recall - 1.0 / 3.0) < 0.001


def test_run_eval_reports_duplicate_result_ids():
    golden_set = GoldenSet(
        entries=[GoldenEntry(query="q", relevant_ids=["a"])]
    )

    report = run_eval(golden_set, lambda query: ["a", "a", "b"], k=3)

    assert report.metrics[0].duplicate_result_ids == ["a"]
    assert report.to_dict()["per_query_metrics"][0]["duplicate_result_ids"] == ["a"]


def test_run_eval_preserves_diagnostic_unavailability_and_measured_zero() -> None:
    """Unavailable diagnostics remain null while measured false values yield zero."""
    golden_set = GoldenSet(
        entries=[
            GoldenEntry(query="known", relevant_ids=["a"]),
            GoldenEntry(query="unknown", relevant_ids=["b"]),
        ]
    )

    def search(query: str) -> SearchExecution:
        if query == "known":
            return SearchExecution(
                ids=("a",),
                diagnostics_complete=True,
                degraded=False,
                semantic_abstained=False,
            )
        return SearchExecution(ids=("b",))

    report = run_eval(golden_set, search)

    assert report.diagnostics_complete is False
    assert report.degradation_rate == 0.0
    assert report.semantic_abstention_rate == 0.0
    assert report.duplicate_result_count == 0
    assert report.duplicate_result_rate == 0.0


def test_run_eval_aggregates_query_class_slices() -> None:
    """Query classes produce deterministic provider-neutral slice aggregates."""
    golden_set = GoldenSet(
        entries=[GoldenEntry(query="q", relevant_ids=["a"], query_class="broad")]
    )

    report = run_eval(golden_set, lambda query: ["a"])

    assert report.slices["query_class:broad"].count == 1
    assert report.metrics[0].query_class == "broad"


def test_run_eval_latency_percentiles():
    """Test that latency percentiles are computed."""
    golden_set = GoldenSet(
        entries=[
            GoldenEntry(query="q1", relevant_ids=["a"]),
            GoldenEntry(query="q2", relevant_ids=["b"]),
            GoldenEntry(query="q3", relevant_ids=["c"]),
        ]
    )

    def dummy_search(query: str) -> list[str]:
        return [query[1]]  # Just return something

    report = run_eval(golden_set, dummy_search, k=10)

    # Should have latency percentiles
    assert report.latency_p50_ms is not None
    assert report.latency_p95_ms is not None
    assert report.latency_p99_ms is not None

    # Percentiles should be ordered
    p50 = report.latency_p50_ms
    p95 = report.latency_p95_ms
    p99 = report.latency_p99_ms
    assert p50 is not None
    assert p95 is not None
    assert p99 is not None
    assert p50 <= p95
    assert p95 <= p99


def test_percentile_uses_linear_interpolation_for_small_samples():
    """Type 7 interpolation matches the documented one-to-many examples."""
    assert _percentile([7.0], 50) == 7.0
    assert _percentile([0.0, 10.0], 50) == 5.0
    assert _percentile([0.0, 10.0, 20.0], 50) == 10.0
    assert _percentile(list(map(float, range(100))), 95) == 94.05


def test_stage_latency_percentiles_aggregate_only_measured_stage_samples():
    """Stage distributions use type-7 percentiles and omit absent stages."""
    metrics = [
        MetricSnapshot(
            query=f"q{index}",
            recall_at_k=1.0,
            ndcg_at_k=1.0,
            mrr=1.0,
            ap=1.0,
            latency_ms=10.0,
            stage_timings_ms={"retrieval": float(index), "reranking": 100.0},
        )
        for index in (0, 10, 20, 30)
    ]

    p50, p95, p99 = _stage_latency_percentiles(metrics)

    assert p50 == {"reranking": 100.0, "retrieval": 15.0}
    assert p95["retrieval"] == pytest.approx(28.5)
    assert p99["retrieval"] == pytest.approx(29.7)
    assert p95["reranking"] == 100.0


def test_run_eval_reports_stage_latency_percentiles_and_keeps_warmups_out():
    """Stage reports contain measured spans while warmups remain excluded."""
    calls = 0
    golden_set = GoldenSet(entries=[GoldenEntry(query="q", relevant_ids=["a"])])

    def search(query: str, *, trace) -> SearchExecution:
        nonlocal calls
        calls += 1
        with trace.span("retrieval"):
            pass
        with trace.span("hydration"):
            pass
        return SearchExecution(ids=("a",), trace=trace)

    report = run_eval(
        golden_set,
        search,
        config=BenchmarkConfig(
            capture_trace=True,
            warmup_count=2,
            measured_repetitions=3,
        ),
    )

    assert calls == 5
    assert set(report.stage_latency_p50_ms) == {"hydration", "retrieval"}
    assert set(report.stage_latency_p95_ms) == set(report.stage_latency_p50_ms)
    assert set(report.stage_latency_p99_ms) == set(report.stage_latency_p50_ms)
    assert report.to_dict()["stage_latency_p50_ms"] == report.stage_latency_p50_ms


def test_run_eval_excludes_warmups_and_reports_repetitions():
    """Only measured calls appear in snapshots and aggregate counts."""
    calls: list[str] = []
    golden_set = GoldenSet(
        entries=[
            GoldenEntry(query="q1", relevant_ids=["a"]),
            GoldenEntry(query="q2", relevant_ids=["b"]),
        ]
    )

    def search(query: str) -> list[str]:
        calls.append(query)
        return ["a" if query == "q1" else "b"]

    report = run_eval(
        golden_set,
        search,
        config=BenchmarkConfig(warmup_count=2, measured_repetitions=3),
    )

    assert len(calls) == 2 * (2 + 3)
    assert len(report.metrics) == 2 * 3
    assert {metric.repetition for metric in report.metrics} == {0, 1, 2}
    assert report.warmup_count == 2


def test_run_eval_concurrent_outputs_are_deterministic():
    """Concurrent execution keeps input order and metric values stable."""
    golden_set = GoldenSet(
        entries=[
            GoldenEntry(query="q1", relevant_ids=["a"]),
            GoldenEntry(query="q2", relevant_ids=["b"]),
        ]
    )

    def search(query: str) -> list[str]:
        return ["a" if query == "q1" else "b"]

    report = run_eval(
        golden_set,
        search,
        config=BenchmarkConfig(measured_repetitions=2, concurrency=2),
    )

    assert [(metric.query, metric.repetition) for metric in report.metrics] == [
        ("q1", 0),
        ("q2", 0),
        ("q1", 1),
        ("q2", 1),
    ]
    assert [metric.recall_at_k for metric in report.metrics] == [1.0] * 4


def test_run_eval_reports_graded_slices_and_trace():
    """Graded nDCG, source slices, coverage, and stage timings are reported."""
    golden_set = GoldenSet(
        entries=[
            GoldenEntry(
                query="q",
                relevant_ids=["high", "low"],
                relevance={"high": 3.0, "low": 1.0},
                query_type="identifier",
                source_kinds=["docs", "issues"],
                tags=["hard"],
            )
        ]
    )

    def search(query: str, *, trace) -> SearchExecution:
        with trace.span("stage"):
            return SearchExecution(
                ids=("low", "high"),
                source_kinds={"low": "issues", "high": "docs"},
                trace=trace,
            )

    report = run_eval(
        golden_set,
        search,
        k=2,
        config=BenchmarkConfig(capture_trace=True, relevant_source_fn=lambda value: {
            "high": "docs",
            "low": "issues",
        }[value]),
    )

    assert report.mean_ndcg_at_k is not None
    assert report.mean_ndcg_at_k < 1.0
    assert report.mean_source_coverage == 1.0
    assert report.per_source_recall == {"docs": 1.0, "issues": 1.0}
    assert report.slices["query_type:identifier"].count == 1
    assert report.slices["source:docs"].count == 1
    assert report.slices["tag:hard"].count == 1
    assert report.metrics[0].stage_timings_ms["stage"] >= 0.0


def test_run_benchmark_reports_cold_warm_and_metadata():
    """Benchmark reports include lifecycle hooks and reproducibility fields."""
    lifecycle: list[str] = []
    golden_set = GoldenSet(
        entries=[GoldenEntry(query="q", relevant_ids=["a"])]
    )

    def search(query: str) -> list[str]:
        return ["a"]

    report = run_benchmark(
        golden_set,
        search,
        config=BenchmarkConfig(
            warmup_count=1,
            corpus_version="corpus-v1",
            backend="fake",
            model_fingerprint="model-v1",
        ),
        hooks=BenchmarkHooks(
            before_cold=lambda: lifecycle.append("cold"),
            before_warm=lambda: lifecycle.append("warm"),
            build_index=lambda: lifecycle.append("build"),
            load_index=lambda: lifecycle.append("load"),
            index_size_bytes=lambda: 123,
            rss_bytes=lambda: 456,
        ),
    )

    assert lifecycle == ["build", "load", "cold", "warm"]
    assert report.cold.mode == "cold"
    assert report.warm.mode == "warm"
    assert report.metadata["corpus_version"] == "corpus-v1"
    assert report.metadata["config_fingerprint"]
    assert report.metadata["environment_fingerprint"]
    assert report.metadata["index_size_bytes"] == 123
    assert report.metadata["rss_before_index_load_bytes"] == 456
    json.dumps(report.to_dict())


def test_run_eval_empty_golden_set():
    """Test eval with empty golden set."""
    golden_set = GoldenSet(entries=[])

    def dummy_search(query: str) -> list[str]:
        return []

    report = run_eval(golden_set, dummy_search, k=10)

    assert report.golden_set_size == 0
    assert len(report.metrics) == 0
    assert report.mean_recall_at_k is None
    assert report.latency_p50_ms is None
    assert report.degradation_rate is None
    assert report.semantic_abstention_rate is None


def test_ab_eval():
    """Test A/B evaluation."""
    golden_set = GoldenSet(
        entries=[
            GoldenEntry(query="q1", relevant_ids=["a"]),
            GoldenEntry(query="q2", relevant_ids=["b"]),
        ]
    )

    def search_a(query: str) -> list[str]:
        """Baseline: always returns nothing."""
        return []

    def search_b(query: str) -> list[str]:
        """Candidate: returns correct answer."""
        if query == "q1":
            return ["a", "x"]
        elif query == "q2":
            return ["b", "y"]
        return []

    ab_report = ab_eval(golden_set, search_a, search_b, k=10)

    # Baseline should have 0 recall
    assert ab_report.report_a.mean_recall_at_k == 0.0
    # Candidate should have 1.0 recall
    assert ab_report.report_b.mean_recall_at_k == 1.0

    # Delta should be positive
    assert ab_report.recall_at_k_delta == 1.0
    assert ab_report.ndcg_at_k_delta == 1.0


def test_ab_eval_regression():
    """Test A/B eval detecting a regression."""
    golden_set = GoldenSet(
        entries=[
            GoldenEntry(query="q1", relevant_ids=["a", "b"]),
        ]
    )

    def search_good(query: str) -> list[str]:
        return ["a", "b", "c"]

    def search_bad(query: str) -> list[str]:
        return ["c", "d", "e"]

    ab_report = ab_eval(golden_set, search_good, search_bad, k=5)

    # Good should have higher recall
    recall_a = ab_report.report_a.mean_recall_at_k
    recall_b = ab_report.report_b.mean_recall_at_k
    assert recall_a is not None
    assert recall_b is not None
    assert recall_a > recall_b
    # Delta should be negative
    recall_delta = ab_report.recall_at_k_delta
    assert recall_delta is not None
    assert recall_delta < 0


class TestPerQueryMetricScores:
    """Extraction of per-query metric values for paired significance testing."""

    def test_returns_one_value_per_query_in_first_seen_order(self):
        """Single-repetition reports map each query straight to its metric."""
        golden_set = GoldenSet(
            entries=[
                GoldenEntry(query="q1", relevant_ids=["a"]),
                GoldenEntry(query="q2", relevant_ids=["b"]),
            ]
        )
        report = run_eval(golden_set, lambda query: ["a", "b"], k=1)

        scores = per_query_metric_scores(report, "recall_at_k")

        assert list(scores.keys()) == ["q1", "q2"]
        assert scores["q1"] == 1.0
        assert scores["q2"] == 0.0

    def test_averages_across_repetitions_for_same_query(self):
        """A query measured multiple times collapses to its mean score."""
        golden_set = GoldenSet(entries=[GoldenEntry(query="q1", relevant_ids=["a"])])
        calls = {"count": 0}

        def flaky_search(query: str) -> list[str]:
            calls["count"] += 1
            return ["a"] if calls["count"] % 2 == 1 else ["z"]

        report = run_eval(golden_set, flaky_search, k=1, measured_repetitions=2)

        scores = per_query_metric_scores(report, "recall_at_k")

        assert scores["q1"] == pytest.approx(0.5)

    def test_unsupported_metric_name_raises(self):
        """Only the documented metric fields are valid extraction targets."""
        golden_set = GoldenSet(entries=[GoldenEntry(query="q1", relevant_ids=["a"])])
        report = run_eval(golden_set, lambda query: ["a"], k=1)

        with pytest.raises(ValueError):
            per_query_metric_scores(report, "latency_ms")


class TestPairedMetricScores:
    """Aligning two reports' per-query values for compare_paired."""

    def test_aligns_scores_by_query_across_two_reports(self):
        """Values line up index-for-index by query, ready for a paired test."""
        golden_set = GoldenSet(
            entries=[
                GoldenEntry(query="q1", relevant_ids=["a"]),
                GoldenEntry(query="q2", relevant_ids=["b"]),
            ]
        )
        baseline_report = run_eval(golden_set, lambda query: ["x"], k=2)
        candidate_report = run_eval(golden_set, lambda query: ["a", "b"], k=2)

        baseline, candidate = paired_metric_scores(
            baseline_report, candidate_report, "recall_at_k"
        )

        assert baseline == [0.0, 0.0]
        assert candidate == [1.0, 1.0]

    def test_mismatched_query_sets_raise(self):
        """A comparison across reports with different queries is not paired."""
        report_a = run_eval(
            GoldenSet(entries=[GoldenEntry(query="q1", relevant_ids=["a"])]),
            lambda query: ["a"],
            k=1,
        )
        report_b = run_eval(
            GoldenSet(entries=[GoldenEntry(query="q2", relevant_ids=["a"])]),
            lambda query: ["a"],
            k=1,
        )

        with pytest.raises(ValueError):
            paired_metric_scores(report_a, report_b, "recall_at_k")
