"""Tests for benchmark artifact contracts and stable regression gates."""

from benchmarks.evidence import EvidencePolicy, compare_report, validate_report


def _report() -> dict[str, object]:
    return {
        "golden_set_size": 2,
        "k": 3,
        "mode": "warm",
        "warmup_count": 2,
        "measured_repetitions": 3,
        "metadata": {"corpus_version": "v1", "backend": "sqlite"},
        "mean_recall_at_k": 0.9,
        "mean_ndcg_at_k": 0.8,
        "mean_mrr": 0.8,
        "mean_ap": 0.75,
        "latency_p95_ms": 10.0,
        "per_query_metrics": [{}] * 6,
    }


def test_validate_report_requires_repeated_warm_measurements() -> None:
    assert validate_report(_report(), EvidencePolicy()) == []

    report = _report()
    report["measured_repetitions"] = 1
    assert "measured_repetitions must be >= 3" in validate_report(
        report, EvidencePolicy()
    )


def test_compare_report_uses_baseline_quality_without_latency_gate() -> None:
    candidate = _report()
    baseline = {"mean_recall_at_k": 0.9, "mean_ndcg_at_k": 0.8, "mean_mrr": 0.8, "mean_ap": 0.75}

    result = compare_report(candidate, baseline, EvidencePolicy())

    assert result["passed"] is True
    assert result["deltas"] == {
        "mean_recall_at_k": 0.0,
        "mean_ndcg_at_k": 0.0,
        "mean_mrr": 0.0,
        "mean_ap": 0.0,
    }


def test_compare_report_can_enable_relative_latency_gate() -> None:
    policy = EvidencePolicy(max_latency_regression_ratio=0.1)
    result = compare_report(
        _report(),
        {
            "mean_recall_at_k": 0.9,
            "mean_ndcg_at_k": 0.8,
            "mean_mrr": 0.8,
            "mean_ap": 0.75,
            "latency_p95_ms": 8.0,
        },
        policy,
    )

    assert result["passed"] is False
    assert any("latency_p95 regression" in failure for failure in result["failures"])
