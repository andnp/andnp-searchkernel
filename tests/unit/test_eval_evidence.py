"""Tests for benchmark artifact contracts and stable regression gates."""

import pytest

from benchmarks.evidence import EvidencePolicy, compare_report, validate_report
from searchkernel.eval.evidence import (
    EvidencePolicy as ProviderEvidencePolicy,
)
from searchkernel.eval.evidence import (
    compare_report as compare_provider_report,
)
from searchkernel.eval.evidence import (
    validate_report as validate_provider_report,
)


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


def test_benchmark_evidence_rejects_incompatible_metadata_before_deltas() -> None:
    """
    The public benchmark wrapper rejects strict metadata mismatches first.
    """
    baseline = {
        "metadata": {
            "corpus_version": "v1",
            "split": "test",
            "backend": "sqlite",
            "model_fingerprint": "model-v1",
            "vector_dimension": 384,
            "indexing_fingerprint": "index-v1",
            "ann_build_fingerprint": "ann-build-v1",
            "ann_query_policy_fingerprint": "ann-query-v1",
            "routing_fingerprint": "routing-v1",
            "fusion_fingerprint": "fusion-v1",
            "config_fingerprint": "config-v1",
            "environment_fingerprint": "env-v1",
        },
        "mean_recall_at_k": 0.8,
    }
    candidate = dict(baseline)
    candidate["metadata"] = dict(baseline["metadata"])
    candidate["metadata"]["config_fingerprint"] = "config-v2"
    candidate["mean_recall_at_k"] = 0.9

    result = compare_report(
        candidate,
        baseline,
        EvidencePolicy(require_metadata_compatibility=True),
    )

    assert result["passed"] is False
    assert any("metadata incompatible: config_fingerprint" in failure for failure in result["failures"])
    assert result["deltas"] == {}


def test_benchmark_evidence_loads_strict_metadata_policy() -> None:
    """
    JSON policies can opt into strict report metadata compatibility.
    """
    policy = EvidencePolicy.from_dict({"require_metadata_compatibility": True})

    result = compare_report(
        {"mean_recall_at_k": 1.0},
        {"mean_recall_at_k": 1.0},
        policy,
    )

    assert result["passed"] is False
    assert result["failures"] == [
        "metadata compatibility requires metadata on both reports"
    ]


def test_provider_evidence_uses_none_for_unavailable_diagnostics() -> None:
    """Provider-neutral validation distinguishes absent diagnostics from zero rates."""
    report = {
        "measured_repetitions": 2,
        "per_query_metrics": [{}, {}],
        "mean_recall_at_k": 1.0,
        "mean_ndcg_at_k": 1.0,
        "mean_mrr": 1.0,
        "mean_ap": 1.0,
        "diagnostics_complete": True,
        "degradation_rate": 0.0,
        "duplicate_result_count": 0,
        "semantic_abstention_rate": 0.0,
    }

    assert validate_provider_report(
        report, ProviderEvidencePolicy(min_repetitions=2, require_diagnostics=True)
    ) == []

    unavailable = dict(report)
    unavailable["diagnostics_complete"] = None
    assert "diagnostics are unavailable or incomplete" in validate_provider_report(
        unavailable,
        ProviderEvidencePolicy(min_repetitions=2, require_diagnostics=True),
    )


def test_provider_evidence_returns_deterministic_acceptance_report() -> None:
    """Acceptance serialization sorts deltas and reports stability failures."""
    baseline = {
        "mean_recall_at_k": 1.0,
        "mean_ndcg_at_k": 1.0,
        "mean_mrr": 1.0,
        "mean_ap": 1.0,
        "diagnostics_complete": True,
        "degradation_rate": 0.0,
        "duplicate_result_count": 0,
        "semantic_abstention_rate": 0.1,
    }
    candidate = dict(baseline)
    candidate.update(
        degradation_rate=0.2,
        duplicate_result_count=1,
        semantic_abstention_rate=0.3,
    )

    result = compare_provider_report(
        candidate,
        baseline,
        ProviderEvidencePolicy(
            require_diagnostics=True,
            max_degradation_rate=0.1,
            max_duplicate_result_count=0,
            max_semantic_abstention_rate_delta=0.1,
        ),
    )

    assert result["passed"] is False
    assert list(result.to_dict()["deltas"]) == [
        "mean_ap",
        "mean_mrr",
        "mean_ndcg_at_k",
        "mean_recall_at_k",
    ]
    assert any("degradation_rate" in failure for failure in result.failures)


def test_provider_evidence_treats_unavailable_baseline_metrics_as_unset() -> None:
    """
    Null baseline quality fields do not crash comparison or create a floor.
    """
    baseline = {field: None for field in ("mean_recall_at_k", "mean_ndcg_at_k", "mean_mrr", "mean_ap")}
    candidate = {
        "mean_recall_at_k": 0.9,
        "mean_ndcg_at_k": 0.8,
        "mean_mrr": 0.7,
        "mean_ap": 0.6,
    }

    result = compare_provider_report(candidate, baseline, ProviderEvidencePolicy())

    assert result.passed is True


def test_provider_evidence_rejects_incompatible_metadata_before_deltas() -> None:
    """
    Strictly comparable reports reject metadata mismatches before quality deltas.
    """
    baseline = {
        "metadata": {
            "corpus_version": "v1",
            "split": "test",
            "backend": "sqlite",
            "model_fingerprint": "model-v1",
            "vector_dimension": 384,
            "indexing_fingerprint": "index-v1",
            "ann_build_fingerprint": "ann-build-v1",
            "ann_query_policy_fingerprint": "ann-query-v1",
            "routing_fingerprint": "routing-v1",
            "fusion_fingerprint": "fusion-v1",
            "config_fingerprint": "config-v1",
            "environment_fingerprint": "env-v1",
        },
        "mean_recall_at_k": 0.8,
    }
    candidate = dict(baseline)
    candidate["metadata"] = dict(baseline["metadata"])
    candidate["metadata"]["backend"] = "postgres"
    candidate["mean_recall_at_k"] = 0.9

    result = compare_provider_report(
        candidate,
        baseline,
        ProviderEvidencePolicy(require_metadata_compatibility=True),
    )

    assert result.passed is False
    assert any(
        "metadata incompatible: backend='postgres'" in failure
        for failure in result.failures
    )
    assert result.deltas == ()


def test_provider_evidence_strict_metadata_requires_the_report_contract() -> None:
    """
    Strict metadata comparison rejects legacy reports without contract fields.
    """
    report = {"mean_recall_at_k": 1.0}

    result = compare_provider_report(
        report,
        report,
        ProviderEvidencePolicy(require_metadata_compatibility=True),
    )

    assert result.passed is False
    assert result.failures == (
        "metadata compatibility requires metadata on both reports",
    )


def test_provider_evidence_allows_omitted_metadata_by_default() -> None:
    """
    Default comparison remains compatible with reports from before metadata.
    """
    candidate = _report()
    baseline = {
        field: candidate[field]
        for field in (
            "mean_recall_at_k",
            "mean_ndcg_at_k",
            "mean_mrr",
            "mean_ap",
        )
    }
    candidate.pop("metadata")
    candidate["mean_recall_at_k"] = 0.9
    baseline["mean_recall_at_k"] = 0.8

    result = compare_provider_report(candidate, baseline, ProviderEvidencePolicy())

    assert result.passed is True
    assert dict(result.deltas)["mean_recall_at_k"] == pytest.approx(0.1)
