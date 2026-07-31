from searchkernel.eval.gates import (
    EvalGatePolicy,
    EvaluationGateError,
    enforce_ab_gate,
    evaluate_ab_gate,
)
from searchkernel.eval.runner import AbReport, EvalReport


def _report() -> EvalReport:
    return EvalReport(
        golden_set_size=1,
        k=10,
        mean_recall_at_k=0.8,
        mean_ndcg_at_k=0.7,
        mean_mrr=0.6,
        mean_ap=0.5,
        latency_p95_ms=120.0,
    )


def test_evaluate_ab_gate_accepts_report_within_policy() -> None:
    baseline = _report()
    candidate = _report()
    candidate.mean_recall_at_k = 0.9
    candidate.mean_ndcg_at_k = 0.8
    candidate.mean_mrr = 0.7
    candidate.mean_ap = 0.6
    candidate.latency_p95_ms = 125.0

    result = evaluate_ab_gate(
        AbReport(
            report_a=baseline,
            report_b=candidate,
            recall_at_k_delta=0.1,
            ndcg_at_k_delta=0.1,
            mrr_delta=0.1,
            ap_delta=0.1,
            latency_p95_delta_ms=5.0,
        ),
        EvalGatePolicy(
            min_recall_at_k_delta=0.0,
            min_ndcg_at_k_delta=0.0,
            min_mrr_delta=0.0,
            min_ap_delta=0.0,
            max_latency_p95_delta_ms=10.0,
        ),
    )

    assert result.passed is True
    assert result.failures == ()


def test_evaluate_ab_gate_reports_quality_and_latency_failures() -> None:
    report = AbReport(
        report_a=_report(),
        report_b=_report(),
        recall_at_k_delta=-0.1,
        ndcg_at_k_delta=None,
        mrr_delta=0.0,
        ap_delta=0.0,
        latency_p95_delta_ms=25.0,
    )

    result = evaluate_ab_gate(
        report,
        EvalGatePolicy(
            min_recall_at_k_delta=0.0,
            min_ndcg_at_k_delta=0.0,
            max_latency_p95_delta_ms=10.0,
        ),
    )

    assert result.passed is False
    assert "recall_at_k_delta" in result.failures[0]
    assert "ndcg_at_k_delta is unavailable" in result.failures
    assert "latency_p95_delta_ms" in result.failures[2]


def test_enforce_ab_gate_raises_on_failure() -> None:
    report = AbReport(
        report_a=_report(),
        report_b=_report(),
        recall_at_k_delta=-0.01,
        ndcg_at_k_delta=0.0,
        mrr_delta=0.0,
        ap_delta=0.0,
    )

    try:
        enforce_ab_gate(report)
    except EvaluationGateError as error:
        assert "recall_at_k_delta" in str(error)
    else:
        raise AssertionError("expected evaluation gate failure")
