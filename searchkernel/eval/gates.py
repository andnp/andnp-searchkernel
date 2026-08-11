"""Release-quality gates for retrieval evaluation reports."""

from __future__ import annotations

from dataclasses import dataclass

from searchkernel.eval.runner import AbReport


@dataclass(frozen=True, slots=True)
class EvalGatePolicy:
    """Minimum A/B deltas required for a candidate search pipeline."""

    min_recall_at_k_delta: float = 0.0
    min_ndcg_at_k_delta: float = 0.0
    min_mrr_delta: float = 0.0
    min_ap_delta: float = 0.0
    max_latency_p95_delta_ms: float | None = None
    require_diagnostics: bool = False
    max_degradation_rate: float | None = None
    max_duplicate_case_count: int | None = None
    max_duplicate_result_rate: float | None = None
    max_semantic_abstention_rate_delta: float | None = None


@dataclass(frozen=True, slots=True)
class EvalGateResult:
    """Structured result from evaluating an A/B report against a policy."""

    passed: bool
    failures: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "failures": list(self.failures),
        }


class EvaluationGateError(AssertionError):
    """Raised when a report fails an evaluation gate."""


def evaluate_ab_gate(
    report: AbReport,
    policy: EvalGatePolicy | None = None,
) -> EvalGateResult:
    """Check an A/B report against minimum quality and latency deltas."""
    effective_policy = policy or EvalGatePolicy()
    failures: list[str] = []

    _check_delta(
        failures,
        "recall_at_k",
        report.recall_at_k_delta,
        effective_policy.min_recall_at_k_delta,
    )
    _check_delta(
        failures,
        "ndcg_at_k",
        report.ndcg_at_k_delta,
        effective_policy.min_ndcg_at_k_delta,
    )
    _check_delta(
        failures,
        "mrr",
        report.mrr_delta,
        effective_policy.min_mrr_delta,
    )
    _check_delta(
        failures,
        "ap",
        report.ap_delta,
        effective_policy.min_ap_delta,
    )

    max_latency_delta = effective_policy.max_latency_p95_delta_ms
    if max_latency_delta is not None:
        actual_latency_delta = report.latency_p95_delta_ms
        if actual_latency_delta is None:
            failures.append("latency_p95_delta_ms is unavailable")
        elif actual_latency_delta > max_latency_delta:
            failures.append(
                f"latency_p95_delta_ms={actual_latency_delta:.3f} "
                f"exceeds {max_latency_delta:.3f}"
            )

    if effective_policy.require_diagnostics:
        for name, candidate in (
            ("baseline", report.report_a),
            ("candidate", report.report_b),
        ):
            if candidate.diagnostics_complete is not True:
                failures.append(f"{name} diagnostics are unavailable or incomplete")

    max_degradation_rate = effective_policy.max_degradation_rate
    if max_degradation_rate is not None:
        candidate_rate = report.report_b.degradation_rate
        if candidate_rate is None:
            failures.append("degradation_rate is unavailable")
        elif candidate_rate > max_degradation_rate:
            failures.append(
                f"degradation_rate={candidate_rate:.6f} exceeds "
                f"{max_degradation_rate:.6f}"
            )

    max_duplicate_count = effective_policy.max_duplicate_case_count
    if max_duplicate_count is not None:
        duplicate_count = report.report_b.duplicate_result_count
        if duplicate_count > max_duplicate_count:
            failures.append(
                f"duplicate_result_count={duplicate_count} exceeds "
                f"{max_duplicate_count}"
            )

    max_duplicate_rate = effective_policy.max_duplicate_result_rate
    if max_duplicate_rate is not None:
        duplicate_rate = report.report_b.duplicate_result_rate
        if duplicate_rate is None:
            failures.append("duplicate_result_rate is unavailable")
        elif duplicate_rate > max_duplicate_rate:
            failures.append(
                f"duplicate_result_rate={duplicate_rate:.6f} exceeds "
                f"{max_duplicate_rate:.6f}"
            )

    abstention_delta_limit = effective_policy.max_semantic_abstention_rate_delta
    if abstention_delta_limit is not None:
        abstention_delta = _report_delta(
            report.semantic_abstention_rate_delta,
            report.report_a.semantic_abstention_rate,
            report.report_b.semantic_abstention_rate,
        )
        if abstention_delta is None:
            failures.append("semantic_abstention_rate_delta is unavailable")
        elif abstention_delta > abstention_delta_limit:
            failures.append(
                f"semantic_abstention_rate_delta={abstention_delta:.6f} exceeds "
                f"{abstention_delta_limit:.6f}"
            )

    return EvalGateResult(passed=not failures, failures=tuple(failures))


def enforce_ab_gate(
    report: AbReport,
    policy: EvalGatePolicy | None = None,
) -> None:
    """Raise a structured assertion if an A/B report fails its gate."""
    result = evaluate_ab_gate(report, policy)
    if not result.passed:
        raise EvaluationGateError("; ".join(result.failures))


def _check_delta(
    failures: list[str],
    name: str,
    actual: float | None,
    minimum: float,
) -> None:
    if actual is None:
        failures.append(f"{name}_delta is unavailable")
    elif actual < minimum:
        failures.append(f"{name}_delta={actual:.6f} is below {minimum:.6f}")


def _report_delta(
    explicit: float | None,
    baseline: float | None,
    candidate: float | None,
) -> float | None:
    if explicit is not None:
        return explicit
    if baseline is None or candidate is None:
        return None
    return candidate - baseline
