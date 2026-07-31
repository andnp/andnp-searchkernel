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
