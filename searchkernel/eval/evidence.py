"""Provider-neutral evidence validation and acceptance reports."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, TypeGuard

QUALITY_METRICS = (
    "mean_recall_at_k",
    "mean_ndcg_at_k",
    "mean_mrr",
    "mean_ap",
)
METADATA_COMPATIBILITY_FIELDS = (
    "corpus_version",
    "backend",
    "model_fingerprint",
    "config_fingerprint",
    "environment_fingerprint",
)


@dataclass(frozen=True, slots=True)
class EvidencePolicy:
    """Optional, provider-neutral evidence requirements."""

    min_repetitions: int = 1
    min_mean_recall_at_k: float = 0.0
    min_mean_ndcg_at_k: float = 0.0
    min_mean_mrr: float = 0.0
    min_mean_ap: float = 0.0
    max_latency_regression_ratio: float | None = None
    require_diagnostics: bool = False
    max_degradation_rate: float | None = None
    max_duplicate_result_count: int | None = None
    max_semantic_abstention_rate_delta: float | None = None
    require_metadata_compatibility: bool = False

    def __post_init__(self) -> None:
        if self.min_repetitions < 1:
            raise ValueError("min_repetitions must be positive")
        if (
            self.max_latency_regression_ratio is not None
            and self.max_latency_regression_ratio < 0
        ):
            raise ValueError("max_latency_regression_ratio must be non-negative")
        if self.max_degradation_rate is not None and self.max_degradation_rate < 0:
            raise ValueError("max_degradation_rate must be non-negative")
        if (
            self.max_duplicate_result_count is not None
            and self.max_duplicate_result_count < 0
        ):
            raise ValueError("max_duplicate_result_count must be non-negative")
        if (
            self.max_semantic_abstention_rate_delta is not None
            and self.max_semantic_abstention_rate_delta < 0
        ):
            raise ValueError(
                "max_semantic_abstention_rate_delta must be non-negative"
            )


@dataclass(frozen=True, slots=True)
class AcceptanceReport:
    """Deterministic, serializable result of an evidence comparison."""

    passed: bool
    failures: tuple[str, ...] = ()
    deltas: tuple[tuple[str, float], ...] = ()
    latency_regression_ratio: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize stable keys and sorted metric deltas."""
        return {
            "passed": self.passed,
            "failures": list(self.failures),
            "deltas": {name: value for name, value in sorted(self.deltas)},
            "latency_regression_ratio": self.latency_regression_ratio,
        }

    def __getitem__(self, key: str) -> object:
        """Allow callers to consume the report like the legacy mapping."""
        return self.to_dict()[key]


def validate_report(report: dict[str, Any], policy: EvidencePolicy) -> list[str]:
    """Return provider-neutral schema and evidence failures."""
    failures: list[str] = []
    required = ("measured_repetitions", "per_query_metrics")
    failures.extend(f"missing field: {field}" for field in required if field not in report)
    if failures:
        return failures

    repetitions = report["measured_repetitions"]
    if not isinstance(repetitions, int) or repetitions < policy.min_repetitions:
        failures.append(f"measured_repetitions must be >= {policy.min_repetitions}")
    metrics = report["per_query_metrics"]
    if not isinstance(metrics, list):
        failures.append("per_query_metrics must be a list")
    elif isinstance(report.get("golden_set_size"), int) and isinstance(
        repetitions, int
    ) and len(metrics) != report["golden_set_size"] * repetitions:
        failures.append("per_query_metrics count does not match repetitions")

    if policy.require_diagnostics and report.get("diagnostics_complete") is not True:
        failures.append("diagnostics are unavailable or incomplete")
    if (
        policy.max_degradation_rate is not None
        and not _finite_number(report.get("degradation_rate"))
    ):
        failures.append("degradation_rate is unavailable")
    if (
        policy.max_duplicate_result_count is not None
        and not isinstance(report.get("duplicate_result_count"), int)
    ):
        failures.append("duplicate_result_count is unavailable")

    for field in QUALITY_METRICS:
        if not _finite_number(report.get(field)):
            failures.append(f"{field} must be finite")
    return failures


def compare_report(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    policy: EvidencePolicy,
) -> AcceptanceReport:
    """Compare quality, diagnostics, duplicates, abstention, and latency."""
    failures: list[str] = []
    metadata_failures = _metadata_compatibility_failures(
        candidate, baseline, policy.require_metadata_compatibility
    )
    if metadata_failures:
        return AcceptanceReport(passed=False, failures=tuple(metadata_failures))

    deltas: dict[str, float] = {}
    for field in QUALITY_METRICS:
        value = candidate.get(field)
        raw_baseline_value = baseline.get(field)
        baseline_value = (
            raw_baseline_value
            if isinstance(raw_baseline_value, (int, float))
            else 0.0
        )
        minimum = max(getattr(policy, f"min_{field}"), baseline_value)
        if not isinstance(value, (int, float)) or value < minimum:
            failures.append(f"{field}={value!r} is below {minimum:.6f}")
        if isinstance(value, (int, float)) and isinstance(baseline_value, (int, float)):
            deltas[field] = value - baseline_value

    if policy.require_diagnostics:
        for name, report in (("baseline", baseline), ("candidate", candidate)):
            if report.get("diagnostics_complete") is not True:
                failures.append(f"{name} diagnostics are unavailable or incomplete")

    candidate_degradation = candidate.get("degradation_rate")
    if policy.max_degradation_rate is not None:
        if not _finite_number(candidate_degradation):
            failures.append("degradation_rate is unavailable")
        elif candidate_degradation > policy.max_degradation_rate:
            failures.append(
                f"degradation_rate={candidate_degradation:.6f} exceeds "
                f"{policy.max_degradation_rate:.6f}"
            )

    duplicate_count = candidate.get("duplicate_result_count")
    if policy.max_duplicate_result_count is not None:
        if not isinstance(duplicate_count, int):
            failures.append("duplicate_result_count is unavailable")
        elif duplicate_count > policy.max_duplicate_result_count:
            failures.append(
                f"duplicate_result_count={duplicate_count} exceeds "
                f"{policy.max_duplicate_result_count}"
            )

    latency_ratio: float | None = None
    if policy.max_latency_regression_ratio is not None:
        baseline_latency = baseline.get("latency_p95_ms")
        candidate_latency = candidate.get("latency_p95_ms")
        if (
            not isinstance(baseline_latency, (int, float))
            or not isinstance(candidate_latency, (int, float))
            or baseline_latency <= 0
        ):
            failures.append("latency_p95_ms is unavailable for configured gate")
        else:
            latency_ratio = (candidate_latency - baseline_latency) / baseline_latency
            if latency_ratio > policy.max_latency_regression_ratio:
                failures.append(
                    f"latency_p95 regression={latency_ratio:.6f} exceeds "
                    f"{policy.max_latency_regression_ratio:.6f}"
                )

    if policy.max_semantic_abstention_rate_delta is not None:
        baseline_rate = baseline.get("semantic_abstention_rate")
        candidate_rate = candidate.get("semantic_abstention_rate")
        if not _finite_number(baseline_rate) or not _finite_number(candidate_rate):
            failures.append("semantic_abstention_rate_delta is unavailable")
        elif candidate_rate - baseline_rate > policy.max_semantic_abstention_rate_delta:
            failures.append(
                "semantic_abstention_rate_delta="
                f"{candidate_rate - baseline_rate:.6f} exceeds "
                f"{policy.max_semantic_abstention_rate_delta:.6f}"
            )

    return AcceptanceReport(
        passed=not failures,
        failures=tuple(failures),
        deltas=tuple(sorted(deltas.items())),
        latency_regression_ratio=latency_ratio,
    )


def _finite_number(value: object) -> TypeGuard[int | float]:
    return isinstance(value, (int, float)) and math.isfinite(value)


def _metadata_compatibility_failures(
    candidate: dict[str, Any], baseline: dict[str, Any], strict: bool
) -> list[str]:
    """Return incompatibilities while tolerating omitted legacy metadata."""
    candidate_metadata = candidate.get("metadata")
    baseline_metadata = baseline.get("metadata")
    if not strict and not isinstance(candidate_metadata, dict):
        return []
    if not strict and not isinstance(baseline_metadata, dict):
        return []
    if not isinstance(candidate_metadata, dict) or not isinstance(baseline_metadata, dict):
        return ["metadata compatibility requires metadata on both reports"]

    failures: list[str] = []
    for field in METADATA_COMPATIBILITY_FIELDS:
        candidate_value = candidate_metadata.get(field)
        baseline_value = baseline_metadata.get(field)
        if strict and (not candidate_value or not baseline_value):
            failures.append(f"metadata field unavailable for compatibility: {field}")
        elif (
            candidate_value is not None
            and baseline_value is not None
            and candidate_value != baseline_value
        ):
            failures.append(
                f"metadata incompatible: {field}={candidate_value!r}"
            )
    return failures
