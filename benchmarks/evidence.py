"""Validation and deterministic comparison for checked-in evidence artifacts."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


@dataclass(frozen=True)
class EvidencePolicy:
    """Configurable quality and optional relative-latency gate."""

    min_repetitions: int = 3
    min_mean_recall_at_k: float = 0.0
    min_mean_ndcg_at_k: float = 0.0
    min_mean_mrr: float = 0.0
    min_mean_ap: float = 0.0
    max_latency_regression_ratio: float | None = None
    require_metadata_compatibility: bool = False

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> EvidencePolicy:
        policy = cls(
            min_repetitions=int(values.get("min_repetitions", 3)),
            min_mean_recall_at_k=float(values.get("min_mean_recall_at_k", 0.0)),
            min_mean_ndcg_at_k=float(values.get("min_mean_ndcg_at_k", 0.0)),
            min_mean_mrr=float(values.get("min_mean_mrr", 0.0)),
            min_mean_ap=float(values.get("min_mean_ap", 0.0)),
            max_latency_regression_ratio=(
                None
                if values.get("max_latency_regression_ratio") is None
                else float(values["max_latency_regression_ratio"])
            ),
            require_metadata_compatibility=bool(
                values.get("require_metadata_compatibility", False)
            ),
        )
        if policy.min_repetitions < 1:
            raise ValueError("min_repetitions must be positive")
        if policy.max_latency_regression_ratio is not None and policy.max_latency_regression_ratio < 0:
            raise ValueError("max_latency_regression_ratio must be non-negative")
        return policy


def load_policy(path: Path) -> EvidencePolicy:
    """Load a JSON evidence policy."""
    values = json.loads(path.read_text())
    if not isinstance(values, dict):
        raise TypeError("evidence policy must be a JSON object")
    return EvidencePolicy.from_dict(values)


def validate_report(report: dict[str, Any], policy: EvidencePolicy) -> list[str]:
    """Return schema failures for one serialized evaluation report."""
    failures: list[str] = []
    required = ("golden_set_size", "k", "mode", "warmup_count", "measured_repetitions", "metadata", "per_query_metrics")
    failures.extend(f"missing field: {field}" for field in required if field not in report)
    if failures:
        return failures
    if report["mode"] != "warm":
        failures.append("mode must be warm")
    if not isinstance(report["warmup_count"], int) or report["warmup_count"] < 1:
        failures.append("warmup_count must be positive")
    repetitions = report["measured_repetitions"]
    if not isinstance(repetitions, int) or repetitions < policy.min_repetitions:
        failures.append(f"measured_repetitions must be >= {policy.min_repetitions}")
    metrics = report["per_query_metrics"]
    if not isinstance(metrics, list):
        failures.append("per_query_metrics must be a list")
    elif len(metrics) != report["golden_set_size"] * repetitions:
        failures.append("per_query_metrics count does not match repetitions")
    metadata = report["metadata"]
    if not isinstance(metadata, dict) or not metadata.get("corpus_version") or not metadata.get("backend"):
        failures.append("metadata must identify corpus_version and backend")
    for field in QUALITY_METRICS:
        value = report.get(field)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            failures.append(f"{field} must be finite")
    return failures


def compare_report(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    policy: EvidencePolicy,
) -> dict[str, Any]:
    """Compare stable quality metrics and optionally relative latency."""
    failures: list[str] = []
    metadata_failures = _metadata_compatibility_failures(
        candidate, baseline, policy.require_metadata_compatibility
    )
    if metadata_failures:
        return {
            "passed": False,
            "failures": metadata_failures,
            "deltas": {},
            "latency_regression_ratio": None,
        }

    deltas: dict[str, float] = {}
    for field in QUALITY_METRICS:
        value = candidate.get(field)
        minimum = max(policy.__getattribute__(f"min_{field}"), baseline.get(field, 0.0))
        if not isinstance(value, (int, float)) or value < minimum:
            failures.append(f"{field}={value!r} is below {minimum:.6f}")
        if isinstance(value, (int, float)) and isinstance(baseline.get(field), (int, float)):
            deltas[field] = value - baseline[field]
    baseline_latency = baseline.get("latency_p95_ms")
    candidate_latency = candidate.get("latency_p95_ms")
    latency_ratio = None
    if policy.max_latency_regression_ratio is not None:
        if not isinstance(baseline_latency, (int, float)) or not isinstance(candidate_latency, (int, float)) or baseline_latency <= 0:
            failures.append("latency_p95_ms is unavailable for configured gate")
        else:
            latency_ratio = (candidate_latency - baseline_latency) / baseline_latency
            if latency_ratio > policy.max_latency_regression_ratio:
                failures.append(
                    f"latency_p95 regression={latency_ratio:.6f} exceeds "
                    f"{policy.max_latency_regression_ratio:.6f}"
                )
    return {"passed": not failures, "failures": failures, "deltas": deltas, "latency_regression_ratio": latency_ratio}


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
                f"metadata incompatible: {field}={candidate_value!r} "
                f"does not match baseline {baseline_value!r}"
            )
    return failures
