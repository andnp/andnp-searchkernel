import math
import random

import pytest

from searchkernel.search.isotonic import (
    IsotonicCurve,
    calibrated_lane_confidence,
    fit_isotonic,
)
from searchkernel.search.lane_confidence import lane_confidence


def _separable_dataset(n_per_class: int = 60) -> tuple[list[float], list[int]]:
    """Perfectly separable data: negatives near 0, positives near 10."""
    rng = random.Random(0)
    negatives = [rng.uniform(0.0, 1.0) for _ in range(n_per_class)]
    positives = [rng.uniform(9.0, 10.0) for _ in range(n_per_class)]
    scores = negatives + positives
    labels = [0] * n_per_class + [1] * n_per_class
    return scores, labels


def test_fit_isotonic_known_answer_separates_clusters() -> None:
    scores, labels = _separable_dataset()

    curve = fit_isotonic(scores, labels, minimum_samples=50)

    assert curve is not None
    assert curve.confidence(0.5) < 0.1
    assert curve.confidence(9.5) > 0.9


def test_fit_isotonic_returns_none_below_minimum_samples() -> None:
    scores = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    labels = [0, 0, 0, 1, 1, 1]

    curve = fit_isotonic(scores, labels, minimum_samples=50)

    assert curve is None


def test_fit_isotonic_returns_none_for_empty_input() -> None:
    assert fit_isotonic([], [], minimum_samples=1) is None


def test_fit_isotonic_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="paired"):
        fit_isotonic([1.0, 2.0], [0], minimum_samples=1)


@pytest.mark.parametrize("raw_score", [math.nan, math.inf, -math.inf])
def test_fit_isotonic_rejects_non_finite_scores(raw_score: float) -> None:
    scores = [0.0] * 60 + [raw_score]
    labels = [0] * 30 + [1] * 30 + [1]

    with pytest.raises(ValueError, match="finite"):
        fit_isotonic(scores, labels, minimum_samples=10)


def test_fit_isotonic_rejects_non_boolean_labels() -> None:
    scores = list(range(60))
    labels = [0] * 30 + [2] * 30

    with pytest.raises(ValueError, match="boolean"):
        fit_isotonic(scores, labels, minimum_samples=10)


def test_fit_isotonic_rejects_negative_minimum_samples() -> None:
    with pytest.raises(ValueError, match="minimum_samples"):
        fit_isotonic([1.0, 2.0], [0, 1], minimum_samples=-1)


def test_fit_isotonic_all_positive_labels_returns_none_under_default_gate() -> None:
    scores = [float(i) for i in range(60)]
    labels = [1] * 60

    curve = fit_isotonic(scores, labels, minimum_samples=50)

    assert curve is None


def test_fit_isotonic_all_positive_labels_collapses_when_gate_disabled() -> None:
    scores = [1.0, 2.0, 3.0]
    labels = [1, 1, 1]

    curve = fit_isotonic(scores, labels, minimum_samples=0)

    assert curve is not None
    assert curve.confidence(1.0) == pytest.approx(1.0)
    assert curve.confidence(100.0) == pytest.approx(1.0)
    assert curve.confidence(-100.0) == pytest.approx(1.0)


def test_fit_isotonic_all_negative_labels_collapses_when_gate_disabled() -> None:
    scores = [1.0, 2.0, 3.0]
    labels = [0, 0, 0]

    curve = fit_isotonic(scores, labels, minimum_samples=0)

    assert curve is not None
    assert curve.confidence(1.0) == pytest.approx(0.0)
    assert curve.confidence(100.0) == pytest.approx(0.0)


def test_fit_isotonic_duplicate_x_with_conflicting_labels_averages() -> None:
    # Same raw score of 5.0 seen with both labels: relevance rate is 0.5,
    # flanked by clear negatives below and positives above.
    scores = [1.0] * 30 + [5.0] * 20 + [5.0] * 20 + [9.0] * 30
    labels = [0] * 30 + [0] * 20 + [1] * 20 + [1] * 30

    curve = fit_isotonic(scores, labels, minimum_samples=20)

    assert curve is not None
    assert curve.confidence(5.0) == pytest.approx(0.5, abs=0.05)


def test_fit_isotonic_interpolates_inside_a_pooled_plateau() -> None:
    curve = fit_isotonic(
        [0.0, 1.0, 2.0, 3.0],
        [0, 1, 0, 1],
        minimum_samples=0,
    )

    assert curve is not None
    assert curve.confidence(1.0) == pytest.approx(1 / 3)
    assert curve.confidence(2.0) == pytest.approx(2 / 3)


def test_fit_isotonic_monotonic_across_sweep() -> None:
    rng = random.Random(1)
    scores = [rng.uniform(0.0, 20.0) for _ in range(400)]
    labels = [1 if score + rng.gauss(0.0, 2.0) > 10.0 else 0 for score in scores]

    curve = fit_isotonic(scores, labels, minimum_samples=20)

    assert curve is not None
    sweep = [i * 0.05 for i in range(400)]  # 0.0 .. 19.95
    confidences = [curve.confidence(score) for score in sweep]

    assert confidences == sorted(confidences)


def test_isotonic_curve_confidence_rejects_non_finite() -> None:
    curve = IsotonicCurve(knot_x=(0.0, 1.0), knot_y=(0.0, 1.0))

    with pytest.raises(ValueError, match="finite"):
        curve.confidence(math.nan)


def test_isotonic_curve_rejects_non_increasing_knot_x() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        IsotonicCurve(knot_x=(1.0, 1.0), knot_y=(0.0, 1.0))


def test_isotonic_curve_rejects_non_monotonic_knot_y() -> None:
    with pytest.raises(ValueError, match="non-decreasing"):
        IsotonicCurve(knot_x=(0.0, 1.0), knot_y=(1.0, 0.0))


def test_isotonic_curve_rejects_empty_knots() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        IsotonicCurve(knot_x=(), knot_y=())


def test_isotonic_curve_round_trips_through_dict() -> None:
    curve = IsotonicCurve(knot_x=(0.0, 5.0, 10.0), knot_y=(0.1, 0.5, 0.9))

    restored = IsotonicCurve.from_dict(curve.to_dict())

    assert restored == curve
    assert restored.confidence(2.5) == pytest.approx(curve.confidence(2.5))


def test_isotonic_curve_interpolates_between_knots() -> None:
    curve = IsotonicCurve(knot_x=(0.0, 10.0), knot_y=(0.0, 1.0))

    assert curve.confidence(5.0) == pytest.approx(0.5)
    assert curve.confidence(2.5) == pytest.approx(0.25)


def test_isotonic_curve_flat_extrapolation_outside_knot_range() -> None:
    curve = IsotonicCurve(knot_x=(1.0, 2.0), knot_y=(0.2, 0.8))

    assert curve.confidence(-5.0) == pytest.approx(0.2)
    assert curve.confidence(50.0) == pytest.approx(0.8)


def test_calibrated_lane_confidence_uses_fitted_curve_when_present() -> None:
    curve = IsotonicCurve(knot_x=(0.0, 10.0), knot_y=(0.0, 1.0))
    confidence = calibrated_lane_confidence({"vector": curve}, saturation_k=10.0)

    assert confidence("vector", 5.0) == pytest.approx(0.5)


def test_calibrated_lane_confidence_falls_back_to_parametric_when_absent() -> None:
    curve = IsotonicCurve(knot_x=(0.0, 10.0), knot_y=(0.0, 1.0))
    confidence = calibrated_lane_confidence({"vector": curve}, saturation_k=10.0)

    assert confidence("keyword", 10.0) == pytest.approx(
        lane_confidence("keyword", 10.0, saturation_k=10.0)
    )
    assert confidence("graph", 0.5) == pytest.approx(
        lane_confidence("graph", 0.5, saturation_k=10.0)
    )
