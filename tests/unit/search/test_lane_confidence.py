import math

import pytest

from searchkernel.search.lane_confidence import (
    graph_confidence,
    keyword_confidence,
    lane_confidence,
    vector_confidence,
)


@pytest.mark.parametrize(
    ("raw_score", "expected"),
    [
        (1.0, 1.0),
        (0.0, 0.5),
        (-1.0, 0.0),
    ],
)
def test_vector_confidence_anchor_points(raw_score: float, expected: float) -> None:
    assert vector_confidence(raw_score) == pytest.approx(expected)


@pytest.mark.parametrize("raw_score", [-1.0000001, -5.0, 1.0000001, 5.0])
def test_vector_confidence_clamps_overshoot(raw_score: float) -> None:
    confidence = vector_confidence(raw_score)

    assert 0.0 <= confidence <= 1.0


def test_vector_confidence_monotonic() -> None:
    sweep = [-1.0, -0.5, -0.1, 0.0, 0.3, 0.75, 1.0]

    confidences = [vector_confidence(score) for score in sweep]

    assert confidences == sorted(confidences)


@pytest.mark.parametrize(
    ("raw_score", "expected"),
    [
        (0.0, 0.0),
        (-3.0, 0.0),
        (10.0, 0.5),
    ],
)
def test_keyword_confidence_anchor_points(raw_score: float, expected: float) -> None:
    assert keyword_confidence(raw_score, saturation_k=10.0) == pytest.approx(expected)


def test_keyword_confidence_approaches_but_never_reaches_one() -> None:
    confidence = keyword_confidence(1_000_000.0, saturation_k=10.0)

    assert 0.0 < confidence < 1.0


def test_keyword_confidence_monotonic() -> None:
    sweep = [0.0, 1.0, 5.0, 10.0, 50.0, 300.0]

    confidences = [keyword_confidence(score, saturation_k=10.0) for score in sweep]

    assert confidences == sorted(confidences)


@pytest.mark.parametrize("saturation_k", [0.0, -1.0, math.inf, math.nan])
def test_keyword_confidence_rejects_invalid_saturation_k(saturation_k: float) -> None:
    with pytest.raises(ValueError, match="saturation_k"):
        keyword_confidence(1.0, saturation_k=saturation_k)


@pytest.mark.parametrize(
    ("raw_score", "expected"),
    [
        (0.5, 0.5),
        (0.0, 0.0),
        (1.0, 1.0),
    ],
)
def test_graph_confidence_anchor_points(raw_score: float, expected: float) -> None:
    assert graph_confidence(raw_score) == pytest.approx(expected)


@pytest.mark.parametrize("raw_score", [-0.5, -1.0000001, 1.0000001, 2.0])
def test_graph_confidence_clamps(raw_score: float) -> None:
    confidence = graph_confidence(raw_score)

    assert 0.0 <= confidence <= 1.0


def test_graph_confidence_monotonic() -> None:
    sweep = [-0.5, 0.0, 0.2, 0.6, 1.0, 1.5]

    confidences = [graph_confidence(score) for score in sweep]

    assert confidences == sorted(confidences)


@pytest.mark.parametrize(
    ("lane", "raw_score", "expected"),
    [
        ("vector", 1.0, 1.0),
        ("keyword", 10.0, 0.5),
        ("graph", 0.5, 0.5),
    ],
)
def test_lane_confidence_dispatches_to_matching_transform(
    lane: str, raw_score: float, expected: float
) -> None:
    assert lane_confidence(lane, raw_score, saturation_k=10.0) == pytest.approx(
        expected
    )


def test_lane_confidence_rejects_unknown_lane() -> None:
    with pytest.raises(ValueError, match="unknown lane"):
        lane_confidence("nonsense", 1.0, saturation_k=10.0)


@pytest.mark.parametrize(
    "transform",
    [vector_confidence, graph_confidence],
)
@pytest.mark.parametrize("raw_score", [math.nan, math.inf, -math.inf])
def test_single_arg_transforms_reject_non_finite_scores(transform, raw_score) -> None:
    with pytest.raises(ValueError, match="finite"):
        transform(raw_score)


@pytest.mark.parametrize("raw_score", [math.nan, math.inf, -math.inf])
def test_keyword_confidence_rejects_non_finite_scores(raw_score: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        keyword_confidence(raw_score, saturation_k=10.0)


@pytest.mark.parametrize("raw_score", [math.nan, math.inf, -math.inf])
def test_lane_confidence_rejects_non_finite_scores(raw_score: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        lane_confidence("vector", raw_score, saturation_k=10.0)
