import pytest

from searchkernel.search.normalization import normalize_scores


def test_normalize_scores_bounds_and_order() -> None:
    normalized = normalize_scores([10.0, 5.0, 0.0])

    assert normalized == [1.0, 0.5, 0.0]
    assert all(0.0 <= score <= 1.0 for score in normalized)


def test_normalize_scores_preserves_monotonic_order() -> None:
    normalized = normalize_scores([4.0, 3.0, 3.0, 1.0])

    assert normalized == sorted(normalized, reverse=True)
    assert normalized[0] > normalized[1] == normalized[2] > normalized[3]


@pytest.mark.parametrize(
    ("scores", "expected"),
    [
        ([], []),
        ([0.0], [1.0]),
        ([0.0, 0.0], [1.0, 1.0]),
        ([-2.0, -2.0, -2.0], [1.0, 1.0, 1.0]),
    ],
)
def test_normalize_scores_handles_empty_zero_and_ties(
    scores: list[float],
    expected: list[float],
) -> None:
    assert normalize_scores(scores) == expected
