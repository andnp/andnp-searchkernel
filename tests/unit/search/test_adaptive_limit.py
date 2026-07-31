import pytest

from searchkernel.search.adaptive_limit import resolve_adaptive_result_limit


def _resolve(
    scores: list[float],
    *,
    requested_limit: int = 1,
    adaptive_enabled: bool = True,
    maximum_limit: int = 10,
    score_ratio_floor: float = 0.0,
    minimum_score: float = 0.0,
    maximum_score_gap: float = 1.0,
) -> int:
    return resolve_adaptive_result_limit(
        scores,
        requested_limit=requested_limit,
        adaptive_enabled=adaptive_enabled,
        maximum_limit=maximum_limit,
        score_ratio_floor=score_ratio_floor,
        minimum_score=minimum_score,
        maximum_score_gap=maximum_score_gap,
    )


def test_disabled_mode_keeps_requested_limit():
    assert _resolve([0.9, 0.8, 0.7], requested_limit=2, adaptive_enabled=False) == 2


def test_short_result_list_does_not_grow():
    assert _resolve([0.9, 0.8], requested_limit=4) == 2


def test_score_floor_stops_growth():
    assert _resolve(
        [0.9, 0.8, 0.5],
        minimum_score=0.7,
        maximum_score_gap=1.0,
    ) == 2


def test_score_gap_stops_growth():
    assert _resolve(
        [0.9, 0.85, 0.6],
        maximum_score_gap=0.1,
    ) == 2


def test_adaptive_growth_respects_configured_cap():
    assert _resolve([0.9, 0.89, 0.88, 0.87], maximum_limit=3) == 3


@pytest.mark.parametrize("requested_limit", [0, -3])
def test_requested_limit_is_bounded_to_one(requested_limit: int):
    assert _resolve(
        [0.9, 0.8],
        requested_limit=requested_limit,
        adaptive_enabled=False,
    ) == 1


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("maximum_limit", 0),
        ("score_ratio_floor", -0.1),
        ("score_ratio_floor", 1.1),
        ("minimum_score", -0.1),
        ("minimum_score", 1.1),
        ("maximum_score_gap", -0.1),
        ("maximum_score_gap", 1.1),
    ],
)
def test_invalid_configuration_raises_value_error(parameter: str, value: float):
    kwargs = {
        "maximum_limit": 10,
        "score_ratio_floor": 0.0,
        "minimum_score": 0.0,
        "maximum_score_gap": 1.0,
        parameter: value,
    }
    with pytest.raises(ValueError, match=parameter):
        resolve_adaptive_result_limit(
            [0.9],
            requested_limit=1,
            adaptive_enabled=False,
            **kwargs,
        )
