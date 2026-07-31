"""Pure adaptive result-limit resolution for ranked scores."""

from __future__ import annotations

from collections.abc import Sequence


def resolve_adaptive_result_limit(
    scores: Sequence[float],
    *,
    requested_limit: int,
    adaptive_enabled: bool,
    maximum_limit: int,
    score_ratio_floor: float,
    minimum_score: float,
    maximum_score_gap: float,
) -> int:
    """Resolve how many ranked results to return.

    ``scores`` must already be sorted in descending order. Adaptive growth
    includes each next score while it remains above the absolute/relative
    score floor and within the maximum adjacent score gap.
    """
    _validate_configuration(
        maximum_limit=maximum_limit,
        score_ratio_floor=score_ratio_floor,
        minimum_score=minimum_score,
        maximum_score_gap=maximum_score_gap,
    )

    bounded_requested_limit = max(requested_limit, 1)
    if len(scores) <= bounded_requested_limit or not adaptive_enabled:
        return min(len(scores), bounded_requested_limit)

    adaptive_cap = max(bounded_requested_limit, maximum_limit)
    result_limit = bounded_requested_limit
    top_score = scores[0]
    minimum_ratio_score = top_score * score_ratio_floor
    minimum_accepted_score = max(minimum_score, minimum_ratio_score)

    while result_limit < len(scores) and result_limit < adaptive_cap:
        previous_score = scores[result_limit - 1]
        candidate_score = scores[result_limit]
        if candidate_score < minimum_accepted_score:
            break
        if previous_score - candidate_score > maximum_score_gap:
            break
        result_limit += 1

    return result_limit


def _validate_configuration(
    *,
    maximum_limit: int,
    score_ratio_floor: float,
    minimum_score: float,
    maximum_score_gap: float,
) -> None:
    if maximum_limit < 1:
        raise ValueError("maximum_limit must be >= 1")
    if not 0.0 <= score_ratio_floor <= 1.0:
        raise ValueError("score_ratio_floor must be in [0.0, 1.0]")
    if not 0.0 <= minimum_score <= 1.0:
        raise ValueError("minimum_score must be in [0.0, 1.0]")
    if not 0.0 <= maximum_score_gap <= 1.0:
        raise ValueError("maximum_score_gap must be in [0.0, 1.0]")
