import math
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime

from searchkernel.search.normalization import normalize_scores
from searchkernel.search.time_scoring import (
    TierConfig,
    TimeScoreMode,
    apply_time_boost,
)


def rrf_score(rank: int, k: int):
    return 1 / (k + rank)


def fuse_reciprocal_rank(
    rankings: Iterable[Sequence[str]] | Mapping[str, Sequence[str]],
    k: float = 60.0,
    strategy_weights: Mapping[str, float] | None = None,
    *,
    source_weights: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Fuse rankings with optional per-strategy reliability weights.

    This is the default rank-based fusion path: it uses only one-based rank
    positions and does not normalize raw retrieval scores. Mapping inputs
    retain strategy names so weighted calls never need source fields in domain
    models.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if strategy_weights is not None and source_weights is not None:
        raise ValueError("provide only one of strategy_weights or source_weights")
    weights = strategy_weights or source_weights or {}

    scores: dict[str, float] = {}
    if isinstance(rankings, Mapping):
        named_rankings = rankings.items()
    else:
        named_rankings = enumerate(rankings)
    for strategy, ranking in named_rankings:
        weight = float(weights.get(str(strategy), 1.0))
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("strategy weights must be finite and non-negative")
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + weight / (k + rank)
    return scores


def weighted_reciprocal_rank(
    rankings: Mapping[str, Sequence[str]],
    *,
    strategy_weights: Mapping[str, float],
    k: float = 60.0,
) -> dict[str, float]:
    """Explicit weighted-RRF entry point for callers that prefer named lanes."""
    return fuse_reciprocal_rank(
        rankings,
        k=k,
        strategy_weights=strategy_weights,
    )


def fuse_calibrated_scores(
    rankings: Mapping[str, Sequence[tuple[str, float]]],
    *,
    strategy_weights: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Fuse scores after min-max normalizing each lane independently.

    A lane's minimum and maximum are computed from that lane's result set, so
    calibrated values express relative relevance within a query and lane.
    This opt-in score-based path is separate from default reciprocal-rank
    fusion, which preserves its existing ranking behavior.
    """
    weights = strategy_weights or {}
    scores: dict[str, float] = {}
    for strategy, ranking in rankings.items():
        raw_scores = [score for _item_id, score in ranking]
        if not all(math.isfinite(score) for score in raw_scores):
            raise ValueError("lane scores must be finite")
        weight = float(weights.get(strategy, 1.0))
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("strategy weights must be finite and non-negative")
        for (item_id, _raw_score), calibrated in zip(
            ranking, normalize_scores(raw_scores)
        ):
            scores[item_id] = scores.get(item_id, 0.0) + weight * calibrated
    return scores


def apply_recency_boost(
    doc_id: str,
    score: float,
    modified_times: dict[str, float],
    tiers: list[tuple[int, float]],
):
    if doc_id not in modified_times:
        return score

    modified_time = modified_times[doc_id]
    timestamp = datetime.fromtimestamp(modified_time, UTC)

    config = TierConfig()
    if len(tiers) >= 1:
        config.recent_days = tiers[0][0]
        config.recent_boost = tiers[0][1]
    if len(tiers) >= 2:
        config.moderate_days = tiers[1][0]
        config.moderate_boost = tiers[1][1]

    return apply_time_boost(score, timestamp, TimeScoreMode.TIERS, config)
