from collections.abc import Iterable, Sequence
from datetime import UTC, datetime

from searchkernel.search.time_scoring import (
    TierConfig,
    TimeScoreMode,
    apply_time_boost,
)


def rrf_score(rank: int, k: int):
    return 1 / (k + rank)


def fuse_reciprocal_rank(
    rankings: Iterable[Sequence[str]], k: float = 60.0
) -> dict[str, float]:
    if k <= 0:
        raise ValueError("k must be positive")

    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1 / (k + rank)
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
