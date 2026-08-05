"""Query-relative score normalization for ranked result sets."""

from collections.abc import Sequence


def normalize_scores(scores: Sequence[float]) -> list[float]:
    """Normalize scores to ``[0, 1]`` relative to one result set.

    A tied result set is treated as uniformly top-ranked because no result
    has less relevance than another within that query.
    """
    if not scores:
        return []

    minimum = min(scores)
    maximum = max(scores)
    if minimum == maximum:
        return [1.0] * len(scores)

    span = maximum - minimum
    return [(score - minimum) / span for score in scores]
