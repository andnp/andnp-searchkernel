"""Query-relative score normalization for ranked result sets."""

from collections.abc import Sequence


def normalize_scores(scores: Sequence[float]) -> list[float]:
    """Normalize scores to ``[0, 1]`` relative to the supplied result set.

    Each score is mapped with ``(score - min) / (max - min)``.  This is
    query-relative normalization: the input set defines both endpoints, so
    normalized values are not comparable across queries or result sets.

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
