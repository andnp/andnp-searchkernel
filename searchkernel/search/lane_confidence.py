import math

__all__ = [
    "graph_confidence",
    "keyword_confidence",
    "lane_confidence",
    "vector_confidence",
]


def _require_finite_score(raw_score: float) -> None:
    if not math.isfinite(raw_score):
        raise ValueError("raw_score must be finite")


def vector_confidence(raw_score: float) -> float:
    """Map cosine similarity onto a comparable [0, 1] confidence.

    Cosine similarity is already bounded, but its native range straddles
    zero while the other lanes' confidences are one-sided, so a lone
    absolute threshold applied to raw cosine values does not mean the same
    thing as the same threshold applied to a keyword or graph score. Folding
    [-1, 1] onto [0, 1] puts it on the same footing.
    """
    _require_finite_score(raw_score)
    confidence = (raw_score + 1.0) / 2.0
    return min(1.0, max(0.0, confidence))


def keyword_confidence(raw_score: float, *, saturation_k: float) -> float:
    """Map an unbounded lexical score onto [0, 1) by saturation.

    BM25-derived scores (plus heuristic boosts) have no fixed ceiling and
    their magnitude carries no intrinsic meaning across queries, unlike
    cosine similarity or a discounted seed confidence. A saturating ratio
    turns "bigger is better, without limit" into a value that a fixed
    absolute threshold can compare meaningfully: it lands at exactly 0.5
    when raw_score equals saturation_k, and approaches 1.0 only in the
    limit as raw_score grows without bound.
    """
    _require_finite_score(raw_score)
    if not math.isfinite(saturation_k) or saturation_k <= 0:
        raise ValueError("saturation_k must be finite and positive")
    if raw_score <= 0:
        return 0.0
    return raw_score / (raw_score + saturation_k)


def graph_confidence(raw_score: float) -> float:
    """Map a seed-confidence-times-discount contribution onto [0, 1].

    The graph lane's raw score is already a probability-like seed
    confidence scaled by an edge discount in [0, 1], so no rescaling is
    needed to make it commensurable with the other lanes; only clamping is
    required to tolerate compounding rounding error from chained discounts.
    """
    _require_finite_score(raw_score)
    return min(1.0, max(0.0, raw_score))


def lane_confidence(lane: str, raw_score: float, *, saturation_k: float) -> float:
    """Dispatch a raw lane score to its absolute confidence transfer function."""
    if lane == "vector":
        return vector_confidence(raw_score)
    if lane == "keyword":
        return keyword_confidence(raw_score, saturation_k=saturation_k)
    if lane == "graph":
        return graph_confidence(raw_score)
    raise ValueError(f"unknown lane: {lane!r}")
