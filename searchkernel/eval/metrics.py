"""Pure evaluation metrics for ranked retrieval quality.

Exact implementations of recall@k, nDCG@k, MRR (Mean Reciprocal Rank), etc.
These are pure functions with no I/O or dependencies on indices/models.
"""

import math
from collections.abc import Sequence


def _normalize_inputs(
    ranked_ids: Sequence[str],
    relevant_ids: set[str] | Sequence[str],
    k: int | None = None,
) -> tuple[tuple[str, ...], frozenset[str]]:
    if k is not None and k < 0:
        raise ValueError("k must be non-negative")
    return tuple(ranked_ids), frozenset(relevant_ids)


def recall_at_k(ranked_ids: Sequence[str], relevant_ids: set[str] | Sequence[str], k: int) -> float:
    """Compute recall@k: fraction of relevant items in top-k results.

    Args:
        ranked_ids: Ordered list of result IDs (best-ranked first).
        relevant_ids: Set or list of relevant IDs (ground truth).
        k: Cutoff rank.

    Returns:
        Recall@k in [0, 1]. Returns 0 if no relevant items exist.
    """
    ranked, relevant = _normalize_inputs(ranked_ids, relevant_ids, k)
    if not relevant:
        return 0.0

    top_k = set(ranked[:k])
    hits = len(top_k & relevant)
    return hits / len(relevant)


def ndcg_at_k(
    ranked_ids: Sequence[str],
    relevant_ids: set[str] | Sequence[str],
    k: int,
    gains: dict[str, float] | None = None,
) -> float:
    """Compute nDCG@k: normalized discounted cumulative gain.

    Assumes binary relevance by default (relevant=1, non-relevant=0).
    Optionally accepts custom gains for graded relevance.

    Args:
        ranked_ids: Ordered list of result IDs (best-ranked first).
        relevant_ids: Set or list of relevant IDs (ground truth).
        k: Cutoff rank.
        gains: Optional dict mapping result IDs to relevance scores.
               If None, binary relevance (1 for relevant, 0 for non-relevant).

    Returns:
        nDCG@k in [0, 1].
    """
    ranked, relevant = _normalize_inputs(ranked_ids, relevant_ids, k)
    if not relevant:
        return 0.0

    # Compute DCG@k
    dcg = 0.0
    for i, result_id in enumerate(ranked[:k]):
        rank = i + 1  # 1-based ranking
        if result_id in relevant:
            gain = gains.get(result_id, 1.0) if gains else 1.0
            dcg += gain / math.log2(rank + 1)

    # Compute ideal DCG (IDCG) from the labeled relevant universe only.
    ideal_gains = (
        [gains.get(result_id, 1.0) for result_id in relevant]
        if gains
        else [1.0] * len(relevant)
    )
    idcg = sum(
        gain / math.log2(rank + 1)
        for rank, gain in enumerate(sorted(ideal_gains, reverse=True)[:k], start=1)
    )

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def mrr(ranked_ids: Sequence[str], relevant_ids: set[str] | Sequence[str]) -> float:
    """Compute MRR (Mean Reciprocal Rank): 1 / rank of first relevant item.

    Args:
        ranked_ids: Ordered list of result IDs (best-ranked first).
        relevant_ids: Set or list of relevant IDs (ground truth).

    Returns:
        MRR in [0, 1]. Returns 0 if no relevant item is found.
    """
    ranked, relevant = _normalize_inputs(ranked_ids, relevant_ids)
    if not relevant:
        return 0.0

    for i, result_id in enumerate(ranked):
        if result_id in relevant:
            return 1.0 / (i + 1)  # 1-based ranking

    return 0.0


def average_precision(ranked_ids: Sequence[str], relevant_ids: set[str] | Sequence[str]) -> float:
    """Compute Average Precision (AP): mean of precision@k at each relevant hit.

    Args:
        ranked_ids: Ordered list of result IDs (best-ranked first).
        relevant_ids: Set or list of relevant IDs (ground truth).

    Returns:
        AP in [0, 1].
    """
    ranked, relevant = _normalize_inputs(ranked_ids, relevant_ids)
    if not relevant:
        return 0.0

    num_relevant = len(relevant)

    ap = 0.0
    num_hits = 0
    for i, result_id in enumerate(ranked):
        if result_id in relevant:
            num_hits += 1
            precision_at_i = num_hits / (i + 1)
            ap += precision_at_i

    return ap / num_relevant
