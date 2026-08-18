"""Cascading reranker: composes two Reranker instances into one.

Reranking today is one model over a fixed budget. A cascade runs a cheap
("fast") model over the whole candidate set and escalates to an expensive
("slow") model only for the top few, and only when the fast model's ordering
is genuinely in doubt. This module implements the existing `Reranker` port
by composition, and also implements `RecordReranker` -- when a tier supports
identity-aware scoring, the cascade passes records through to it instead of
flattening them to text.
"""

from __future__ import annotations

from collections.abc import Callable

from searchkernel.domain import Record
from searchkernel.ports.rerank import RecordReranker, Reranker


def _tier_scores(tier: Reranker, query: str, records: list[Record]) -> list[float]:
    if isinstance(tier, RecordReranker):
        return tier.rerank_records(query, records)
    documents = [f"{record.title}\n{record.indexed_text or record.body}".strip() for record in records]
    return tier.rerank(query, documents)


class CascadingReranker:
    """Escalates from a fast reranker to a slow one only when needed.

    Escalation rule
    ----------------
    The fast model scores every document. Documents are then ranked by
    fast score (ties broken by input position, ascending, for determinism).
    The question is always the same -- "is the ordering in doubt?" -- but it
    is answered differently depending on whether there is a cut to reason
    about:

    - More documents than `escalate_top_n` (the common "large candidate
      set" case): the candidate cut is the boundary between rank
      `escalate_top_n - 1` (the last document that would make the cut) and
      rank `escalate_top_n` (the first document that would not). The gap
      between those two scores measures how confidently the fast model has
      separated the "top group" from the rest. If that gap is smaller than
      `confidence_gap`, the cut is in doubt, so the top `escalate_top_n`
      documents are rescored by the slow model to get a better-calibrated
      ordering (and, implicitly, a better-calibrated cut). If the gap is
      large enough, the fast model's cut is trusted and no escalation
      happens.

    - At most `escalate_top_n` documents (the common case in practice,
      since reranking is normally applied to an already-narrowed final
      set): there is no cut, so the cut-boundary gap cannot be measured.
      Instead the leading gap decides: the distance between the best and
      second-best fast scores is compared against `confidence_gap`. A model
      that cannot separate its own top two has left the ordering in doubt
      where it matters, and every document is rescored. Reading the
      smallest gap anywhere in the list instead would escalate on ordinary
      tail clustering -- 0.09 against 0.08 among results nobody reads --
      which is every real score distribution, and the cheap tier would buy
      nothing. This still does NOT reuse the "fewer documents than the
      escalation budget means nothing to escalate" shortcut: a set the fast
      model cannot separate at all has a completely unresolved ordering,
      which is exactly the case the slow model is most useful for.

    Combining scores
    ----------------
    When escalation happens, the escalated documents are re-ranked among
    themselves by the slow model's scores (ties broken by input position),
    then linearly rescaled into a slice of [0, 1] that sits strictly above
    the highest fast score among the non-escalated remainder (or, when
    every document escalates, into [0, 1] directly since there is no
    remainder to stay above). This keeps the escalated group above the
    remainder while preserving the slow model's relative ordering, and
    keeps every returned score within the port's documented [0, 1]
    contract. Note that when all slow scores in the escalated group are
    equal, every one of those documents is assigned the same final score;
    the input-position tie-break used to decide rescaling order does not
    surface as a distinct output value in that case, so a caller that needs
    a strict order among exactly-tied slow scores must re-sort (stably)
    itself.

    Failure handling
    -----------------
    If the slow model raises during escalation, the exception propagates
    unchanged; it is not swallowed to silently degrade to the fast scores.
    Escalation only happens when the fast model's ordering was already in
    doubt, so a silent fast-score fallback would return a ranking the
    cascade itself just flagged as untrustworthy, with no signal to the
    caller that the intended (slow) reranking never happened. Callers that
    want resilience against slow-model failures can wrap `slow` themselves
    (e.g. with their own fallback or retry policy) with full knowledge of
    what "falling back" means for their use case.
    """

    def __init__(
        self,
        fast: Reranker,
        slow: Reranker,
        *,
        escalate_top_n: int = 15,
        confidence_gap: float = 0.1,
    ) -> None:
        if escalate_top_n < 1:
            raise ValueError("escalate_top_n must be positive")
        if not 0.0 <= confidence_gap <= 1.0:
            raise ValueError("confidence_gap must be between 0.0 and 1.0")

        self._fast = fast
        self._slow = slow
        self._escalate_top_n = escalate_top_n
        self._confidence_gap = confidence_gap
        self.model_name = f"cascade({fast.model_name},{slow.model_name})"

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        fast_scores = self._fast.rerank(query, documents)
        return self._cascade(
            query,
            n=len(documents),
            fast_scores=fast_scores,
            score_slow=lambda indices: self._slow.rerank(
                query, [documents[i] for i in indices]
            ),
        )

    def rerank_records(self, query: str, records: list[Record]) -> list[float]:
        if not records:
            return []
        fast_scores = _tier_scores(self._fast, query, records)
        return self._cascade(
            query,
            n=len(records),
            fast_scores=fast_scores,
            score_slow=lambda indices: _tier_scores(
                self._slow, query, [records[i] for i in indices]
            ),
        )

    def _cascade(
        self,
        query: str,
        *,
        n: int,
        fast_scores: list[float],
        score_slow: Callable[[list[int]], list[float]],
    ) -> list[float]:
        ranked_indices = sorted(range(n), key=lambda i: (-fast_scores[i], i))

        if n <= self._escalate_top_n:
            escalate, top_indices, remainder_indices = self._decide_small_set(
                fast_scores, ranked_indices
            )
        else:
            escalate, top_indices, remainder_indices = self._decide_large_set(
                fast_scores, ranked_indices
            )
        if not escalate:
            return fast_scores

        slow_scores = score_slow(top_indices)

        remainder_max = max(
            (fast_scores[i] for i in remainder_indices), default=0.0
        )
        return self._combine(
            fast_scores, top_indices, slow_scores, remainder_max
        )

    def _decide_large_set(
        self, fast_scores: list[float], ranked_indices: list[int]
    ) -> tuple[bool, list[int], list[int]]:
        top_indices = ranked_indices[: self._escalate_top_n]
        remainder_indices = ranked_indices[self._escalate_top_n :]

        boundary_score = fast_scores[top_indices[-1]]
        next_score = fast_scores[remainder_indices[0]]
        if boundary_score - next_score >= self._confidence_gap:
            return False, [], []
        return True, top_indices, remainder_indices

    def _decide_small_set(
        self, fast_scores: list[float], ranked_indices: list[int]
    ) -> tuple[bool, list[int], list[int]]:
        n = len(ranked_indices)
        if n <= 1:
            return False, [], []

        # Only the leading gap decides. Reading the smallest gap anywhere in
        # the list means ordinary tail clustering — 0.09 against 0.08 among
        # results nobody reads — escalates a set whose top the fast model
        # separated cleanly, which is every real score distribution.
        leading_gap = (
            fast_scores[ranked_indices[0]] - fast_scores[ranked_indices[1]]
        )
        if leading_gap >= self._confidence_gap:
            return False, [], []
        return True, ranked_indices, []

    def _combine(
        self,
        fast_scores: list[float],
        top_indices: list[int],
        slow_scores: list[float],
        remainder_max: float,
    ) -> list[float]:
        pairs = list(zip(top_indices, slow_scores, strict=True))
        ranked = sorted(pairs, key=lambda pair: (-pair[1], pair[0]))

        slow_min = min(score for _, score in ranked)
        slow_max = max(score for _, score in ranked)
        span = slow_max - slow_min

        headroom = 1.0 - remainder_max
        result = list(fast_scores)

        if headroom <= 0.0:
            for index, _ in ranked:
                result[index] = 1.0
            return result

        epsilon = headroom * 1e-6
        usable_headroom = headroom - epsilon
        for index, score in ranked:
            normalized = (score - slow_min) / span if span > 1e-9 else 1.0
            final = remainder_max + epsilon + usable_headroom * normalized
            result[index] = min(final, 1.0)

        return result
