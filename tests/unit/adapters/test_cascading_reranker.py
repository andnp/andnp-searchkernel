import pytest

from searchkernel.adapters.rerank.cascading import CascadingReranker
from searchkernel.ports.rerank import Reranker


class _FakeReranker:
    def __init__(self, model_name: str, scores: list[float]) -> None:
        self.model_name = model_name
        self._scores = scores
        self.calls: list[list[str]] = []

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        self.calls.append(list(documents))
        return list(self._scores)


class _RaisingReranker:
    model_name = "raising"

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        raise RuntimeError("slow model is down")


def _cascade(
    fast_scores: list[float],
    slow_scores: list[float],
    *,
    escalate_top_n: int = 2,
    confidence_gap: float = 0.1,
) -> tuple[CascadingReranker, _FakeReranker, _FakeReranker]:
    fast = _FakeReranker("fast-model", fast_scores)
    slow = _FakeReranker("slow-model", slow_scores)
    cascade = CascadingReranker(
        fast,
        slow,
        escalate_top_n=escalate_top_n,
        confidence_gap=confidence_gap,
    )
    return cascade, fast, slow


def test_implements_reranker_protocol() -> None:
    cascade, _, _ = _cascade([0.9, 0.1], [0.5, 0.5])
    assert isinstance(cascade, Reranker)


def test_model_name_identifies_composition() -> None:
    cascade, _, _ = _cascade([0.9, 0.1], [0.5, 0.5])
    assert cascade.model_name == "cascade(fast-model,slow-model)"


def test_no_escalation_when_fast_model_separates_cleanly() -> None:
    # Gap between rank 1 (0.9) and rank 2 (0.2) is 0.7, well above 0.1.
    docs = ["a", "b", "c"]
    cascade, _, slow = _cascade(
        [0.9, 0.85, 0.2], [1.0, 1.0, 1.0], escalate_top_n=2
    )

    scores = cascade.rerank("q", docs)

    assert scores == [0.9, 0.85, 0.2]
    assert slow.calls == []


def test_escalation_reorders_top_group_by_slow_scores() -> None:
    # Fast model puts b > a > c in the top 2, but is unsure of the cut
    # (gap between rank 2 and rank 3 is only 0.05 < confidence_gap 0.1).
    docs = ["a", "b", "c", "d"]
    fast_scores = [0.80, 0.85, 0.76, 0.10]
    # slow rescoring of the top group [b, a] (fast order) reverses them.
    cascade, _, slow = _cascade(
        fast_scores, [0.2, 0.9], escalate_top_n=2, confidence_gap=0.1
    )

    scores = cascade.rerank("q", docs)

    assert slow.calls == [["b", "a"]]
    # 'a' (slow score 0.9) now ranks above 'b' (slow score 0.2).
    assert scores[docs.index("a")] > scores[docs.index("b")]
    # Escalated docs stay above the untouched remainder.
    remainder_max = max(scores[docs.index("c")], scores[docs.index("d")])
    assert scores[docs.index("a")] > remainder_max
    assert scores[docs.index("b")] > remainder_max
    # Remainder scores are untouched fast scores.
    assert scores[docs.index("c")] == fast_scores[2]
    assert scores[docs.index("d")] == fast_scores[3]


def test_returned_scores_are_aligned_to_input_order() -> None:
    docs = ["x", "y", "z", "w"]
    fast_scores = [0.5, 0.95, 0.94, 0.1]
    cascade, _, slow = _cascade(
        fast_scores, [0.3, 0.7], escalate_top_n=2, confidence_gap=0.5
    )

    scores = cascade.rerank("q", docs)

    assert len(scores) == len(docs)
    # slow was called with the top group in fast-rank order: y, z.
    assert slow.calls == [["y", "z"]]
    # 'z' had the higher slow score (0.7) so it now outranks 'y' (0.3).
    assert scores[docs.index("z")] > scores[docs.index("y")]
    assert scores[docs.index("w")] == fast_scores[3]


def test_all_scores_within_zero_one_bounds() -> None:
    docs = ["a", "b", "c", "d"]
    cascade, _, _ = _cascade(
        [0.5, 0.51, 0.505, 0.1], [0.99, 0.01], escalate_top_n=2, confidence_gap=0.5
    )

    scores = cascade.rerank("q", docs)

    assert all(0.0 <= score <= 1.0 for score in scores)


def test_determinism_on_tied_fast_scores_breaks_by_input_position() -> None:
    docs = ["a", "b", "c", "d"]
    fast_scores = [0.5, 0.5, 0.5, 0.5]
    cascade, _, slow = _cascade(
        fast_scores, [0.1, 0.2], escalate_top_n=2, confidence_gap=1.0
    )

    scores = cascade.rerank("q", docs)
    scores_again = cascade.rerank("q", docs)

    assert scores == scores_again
    # ties broken by input position ascending: top group is [a, b].
    assert slow.calls[0] == ["a", "b"]


def test_determinism_on_tied_slow_scores() -> None:
    docs = ["a", "b", "c", "d"]
    fast_scores = [0.80, 0.79, 0.2, 0.1]
    cascade, _, _ = _cascade(
        fast_scores, [0.5, 0.5], escalate_top_n=2, confidence_gap=1.0
    )

    scores = cascade.rerank("q", docs)
    scores_again = cascade.rerank("q", docs)

    assert scores == scores_again


def test_empty_documents_returns_empty_scores() -> None:
    cascade, fast, slow = _cascade([], [])

    assert cascade.rerank("q", []) == []
    assert fast.calls == []
    assert slow.calls == []


def test_small_set_separated_cleanly_skips_escalation() -> None:
    # n < escalate_top_n, but the only adjacent gap (0.8) clears 0.1.
    docs = ["a", "b"]
    cascade, _, slow = _cascade(
        [0.9, 0.1], [], escalate_top_n=5, confidence_gap=0.1
    )

    scores = cascade.rerank("q", docs)

    assert scores == [0.9, 0.1]
    assert slow.calls == []


def test_small_set_undifferentiated_fast_model_escalates() -> None:
    # The fast model cannot separate any of these documents at all
    # (every score is 0.5) -- exactly the case the slow model should
    # resolve, even though n (10) is well under escalate_top_n (15).
    docs = [str(i) for i in range(10)]
    fast_scores = [0.5] * 10
    slow_scores = [round(i * 0.1, 1) for i in range(10)]
    cascade, _, slow = _cascade(
        fast_scores, slow_scores, escalate_top_n=15, confidence_gap=0.1
    )

    scores = cascade.rerank("q", docs)

    assert slow.calls == [docs]
    # slow scores increase with index, so the returned scores must too.
    assert all(scores[i] < scores[i + 1] for i in range(9))


def test_small_set_single_close_adjacent_pair_escalates() -> None:
    # Every adjacent gap clears 0.05 except one (0.5 -> 0.48 = 0.02).
    docs = ["a", "b", "c", "d"]
    fast_scores = [0.9, 0.5, 0.48, 0.1]
    cascade, _, slow = _cascade(
        fast_scores, [0.2, 0.9, 0.1, 0.4], escalate_top_n=5, confidence_gap=0.05
    )

    cascade.rerank("q", docs)

    assert slow.calls == [docs]


def test_small_set_single_document_never_escalates() -> None:
    docs = ["a"]
    cascade, _, slow = _cascade([0.5], [], escalate_top_n=5, confidence_gap=1.0)

    scores = cascade.rerank("q", docs)

    assert scores == [0.5]
    assert slow.calls == []


def test_slow_model_failure_propagates() -> None:
    docs = ["a", "b", "c"]
    fast = _FakeReranker("fast-model", [0.5, 0.49, 0.1])
    cascade = CascadingReranker(
        fast, _RaisingReranker(), escalate_top_n=2, confidence_gap=1.0
    )

    with pytest.raises(RuntimeError, match="slow model is down"):
        cascade.rerank("q", docs)


@pytest.mark.parametrize(
    ("escalate_top_n", "confidence_gap"),
    [(0, 0.1), (-1, 0.1)],
)
def test_constructor_rejects_invalid_escalate_top_n(
    escalate_top_n: int, confidence_gap: float
) -> None:
    fast = _FakeReranker("fast-model", [])
    slow = _FakeReranker("slow-model", [])
    with pytest.raises(ValueError, match="escalate_top_n must be positive"):
        CascadingReranker(
            fast,
            slow,
            escalate_top_n=escalate_top_n,
            confidence_gap=confidence_gap,
        )


@pytest.mark.parametrize("confidence_gap", [-0.1, 1.1])
def test_constructor_rejects_invalid_confidence_gap(confidence_gap: float) -> None:
    fast = _FakeReranker("fast-model", [])
    slow = _FakeReranker("slow-model", [])
    with pytest.raises(
        ValueError, match=r"confidence_gap must be between 0.0 and 1.0"
    ):
        CascadingReranker(
            fast,
            slow,
            escalate_top_n=2,
            confidence_gap=confidence_gap,
        )
