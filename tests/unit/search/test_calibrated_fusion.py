from datetime import UTC, datetime

import pytest

from searchkernel.domain import Record
from searchkernel.eval.metrics import mrr, ndcg_at_k, recall_at_k
from searchkernel.search.fusion import fuse_calibrated_scores, fuse_reciprocal_rank


def _record(record_id: str, title: str) -> Record:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    return Record(
        source_kind="synthetic",
        source_id=record_id,
        title=title,
        body=f"Technical details for {title}.",
        created_at=timestamp,
        updated_at=timestamp,
    )


def _ranked_ids(scores: dict[str, float]) -> list[str]:
    return [
        record_id
        for record_id, _score in sorted(
            scores.items(), key=lambda item: (-item[1], item[0])
        )
    ]


@pytest.mark.parametrize(
    ("records", "rankings", "relevant"),
    [
        (
            [_record("identifier", "Exact identifier")],
            {
                "keyword": [("identifier", 100.0), ("distractor", 80.0)],
                "vector": [("identifier", 0.9), ("distractor", 0.8)],
            },
            {"identifier"},
        ),
        (
            [_record("full-title", "A full descriptive title")],
            {
                "keyword": [("full-title", 10.0), ("common-term", 9.0)],
                "vector": [("full-title", 0.95), ("common-term", 0.1)],
            },
            {"full-title"},
        ),
        (
            [_record("technical-term", "Technical retrieval term")],
            {
                "keyword": [("technical-term", 7.0), ("common-term", 6.0)],
                "vector": [("technical-term", 0.9), ("common-term", 0.2)],
            },
            {"technical-term"},
        ),
    ],
)
def test_calibrated_fusion_prioritizes_precise_synthetic_matches(
    records: list[Record],
    rankings: dict[str, list[tuple[str, float]]],
    relevant: set[str],
) -> None:
    scores = fuse_calibrated_scores(rankings)
    ranked = _ranked_ids(scores)

    assert records[0].source_id in relevant
    assert ranked[0] == records[0].source_id
    assert recall_at_k(ranked, relevant, 1) == 1.0
    assert mrr(ranked, relevant) == 1.0
    assert ndcg_at_k(ranked, relevant, 1) == 1.0


def test_calibrated_fusion_retains_vector_only_recall() -> None:
    records = [_record("vector-only", "Semantic match")]
    rankings = {
        "keyword": [("lexical", 0.9)],
        "vector": [("vector-only", 0.95), ("lexical", 0.7)],
    }

    ranked = _ranked_ids(fuse_calibrated_scores(rankings))

    assert recall_at_k(ranked, {"vector-only"}, 2) == 1.0
    assert ndcg_at_k(ranked, {"vector-only"}, 2) > 0.0
    assert records[0].source_id in ranked


def test_calibrated_fusion_does_not_let_a_single_hit_lane_dominate() -> None:
    """
    A lane returning one weak hit has no basis for ranking it above items
    from a lane with a rich, well-separated result set.
    """
    scores = fuse_calibrated_scores(
        {
            "fallback": [("weak-hit", 0.01)],
            "keyword": [("strong-match", 100.0), ("distractor", 1.0)],
        }
    )

    assert scores["weak-hit"] == 0.5
    assert scores["strong-match"] > scores["weak-hit"]


def test_calibrated_fusion_preserves_lane_scale_and_weights() -> None:
    scores = fuse_calibrated_scores(
        {
            "keyword": [("keyword-best", 10.0), ("keyword-worst", 1.0)],
            "vector": [("vector-best", 0.9), ("vector-worst", 0.1)],
        },
        strategy_weights={"keyword": 2.0, "vector": 3.0},
    )

    assert scores == {
        "keyword-best": 2.0,
        "keyword-worst": 0.0,
        "vector-best": 3.0,
        "vector-worst": 0.0,
    }


def test_calibrated_fusion_rejects_non_finite_lane_scores() -> None:
    with pytest.raises(ValueError, match="lane scores must be finite"):
        fuse_calibrated_scores({"keyword": [("record", float("nan"))]})


def test_calibrated_fusion_is_opt_in_from_default_rrf() -> None:
    rankings = {"keyword": ["a", "b"], "vector": ["b", "a"]}

    expected = 1 / 61 + 1 / 62
    assert fuse_reciprocal_rank(rankings) == {"a": expected, "b": expected}
