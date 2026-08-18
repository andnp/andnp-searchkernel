from datetime import UTC, datetime

import pytest

from searchkernel.domain import Record, SearchResultProvenance
from searchkernel.ports.search_results import RecordSearchResult
from searchkernel.search.diversity import mmr_post_process


def _record(record_id: str) -> Record:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    return Record(
        source_kind="fake",
        source_id=record_id,
        title=record_id,
        body=f"body for {record_id}",
        created_at=timestamp,
        updated_at=timestamp,
    )


def _result(record_id: str, normalized_score: float) -> RecordSearchResult:
    return RecordSearchResult(
        record=_record(record_id),
        score=normalized_score,
        provenance=SearchResultProvenance(),
        normalized_score=normalized_score,
    )


def _ids(results: list[RecordSearchResult]) -> list[str]:
    return [result.record_id for result in results]


def _embedding_lookup(
    embeddings: dict[str, list[float] | None],
):
    def _embedding_of(result: RecordSearchResult) -> list[float] | None:
        return embeddings[result.record_id]

    return _embedding_of


def test_rejects_lambda_outside_unit_interval() -> None:
    with pytest.raises(ValueError, match="lambda_"):
        mmr_post_process(lambda _result: None, lambda_=1.5)
    with pytest.raises(ValueError, match="lambda_"):
        mmr_post_process(lambda _result: None, lambda_=-0.1)


def test_boundary_lambda_values_are_accepted() -> None:
    mmr_post_process(lambda _result: None, lambda_=0.0)
    mmr_post_process(lambda _result: None, lambda_=1.0)


def test_empty_input_returns_empty() -> None:
    post_process = mmr_post_process(lambda _result: None)
    assert post_process([]) == []


def test_single_result_returned_unchanged() -> None:
    result = _result("a", 0.9)
    post_process = mmr_post_process(_embedding_lookup({"a": [1.0, 0.0]}))
    assert post_process([result]) == [result]


def test_near_duplicate_suppression_interleaves_a_distinct_result() -> None:
    # Three near-identical chunks of one document plus one distinct result.
    # A naive relevance sort would put all three duplicates first; MMR
    # should push the distinct result up ahead of at least one duplicate.
    results = [
        _result("dup-1", 0.99),
        _result("dup-2", 0.98),
        _result("dup-3", 0.97),
        _result("distinct", 0.80),
    ]
    embeddings = {
        "dup-1": [1.0, 0.0, 0.0],
        "dup-2": [0.99, 0.01, 0.0],
        "dup-3": [0.98, 0.02, 0.0],
        "distinct": [0.0, 1.0, 0.0],
    }
    post_process = mmr_post_process(_embedding_lookup(embeddings), lambda_=0.5)
    ordered = _ids(post_process(results))

    assert ordered[0] == "dup-1"
    assert "distinct" in ordered[:3]


def test_lambda_one_reduces_to_pure_relevance_order() -> None:
    results = [
        _result("dup-1", 0.99),
        _result("dup-2", 0.98),
        _result("dup-3", 0.97),
        _result("distinct", 0.80),
    ]
    embeddings = {
        "dup-1": [1.0, 0.0],
        "dup-2": [0.99, 0.01],
        "dup-3": [0.98, 0.02],
        "distinct": [0.0, 1.0],
    }
    post_process = mmr_post_process(_embedding_lookup(embeddings), lambda_=1.0)
    ordered = _ids(post_process(results))
    assert ordered == ["dup-1", "dup-2", "dup-3", "distinct"]


def test_lambda_zero_maximises_diversity_after_first_pick() -> None:
    # First pick is always highest relevance. After that, with lambda_=0.0
    # relevance no longer matters at all, so the least-similar-to-selected
    # item must be picked next even though it is the lowest-relevance item.
    results = [
        _result("best", 0.99),
        _result("near-best-duplicate", 0.95),
        _result("far-but-low-relevance", 0.10),
    ]
    embeddings = {
        "best": [1.0, 0.0],
        "near-best-duplicate": [0.99, 0.01],
        "far-but-low-relevance": [0.0, 1.0],
    }
    post_process = mmr_post_process(_embedding_lookup(embeddings), lambda_=0.0)
    ordered = _ids(post_process(results))
    assert ordered == ["best", "far-but-low-relevance", "near-best-duplicate"]


def test_all_identical_embeddings_preserves_relevance_order() -> None:
    results = [
        _result("a", 0.9),
        _result("b", 0.8),
        _result("c", 0.7),
    ]
    embeddings = {name: [1.0, 0.0] for name in ("a", "b", "c")}
    post_process = mmr_post_process(_embedding_lookup(embeddings), lambda_=0.5)
    assert _ids(post_process(results)) == ["a", "b", "c"]


def test_missing_embeddings_are_preserved_and_appended_in_original_order() -> None:
    results = [
        _result("embeddable-1", 0.9),
        _result("no-embedding-1", 0.85),
        _result("embeddable-2", 0.5),
        _result("no-embedding-2", 0.4),
    ]
    embeddings = {
        "embeddable-1": [1.0, 0.0],
        "no-embedding-1": None,
        "embeddable-2": [0.0, 1.0],
        "no-embedding-2": None,
    }
    post_process = mmr_post_process(_embedding_lookup(embeddings), lambda_=0.7)
    ordered = _ids(post_process(results))

    assert set(ordered) == {
        "embeddable-1",
        "no-embedding-1",
        "embeddable-2",
        "no-embedding-2",
    }
    # Both un-embeddable results trail the diversified (embeddable) prefix,
    # in their original relative order.
    assert ordered[-2:] == ["no-embedding-1", "no-embedding-2"]


def test_all_missing_embeddings_returns_input_unchanged() -> None:
    results = [_result("a", 0.9), _result("b", 0.5)]
    post_process = mmr_post_process(lambda _result: None)
    assert post_process(results) == results


def test_ties_are_broken_by_original_input_position_deterministically() -> None:
    results = [
        _result("first", 0.5),
        _result("second", 0.5),
        _result("third", 0.5),
    ]
    embeddings = {name: [1.0, 0.0] for name in ("first", "second", "third")}
    post_process = mmr_post_process(_embedding_lookup(embeddings), lambda_=0.5)

    for _ in range(5):
        assert _ids(post_process(results)) == ["first", "second", "third"]
