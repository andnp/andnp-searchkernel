"""Tests for the checked-in labeled retrieval evaluation."""

from pathlib import Path

from benchmarks.evaluate_labeled_retrieval import evaluate_fixture, load_labeled_fixture

FIXTURE = Path(__file__).parents[1] / "fixtures" / "labeled_retrieval_corpus.json"


def test_labeled_fixture_contains_query_slices_and_records() -> None:
    records, golden_set = load_labeled_fixture(FIXTURE)

    assert len(records) == 5
    assert len(golden_set.entries) == 5
    assert {entry.query_type for entry in golden_set} == {
        "exact",
        "conceptual",
        "vague",
        "multi_term",
        "unrelated",
    }
    assert golden_set.entries[1].relevance == {"incident-42": 3.0, "deploy-17": 2.0}


def test_labeled_fixture_evaluates_expected_local_retrieval() -> None:
    report = evaluate_fixture(FIXTURE)

    assert report.golden_set_size == 5
    assert report.warmup_count == 2
    assert report.measured_repetitions == 5
    assert len(report.metrics) == 25
    assert report.mean_recall_at_k == 0.8
    assert report.slices["query_type:exact"].mean_recall_at_k == 1.0
    assert report.slices["tag:unrelated"].empty_result_rate == 1.0
