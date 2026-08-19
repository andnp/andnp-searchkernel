"""Contract tests for the deterministic representative retrieval corpus."""

import json
from pathlib import Path

from benchmarks.evaluate_labeled_retrieval import load_labeled_fixture

FIXTURE = Path(__file__).parents[1] / "fixtures" / "representative_retrieval_corpus.json"
RECORD_KEYS = {
    "workspace_id",
    "source_kind",
    "source_id",
    "title",
    "body",
    "created_at",
    "updated_at",
    "metadata",
    "uri",
    "status",
    "embedding",
    "embedding_model",
}


def test_representative_fixture_has_backend_neutral_schema() -> None:
    """Records and labels expose the fields needed by every retrieval backend."""
    data = json.loads(FIXTURE.read_text())
    records = data["records"]
    entries = data["entries"]

    assert 30 <= len(records) <= 50
    assert 20 <= len(entries) <= 30
    assert {frozenset(record) for record in records} == {frozenset(RECORD_KEYS)}
    identities = {
        (record["workspace_id"], record["source_kind"], record["source_id"])
        for record in records
    }
    assert len(identities) == len(records)
    assert len({record["workspace_id"] for record in records}) >= 3
    assert len({record["source_kind"] for record in records}) >= 4
    assert sum(record["source_id"] == "shared-brief" for record in records) == 3
    assert all(record["embedding_model"] == "fixture-v1" for record in records)
    assert all(len(record["embedding"]) == 4 for record in records)

    record_ids = {record["source_id"] for record in records}
    assert all(entry["query"] for entry in entries)
    assert len({entry["query"] for entry in entries}) == len(entries)
    assert all(set(entry) <= {"query", "relevant_ids", "relevance", "query_type", "source_kinds", "workspace_id", "tags", "corpus_version", "split"} for entry in entries)
    assert all(set(entry.get("relevance", {})) <= record_ids for entry in entries)
    assert any("relevance" in entry for entry in entries)
    assert all(entry["split"] == "test" for entry in entries)


def test_representative_fixture_loads_deterministically() -> None:
    """Repeated loads preserve record identity, vectors, and graded labels."""
    records_a, golden_a = load_labeled_fixture(FIXTURE)
    records_b, golden_b = load_labeled_fixture(FIXTURE)

    assert [record.to_dict() for record in records_a] == [
        record.to_dict() for record in records_b
    ]
    assert golden_a.to_dict() == golden_b.to_dict()
    assert [record.embedding for record in records_a] == [
        record.embedding for record in records_b
    ]
