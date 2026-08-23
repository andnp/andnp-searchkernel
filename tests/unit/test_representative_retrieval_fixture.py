"""Contract tests for the deterministic representative retrieval corpus."""

import json
from pathlib import Path

from benchmarks.evaluate_labeled_retrieval import load_labeled_fixture
from searchkernel.domain import Record, RecordIdentity
from searchkernel.eval.golden import GoldenEntry

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


def _matching_identities(
    records: list[Record], entry: GoldenEntry, source_id: str
) -> tuple[RecordIdentity, ...]:
    """Resolve one bare label using the entry's declared identity context."""
    return tuple(
        record.identity
        for record in records
        if record.source_id == source_id
        and (entry.workspace_id is None or record.workspace_id == entry.workspace_id)
        and (not entry.source_kinds or record.source_kind in entry.source_kinds)
    )


def test_representative_labels_resolve_to_canonical_storage_keys() -> None:
    """Resolve positive labels through canonical identity context.

    Every positive label must identify exactly one canonical storage key.
    """
    records, golden = load_labeled_fixture(FIXTURE)

    for entry in golden:
        for source_id in entry.relevant_ids:
            matches = _matching_identities(records, entry, source_id)
            assert len(matches) == 1, (
                f"{entry.query!r} label {source_id!r} resolved to {matches}"
            )
            identity = matches[0]
            assert RecordIdentity.from_storage_key(identity.storage_key) == identity


def test_representative_fixture_isolates_duplicate_bare_source_ids() -> None:
    """Keep duplicate source IDs distinct across workspace identities.

    Workspace identity must remain part of every expected storage key.
    """
    records, golden = load_labeled_fixture(FIXTURE)
    shared = [record.identity for record in records if record.source_id == "shared-brief"]

    assert len(shared) == 3
    assert {identity.storage_key for identity in shared} == {
        RecordIdentity(workspace, "issue", "shared-brief").storage_key
        for workspace in ("engineering", "product", "personal")
    }

    for entry in golden:
        if "duplicate-source-id" not in entry.tags:
            continue
        matches = _matching_identities(records, entry, "shared-brief")
        assert len(matches) == 1
        assert matches[0].storage_key == RecordIdentity(
            entry.workspace_id, "issue", "shared-brief"
        ).storage_key


def test_representative_fixture_covers_meaningful_query_label_slices() -> None:
    """Require meaningful ranking and filtering slices in the fixture.

    The corpus must include positive, graded, identity, filter, and empty cases.
    """
    _records, golden = load_labeled_fixture(FIXTURE)
    entries = list(golden)
    query_types = {entry.query_type for entry in entries}
    tags = {tag for entry in entries for tag in entry.tags}

    assert {"conceptual", "exact", "multi_term", "vague", "unrelated"} <= query_types
    assert {"graded", "exact", "identity", "source-kind-filter", "empty"} <= tags
    assert len({entry.workspace_id for entry in entries}) >= 3
    assert sum(entry.relevance is not None for entry in entries) >= 4
    assert sum(not entry.relevant_ids for entry in entries) >= 3
    assert all(entry.source_kinds for entry in entries if entry.relevant_ids)
