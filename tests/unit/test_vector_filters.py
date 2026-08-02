from datetime import UTC, datetime

from searchkernel.domain import Record, RecordIdentity
from searchkernel.indices import LocalRecordBackend


def _record(
    source_id: str,
    *,
    project_id: str,
    file_path: str,
    workspace_id: str = "workspace",
    source_kind: str = "note",
) -> Record:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    return Record(
        workspace_id=workspace_id,
        source_kind=source_kind,
        source_id=source_id,
        title=source_id,
        body="body",
        created_at=timestamp,
        updated_at=timestamp,
        metadata={"doc_id": source_id, "project_id": project_id},
        uri=file_path,
    )


def test_local_vector_filters_match_canonical_identity_and_metadata(tmp_path) -> None:
    backend = LocalRecordBackend(tmp_path / "records.db")
    included = _record(
        "guide/setup",
        project_id="keep",
        file_path="/docs/guide/setup.md",
    )
    excluded = _record(
        "guide/other",
        project_id="drop",
        file_path="/docs/guide/other.md",
    )
    for record in (included, excluded):
        record.embedding = [1.0, 0.0]
    backend.upsert([included, excluded], "model", 2)

    filters = {
        "candidate_ids": [RecordIdentity.from_storage_key(included.storage_key)],
        "project_id": "keep",
        "excluded_files": {"other"},
        "workspace_id": "workspace",
    }
    hits = backend.search_vector(
        [1.0, 0.0],
        5,
        model_name="model",
        dim=2,
        filters=filters,
    )

    assert [hit.storage_key for hit in hits] == [included.storage_key]


def test_local_vector_candidate_filter_does_not_match_colliding_source_ids(tmp_path) -> None:
    backend = LocalRecordBackend(tmp_path / "records.db")
    records = [
        _record("same", project_id="one", file_path="/one/same.md", workspace_id="one"),
        _record("same", project_id="two", file_path="/two/same.md", workspace_id="two"),
        _record(
            "same",
            project_id="commit",
            file_path="/one/same.commit",
            workspace_id="one",
            source_kind="commit",
        ),
    ]
    for record in records:
        record.embedding = [1.0, 0.0]
    backend.upsert(records, "model", 2)

    hits = backend.search_vector(
        [1.0, 0.0],
        5,
        model_name="model",
        dim=2,
        filters={"candidate_ids": [records[0].identity]},
    )

    assert [hit.storage_key for hit in hits] == [records[0].storage_key]


def test_local_vector_candidate_filter_rejects_bare_source_id(tmp_path) -> None:
    backend = LocalRecordBackend(tmp_path / "records.db")
    record = _record("same", project_id="one", file_path="/one/same.md")
    record.embedding = [1.0, 0.0]
    backend.upsert([record], "model", 2)

    assert backend.search_vector(
        [1.0, 0.0],
        5,
        model_name="model",
        dim=2,
        filters={"candidate_ids": ["same"]},
    ) == []
