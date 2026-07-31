from datetime import UTC, datetime

from searchkernel.domain import Record, RecordIdentity
from searchkernel.indices import LocalRecordBackend


def _record(
    source_id: str,
    *,
    project_id: str,
    file_path: str,
    workspace_id: str = "workspace",
) -> Record:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    return Record(
        workspace_id=workspace_id,
        source_kind="note",
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
