from datetime import UTC, datetime

import pytest

from searchkernel.domain import Record, RecordIdentity
from searchkernel.domain.vector_filters import (
    compile_source_scoped_filters,
    compile_vector_filters,
)
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


def test_compiled_vector_filter_reuses_normalized_constraints() -> None:
    record = _record(
        "guide/setup",
        project_id="keep",
        file_path="/docs/guide/setup.md",
    )
    predicate = compile_vector_filters(
        {
            "candidate_ids": [record.identity],
            "workspace_id": "workspace",
            "source_kind": "note",
            "project_id": "keep",
            "document_id": "guide/setup",
            "metadata_equals": {"project_id": "keep"},
        }
    )

    assert predicate.matches(
        storage_key=record.storage_key,
        source_id=record.source_id,
        workspace_id=record.workspace_id,
        source_kind=record.source_kind,
        status=record.status,
        metadata=record.metadata,
        uri=record.uri,
    )
    assert not predicate.matches(
        storage_key=record.storage_key,
        source_id=record.source_id,
        workspace_id="other",
        source_kind=record.source_kind,
        status=record.status,
        metadata=record.metadata,
        uri=record.uri,
    )


def test_compiled_vector_filter_metadata_in_matches_any_allowed_value() -> None:
    record = _record(
        "guide/setup",
        project_id="keep",
        file_path="/docs/guide/setup.md",
    )
    predicate = compile_vector_filters(
        {"metadata_in": {"project_id": ["keep", "other"]}}
    )

    assert predicate.matches(
        storage_key=record.storage_key,
        source_id=record.source_id,
        workspace_id=record.workspace_id,
        source_kind=record.source_kind,
        status=record.status,
        metadata=record.metadata,
        uri=record.uri,
    )
    assert not predicate.matches(
        storage_key=record.storage_key,
        source_id=record.source_id,
        workspace_id=record.workspace_id,
        source_kind=record.source_kind,
        status=record.status,
        metadata={**record.metadata, "project_id": "dropped"},
        uri=record.uri,
    )


def test_local_vector_filters_support_metadata_in(tmp_path) -> None:
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

    hits = backend.search_vector(
        [1.0, 0.0],
        5,
        model_name="model",
        dim=2,
        filters={"metadata_in": {"project_id": ["keep", "other"]}},
    )

    assert [hit.storage_key for hit in hits] == [included.storage_key]


def test_source_scoped_filter_requires_array_overlap_and_passes_unscoped_sources() -> None:
    """Scoped sources require an authorized metadata-array overlap only."""
    predicate = compile_vector_filters(
        {
            "source_scoped_filters": {
                "note": {"metadata_contains_any": {"teams": ["search", "ops"]}}
            }
        }
    )

    assert predicate.matches(
        storage_key="record:workspace:note:allowed",
        source_id="allowed",
        workspace_id="workspace",
        source_kind="note",
        status="active",
        metadata={"teams": ["other", "search"]},
    )
    assert not predicate.matches(
        storage_key="record:workspace:note:blocked",
        source_id="blocked",
        workspace_id="workspace",
        source_kind="note",
        status="active",
        metadata={"teams": ["other"]},
    )
    assert predicate.matches(
        storage_key="record:workspace:commit:unscoped",
        source_id="unscoped",
        workspace_id="workspace",
        source_kind="commit",
        status="active",
        metadata={},
    )


def test_source_scoped_filter_empty_allowed_values_fail_closed() -> None:
    """An empty authorization claim rejects only its scoped source kind."""
    predicate = compile_vector_filters(
        {
            "source_scoped_filters": {
                "note": {"metadata_contains_any": {"teams": []}}
            }
        }
    )

    assert not predicate.matches(
        storage_key="record:workspace:note:blocked",
        source_id="blocked",
        workspace_id="workspace",
        source_kind="note",
        status="active",
        metadata={"teams": ["search"]},
    )
    assert predicate.matches(
        storage_key="record:workspace:commit:unscoped",
        source_id="unscoped",
        workspace_id="workspace",
        source_kind="commit",
        status="active",
        metadata={},
    )


def test_source_scoped_filter_requires_identity_and_non_empty_membership() -> None:
    """Drive authorization combines workspace identity with scope membership."""
    predicate = compile_vector_filters(
        {
            "source_scoped_filters": {
                "gdrive": {
                    "workspace_ids": ["workspace-a"],
                    "metadata_non_empty": ["scope_memberships"],
                }
            }
        }
    )

    assert predicate.matches(
        storage_key="record:workspace-a:gdrive:allowed",
        source_id="allowed",
        workspace_id="workspace-a",
        source_kind="gdrive",
        status="active",
        metadata={"scope_memberships": ["shared-drive:drive-a"]},
    )
    assert not predicate.matches(
        storage_key="record:workspace-b:gdrive:wrong-workspace",
        source_id="wrong-workspace",
        workspace_id="workspace-b",
        source_kind="gdrive",
        status="active",
        metadata={"scope_memberships": ["shared-drive:drive-a"]},
    )
    assert not predicate.matches(
        storage_key="record:workspace-a:gdrive:missing-scope",
        source_id="missing-scope",
        workspace_id="workspace-a",
        source_kind="gdrive",
        status="active",
        metadata={"scope_memberships": []},
    )
    assert predicate.matches(
        storage_key="record:workspace-b:commit:unscoped",
        source_id="unscoped",
        workspace_id="workspace-b",
        source_kind="commit",
        status="active",
        metadata={},
    )


def test_source_scoped_filter_empty_workspace_ids_fail_closed() -> None:
    """An empty workspace claim denies only the scoped source kind."""
    predicate = compile_vector_filters(
        {
            "source_scoped_filters": {
                "gdrive": {
                    "workspace_ids": [],
                    "metadata_non_empty": ["scope_memberships"],
                }
            }
        }
    )

    assert not predicate.matches(
        storage_key="record:workspace-a:gdrive:denied",
        source_id="denied",
        workspace_id="workspace-a",
        source_kind="gdrive",
        status="active",
        metadata={"scope_memberships": ["shared-drive:drive-a"]},
    )
    assert predicate.matches(
        storage_key="record:workspace-a:commit:unscoped",
        source_id="unscoped",
        workspace_id="workspace-a",
        source_kind="commit",
        status="active",
        metadata={},
    )


@pytest.mark.parametrize(
    "filters",
    [
        {"source_scoped_filters": []},
        {
            "source_scoped_filters": {
                "gdrive": {"workspace_ids": "workspace-a"}
            }
        },
        {
            "source_scoped_filters": {
                "note": {"metadata_contains_any": {"teams": "search"}}
            }
        },
        {
            "source_scoped_filters": {
                "note": {"metadata_contains_any": {"teams": [1]}}
            }
        },
    ],
)
def test_source_scoped_filter_rejects_malformed_values(
    filters: dict[str, object],
) -> None:
    """Malformed authorization data raises a deterministic validation error."""
    with pytest.raises((TypeError, ValueError)):
        compile_source_scoped_filters(filters)
