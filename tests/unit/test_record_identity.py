from datetime import UTC, datetime

import pytest

from searchkernel.domain import Record, canonical_storage_key


def test_storage_key_includes_optional_workspace_and_source_kind() -> None:
    assert canonical_storage_key(None, "note", "same") != canonical_storage_key(
        None, "commit", "same"
    )
    assert canonical_storage_key("workspace-a", "note", "same") != (
        canonical_storage_key("workspace-b", "note", "same")
    )


def test_record_serialization_preserves_workspace_identity() -> None:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    record = Record(
        workspace_id="workspace-a",
        source_kind="note",
        source_id="same",
        title="Title",
        body="Body",
        created_at=timestamp,
        updated_at=timestamp,
    )

    restored = Record.from_dict(record.to_dict())

    assert restored.workspace_id == "workspace-a"
    assert restored.storage_key == record.storage_key


def test_record_rejects_empty_identity_parts() -> None:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    with pytest.raises(ValueError):
        record = Record(
            source_kind="",
            source_id="id",
            title="Title",
            body="Body",
            created_at=timestamp,
            updated_at=timestamp,
        )
        _ = record.storage_key
