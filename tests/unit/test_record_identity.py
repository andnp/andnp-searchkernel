import json
from datetime import UTC, datetime

import pytest

from searchkernel.domain import (
    Record,
    RecordHit,
    RecordIdentity,
    canonical_storage_key,
)
from searchkernel.domain import models as models_module


def test_storage_key_includes_optional_workspace_and_source_kind() -> None:
    assert canonical_storage_key(None, "note", "same") != canonical_storage_key(
        None, "commit", "same"
    )


def test_identity_round_trips_through_canonical_storage_key_and_mapping() -> None:
    identity = RecordIdentity("workspace-a", "note", "same/id")

    assert RecordIdentity.from_storage_key(identity.storage_key) == identity
    assert RecordIdentity.from_dict(identity.to_dict()) == identity


def test_identity_rejects_non_canonical_storage_key_encoding() -> None:
    with pytest.raises(ValueError, match="not canonical"):
        RecordIdentity.from_storage_key('record:["workspace-a", "note", "same"]')


def test_from_storage_key_rejects_missing_prefix() -> None:
    with pytest.raises(ValueError, match="must start with"):
        RecordIdentity.from_storage_key('["workspace-a", "note", "same"]')


def test_from_storage_key_rejects_malformed_json() -> None:
    with pytest.raises(ValueError, match="invalid record storage key"):
        RecordIdentity.from_storage_key("record:not-json")


def test_from_storage_key_rejects_wrong_shaped_payload() -> None:
    with pytest.raises(ValueError, match="invalid record storage key"):
        RecordIdentity.from_storage_key('record:["note", "same"]')


def test_from_storage_key_memoizes_repeated_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A repeated storage-key parse must not redo the JSON round trip."""
    models_module._parse_storage_key.cache_clear()
    original_loads = json.loads
    call_count = 0

    def counting_loads(*args: object, **kwargs: object) -> object:
        nonlocal call_count
        call_count += 1
        return original_loads(*args, **kwargs)

    monkeypatch.setattr(json, "loads", counting_loads)

    key = RecordIdentity("workspace-a", "note", "same").storage_key

    first = RecordIdentity.from_storage_key(key)
    second = RecordIdentity.from_storage_key(key)

    assert first == second == RecordIdentity("workspace-a", "note", "same")
    assert call_count == 1


@pytest.mark.parametrize(
    "value",
    [
        (None, "note", "same"),
        ("workspace-a", "note", "same"),
    ],
)
def test_record_hits_expose_the_canonical_identity(value: tuple[str | None, str, str]) -> None:
    workspace_id, source_kind, source_id = value
    expected = RecordIdentity(workspace_id, source_kind, source_id)

    assert RecordHit(expected, 0.5).identity == expected
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
    assert restored.identity == RecordIdentity("workspace-a", "note", "same")


def test_record_serialization_preserves_indexed_text() -> None:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    record = Record(
        "note",
        "same",
        "Title",
        "Raw body",
        timestamp,
        timestamp,
        indexed_text="Searchable text",
    )

    restored = Record.from_dict(record.to_dict())

    assert restored.body == "Raw body"
    assert restored.indexed_text == "Searchable text"


def test_record_deserialization_defaults_indexed_text_for_legacy_payload() -> None:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    payload = Record(
        "note",
        "same",
        "Title",
        "Raw body",
        timestamp,
        timestamp,
    ).to_dict()
    payload.pop("indexed_text")

    restored = Record.from_dict(payload)

    assert restored.indexed_text is None


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
