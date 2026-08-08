"""Identity cases shared by local and Postgres record backends."""

from datetime import UTC, datetime

import pytest

from searchkernel.domain import Record, RecordHit, RecordIdentity


@pytest.mark.parametrize(
    ("workspace_id", "source_kind", "source_id"),
    [
        (None, "note", "same"),
        ("workspace-a", "note", "same"),
        ("workspace-a", "commit", "same"),
        ("workspace-b", "note", "same"),
        ("workspace-a", "note", "path/with:delimiters"),
    ],
)
def test_backend_identity_round_trips_without_collisions(
    workspace_id: str | None,
    source_kind: str,
    source_id: str,
) -> None:
    identity = RecordIdentity(workspace_id, source_kind, source_id)

    assert RecordIdentity.from_storage_key(identity.storage_key) == identity
    assert RecordIdentity.from_dict(identity.to_dict()) == identity


def test_backend_identity_keeps_workspace_and_source_kind_distinct() -> None:
    identities = [
        RecordIdentity(None, "note", "same"),
        RecordIdentity("workspace-a", "note", "same"),
        RecordIdentity("workspace-a", "commit", "same"),
        RecordIdentity("workspace-b", "note", "same"),
    ]

    assert len({identity.storage_key for identity in identities}) == len(identities)


def test_record_and_hit_expose_the_same_canonical_identity() -> None:
    record = Record(
        workspace_id="workspace-a",
        source_kind="note",
        source_id="same/id",
        title="Title",
        body="Body",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    hit = RecordHit(record.identity, 0.75)

    assert hit.identity == record.identity
    assert hit.storage_key == record.storage_key
    assert hit.workspace_id == record.workspace_id
    assert hit.source_kind == record.source_kind
    assert hit.source_id == record.source_id
