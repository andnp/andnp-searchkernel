from searchkernel.domain import RecordIdentity
from searchkernel.ports import ParentRecordExpander


def test_parent_record_expander_requires_canonical_identity() -> None:
    class Expander:
        def parent_identity(
            self,
            identity: RecordIdentity,
        ) -> RecordIdentity | None:
            return RecordIdentity(
                identity.workspace_id,
                identity.source_kind,
                "parent",
            )

    assert isinstance(Expander(), ParentRecordExpander)
