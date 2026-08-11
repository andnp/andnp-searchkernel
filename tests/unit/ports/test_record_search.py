from searchkernel.domain import RecordIdentity
from searchkernel.ports import BatchParentRecordExpander, ParentRecordExpander
from searchkernel.ports.search_results import (
    MAX_FAILURE_DETAIL_LENGTH,
    RecordSearchFailure,
    RecordSearchOutcome,
)


def test_record_search_failure_keeps_legacy_default_construction() -> None:
    """
    The additive detail field does not alter the old constructor shape.
    """
    failure = RecordSearchFailure("keyword", "backend unavailable", "RuntimeError")

    assert failure.detail is None


def test_record_search_failure_bounds_detail() -> None:
    """
    Failure detail stays bounded before it crosses the result boundary.
    """
    detail = "provider response: " + "x" * MAX_FAILURE_DETAIL_LENGTH

    failure = RecordSearchFailure("vector", "unavailable", detail=detail)

    assert failure.detail == detail[:MAX_FAILURE_DETAIL_LENGTH]


def test_record_search_outcome_keeps_legacy_default_construction() -> None:
    """The additive diagnostic projection does not alter old construction."""
    outcome = RecordSearchOutcome()

    assert outcome.results == ()
    assert outcome.diagnostic_evidence is None


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


def test_batch_parent_record_expander_accepts_mapping_contract() -> None:
    class Expander:
        def parent_identities(
            self, identities: list[RecordIdentity]
        ) -> dict[str, RecordIdentity | None]:
            return {identity.storage_key: None for identity in identities}

    assert isinstance(Expander(), BatchParentRecordExpander)
