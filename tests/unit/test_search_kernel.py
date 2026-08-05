from datetime import UTC, datetime
from typing import Any

import pytest

from searchkernel import Record, RecordHit, RecordIdentity, SearchAPI, SearchKernel
from searchkernel.ports.search_results import RecordSearchOutcome, RecordSearchResult


@pytest.mark.asyncio
async def test_public_search_returns_canonical_record_outcome() -> None:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    record = Record(
        workspace_id="workspace",
        source_kind="note",
        source_id="record-1",
        title="Record title",
        body="record body",
        created_at=timestamp,
        updated_at=timestamp,
    )

    class _KeywordStore:
        def search(
            self,
            query: str,
            k: int,
            filters: dict[str, Any] | None = None,
        ) -> list[RecordHit]:
            assert query == "query"
            assert filters == {"statuses": ["active"]}
            return [RecordHit(record.identity, 1.0)][:k]

        def index(self, records: list[Record]) -> None:
            pass

    kernel = SearchKernel.build(
        record_hydrator=lambda identity: record,
        keyword_store=_KeywordStore(),
    )

    assert isinstance(kernel, SearchAPI)
    outcome = await kernel.search(
        "query",
        filters={"statuses": ["active"]},
        limit=1,
    )

    assert isinstance(outcome, RecordSearchOutcome)
    assert len(outcome.results) == 1
    result = outcome.results[0]
    assert isinstance(result, RecordSearchResult)
    assert result.record is record
    assert result.record.identity == RecordIdentity("workspace", "note", "record-1")
    assert result.storage_key == record.storage_key
    assert result.provenance.record_identity == result.record.identity
    assert result.score == pytest.approx(1 / 61)
    assert result.normalized_score == 1.0


@pytest.mark.asyncio
async def test_public_search_preserves_degraded_outcome_shape() -> None:
    kernel = SearchKernel.build(
        record_hydrator=lambda identity: None,
    )

    outcome = await kernel.search("query", limit=1)

    assert isinstance(outcome, RecordSearchOutcome)
    assert outcome.results == ()
    assert outcome.failures == ()
