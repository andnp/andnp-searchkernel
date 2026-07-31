import pytest

from searchkernel.domain import Record
from searchkernel.indexing.async_ingestion import AsyncIndexIngestor


def _record(source_id: str) -> Record:
    from datetime import UTC, datetime

    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    return Record(
        source_kind="notes",
        source_id=source_id,
        title=source_id,
        body=source_id,
        created_at=timestamp,
        updated_at=timestamp,
    )


class _BlockingIndexer:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def index_record(self, record: Record) -> bool:
        self.calls.append(record.source_id)
        return record.source_id != "unchanged"


@pytest.mark.asyncio
async def test_async_index_ingestor_offloads_blocking_index_and_reports_idempotency() -> None:
    indexer = _BlockingIndexer()
    ingestor = AsyncIndexIngestor(indexer)

    receipt = await ingestor.index_records(
        [_record("new"), _record("unchanged")],
    )

    assert [outcome.status for outcome in receipt.records] == [
        "committed",
        "skipped",
    ]
    assert indexer.calls == ["new", "unchanged"]


@pytest.mark.asyncio
async def test_async_index_ingestor_strict_mode_stops_after_failure() -> None:
    class FailingIndexer(_BlockingIndexer):
        def index_record(self, record: Record) -> bool:
            if record.source_id == "bad":
                raise ValueError("bad record")
            return super().index_record(record)

    receipt = await AsyncIndexIngestor(FailingIndexer()).index_records(
        [_record("new"), _record("bad"), _record("later")],
    )

    assert [outcome.status for outcome in receipt.records] == [
        "committed",
        "failed",
    ]


@pytest.mark.asyncio
async def test_async_index_ingestor_lenient_mode_continues_after_failure() -> None:
    class FailingIndexer(_BlockingIndexer):
        def index_record(self, record: Record) -> bool:
            if record.source_id == "bad":
                raise ValueError("bad record")
            return super().index_record(record)

    receipt = await AsyncIndexIngestor(FailingIndexer()).index_records(
        [_record("new"), _record("bad"), _record("later")],
        failure_mode="lenient",
    )

    assert [outcome.status for outcome in receipt.records] == [
        "committed",
        "failed",
        "committed",
    ]
