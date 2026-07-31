import asyncio
from datetime import UTC, datetime

import pytest

from searchkernel.domain import Record, RecordStatus
from searchkernel.indexing.checkpoints import MemoryCheckpointStore
from searchkernel.kernel import SearchKernel
from searchkernel.ports.content_source import (
    IngestionError,
    IngestionReceipt,
    RecordIngestionResult,
)


class _ContentSource:
    source_kind = "notes"

    def __init__(self, records: list[Record]) -> None:
        self.records = records
        self.since_values: list[str | None] = []

    def iter_records(self, since: str | None = None):
        self.since_values.append(since)

        async def stream():
            for record in self.records:
                yield record

        return stream()

    def change_signal(self):
        return {"poll_interval": 60}

    def cursor_for(self, record: Record) -> str:
        return str(record.metadata["cursor"])


class _RecordIngestor:
    def __init__(self, failures: set[str] | None = None) -> None:
        self.records: list[Record] = []
        self.failures = failures or set()
        self.calls: list[list[str]] = []

    async def index_records(
        self,
        records,
        *,
        checkpoint=None,
        failure_mode="strict",
    ) -> IngestionReceipt:
        self.calls.append([record.source_id for record in records])
        outcomes = []
        for record in records:
            if record.source_id in self.failures:
                outcomes.append(
                    RecordIngestionResult(
                        source_kind=record.source_kind,
                        source_id=record.source_id,
                        workspace_id=record.workspace_id,
                        status="failed",
                        error="injected failure",
                    )
                )
                if failure_mode == "strict":
                    break
            else:
                self.records.append(record)
                outcomes.append(
                    RecordIngestionResult(
                        source_kind=record.source_kind,
                        source_id=record.source_id,
                        workspace_id=record.workspace_id,
                        status="committed",
                    )
                )
        return IngestionReceipt(
            source_kind=records[0].source_kind if records else "",
            workspace_id=records[0].workspace_id if records else None,
            checkpoint=checkpoint,
            records=tuple(outcomes),
        )


def _record(source_id: str, *, cursor: str | None = None) -> Record:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    return Record(
        source_kind="notes",
        source_id=source_id,
        title=source_id,
        body=f"Body for {source_id}",
        created_at=timestamp,
        updated_at=timestamp,
        metadata={"cursor": cursor or source_id},
        workspace_id="workspace",
    )


@pytest.mark.asyncio
async def test_ingest_source_batches_records_and_persists_after_commit() -> None:
    source = _ContentSource([_record("one"), _record("two"), _record("three")])
    ingestor = _RecordIngestor()
    checkpoints = MemoryCheckpointStore()
    kernel = SearchKernel.build(
        content_sources=[source],
        ingestor=ingestor,
    )

    receipt = await kernel.ingest_source(
        "notes",
        batch_size=2,
        checkpoint_store=checkpoints,
    )

    assert receipt.checkpoint == "three"
    assert receipt.successful == 3
    assert source.since_values == [None]
    assert ingestor.calls == [["one", "two"], ["three"]]
    assert await checkpoints.load("notes") == "three"


@pytest.mark.asyncio
async def test_strict_mode_does_not_advance_failed_batch() -> None:
    source = _ContentSource([_record("one"), _record("bad"), _record("three")])
    ingestor = _RecordIngestor({"bad"})
    checkpoints = MemoryCheckpointStore()
    kernel = SearchKernel.build(content_sources=[source], ingestor=ingestor)

    with pytest.raises(IngestionError) as error:
        await kernel.ingest_source(
            "notes",
            batch_size=2,
            checkpoint_store=checkpoints,
        )

    assert error.value.receipt.checkpoint is None
    assert await checkpoints.load("notes", "workspace") is None
    assert [record.source_id for record in ingestor.records] == ["one"]


@pytest.mark.asyncio
async def test_lenient_mode_returns_failures_and_advances_contiguous_successes() -> None:
    source = _ContentSource([_record("one"), _record("bad"), _record("three")])
    ingestor = _RecordIngestor({"bad"})
    checkpoints = MemoryCheckpointStore()
    kernel = SearchKernel.build(content_sources=[source], ingestor=ingestor)

    receipt = await kernel.ingest_source(
        "notes",
        batch_size=2,
        failure_mode="lenient",
        checkpoint_store=checkpoints,
    )

    assert receipt.failed == 1
    assert receipt.checkpoint == "one"
    assert await checkpoints.load("notes") == "one"
    assert [record.source_id for record in ingestor.records] == ["one", "three"]


@pytest.mark.asyncio
async def test_retry_resumes_from_durable_checkpoint_and_is_idempotent() -> None:
    records = [_record("one"), _record("two")]
    source = _ContentSource(records)
    ingestor = _RecordIngestor()
    checkpoints = MemoryCheckpointStore()
    kernel = SearchKernel.build(content_sources=[source], ingestor=ingestor)

    first = await kernel.ingest_source(
        "notes", checkpoint_store=checkpoints, batch_size=2
    )
    second = await kernel.ingest_source(
        "notes", checkpoint_store=checkpoints, batch_size=2
    )

    assert first.checkpoint == second.checkpoint == "two"
    assert source.since_values == [None, "two"]


@pytest.mark.asyncio
async def test_ingest_source_rejects_sync_sources() -> None:
    class SyncSource(_ContentSource):
        def iter_records(self, since=None):
            return iter(self.records)

    kernel = SearchKernel.build(
        content_sources=[SyncSource([_record("one")])],
        ingestor=_RecordIngestor(),
    )

    with pytest.raises(TypeError, match="async iterator"):
        await kernel.ingest_source("notes")


@pytest.mark.asyncio
async def test_cancellation_does_not_persist_checkpoint() -> None:
    class BlockingIngestor(_RecordIngestor):
        async def index_records(self, records, **kwargs):
            await asyncio.sleep(10)
            return await super().index_records(records, **kwargs)

    checkpoints = MemoryCheckpointStore()
    kernel = SearchKernel.build(
        content_sources=[_ContentSource([_record("one")])],
        ingestor=BlockingIngestor(),
    )
    task = asyncio.create_task(
        kernel.ingest_source("notes", checkpoint_store=checkpoints)
    )
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert await checkpoints.load("notes") is None


@pytest.mark.asyncio
async def test_tombstone_status_is_passed_through_with_identity() -> None:
    record = _record("deleted")
    record.status = RecordStatus.ARCHIVED
    ingestor = _RecordIngestor()
    kernel = SearchKernel.build(
        content_sources=[_ContentSource([record])],
        ingestor=ingestor,
    )

    receipt = await kernel.ingest_source("notes")

    assert receipt.records[0].workspace_id == "workspace"
    assert ingestor.records[0].status is RecordStatus.ARCHIVED
