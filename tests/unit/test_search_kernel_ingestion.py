from collections.abc import Iterable
from datetime import UTC, datetime

import pytest

from searchkernel import SearchKernel
from searchkernel.domain import ChangeSignal, Record


class _ContentSource:
    source_kind = "notes"

    def __init__(self, records: list[Record]):
        self.records = records
        self.since_values: list[str | None] = []

    def iter_records(self, since: str | None = None) -> Iterable[Record]:
        self.since_values.append(since)
        return iter(self.records)

    def change_signal(self) -> ChangeSignal:
        return {"poll_interval": 60}


class _RecordIngestor:
    def __init__(self):
        self.records: list[Record] = []

    def index_record(self, record: Record) -> bool:
        self.records.append(record)
        return True


def _record(source_id: str) -> Record:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    return Record(
        source_kind="notes",
        source_id=source_id,
        title=source_id,
        body=f"Body for {source_id}",
        created_at=timestamp,
        updated_at=timestamp,
    )


def test_ingest_source_iterates_records_returns_count_and_passes_cursor():
    source = _ContentSource([_record("one"), _record("two")])
    ingestor = _RecordIngestor()
    kernel = SearchKernel.build(content_sources=[source], ingestor=ingestor)

    assert kernel.ingest_source("notes", since="cursor-1") == 2
    assert source.since_values == ["cursor-1"]
    assert [record.source_id for record in ingestor.records] == ["one", "two"]


def test_ingest_source_can_be_registered_after_kernel_build():
    source = _ContentSource([_record("one")])
    ingestor = _RecordIngestor()
    kernel = SearchKernel.build(ingestor=ingestor)

    kernel.register_content_source(source)

    assert kernel.ingest_source("notes") == 1


def test_ingest_source_rejects_unknown_source():
    kernel = SearchKernel.build(ingestor=_RecordIngestor())

    with pytest.raises(KeyError, match="No content source registered"):
        kernel.ingest_source("missing")


def test_ingest_source_rejects_missing_ingestor():
    source = _ContentSource([_record("one")])
    kernel = SearchKernel.build(content_sources=[source])

    with pytest.raises(RuntimeError, match="no record ingestor"):
        kernel.ingest_source("notes")

    assert source.since_values == []


def test_kernels_keep_content_sources_and_ingestors_isolated():
    first_source = _ContentSource([_record("first")])
    second_source = _ContentSource([_record("second")])
    first_ingestor = _RecordIngestor()
    second_ingestor = _RecordIngestor()
    first_kernel = SearchKernel.build(
        content_sources=[first_source], ingestor=first_ingestor
    )
    second_kernel = SearchKernel.build(
        content_sources=[second_source], ingestor=second_ingestor
    )

    assert first_kernel.ingest_source("notes") == 1
    assert [record.source_id for record in first_ingestor.records] == ["first"]
    assert second_ingestor.records == []
    assert second_kernel.registry is not first_kernel.registry
    assert second_source.since_values == []
