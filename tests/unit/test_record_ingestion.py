from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

import pytest

from searchkernel.api import SearchKernel
from searchkernel.api import SemanticRecordIngestor as ApiRecordIngestor
from searchkernel.domain import Record, RecordStatus
from searchkernel.indexing.embedding_cache import SQLiteEmbeddingCache
from searchkernel.ingestion import SemanticRecordIngestor


@dataclass
class _Provider:
    model_name: str = "test-model"
    dim: int = 1
    calls: list[list[str]] = field(default_factory=list)

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[float(len(text))] for text in texts]


class _KeywordStore:
    def __init__(self, failures: set[str] | None = None) -> None:
        self.records: list[Record] = []
        self.failures = failures or set()

    def index(self, records: list[Record]) -> None:
        if any(record.source_id in self.failures for record in records):
            raise RuntimeError("keyword write failed")
        self.records.extend(records)

    def search(self, query, k, filters=None):
        return []


class _VectorStore:
    def __init__(self, failures: set[str] | None = None) -> None:
        self.records: list[Record] = []
        self.failures = failures or set()

    def upsert(self, records, model_name, dim) -> None:
        if any(record.source_id in self.failures for record in records):
            raise RuntimeError("vector write failed")
        self.records.extend(records)

    def search(self, query_vector, k, *, model_name, dim, filters=None):
        return []

    def delete(self, record_ids):
        raise AssertionError("record ingestion must not delete source provenance")

    def epoch(self):
        return 0


def _record(
    source_id: str,
    *,
    body: str | None = None,
    indexed_text: str | None = None,
    status: RecordStatus = RecordStatus.ACTIVE,
) -> Record:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    return Record(
        source_kind="notes",
        source_id=source_id,
        title=source_id,
        body=body or f"raw body for {source_id}",
        indexed_text=indexed_text,
        created_at=timestamp,
        updated_at=timestamp,
        status=status,
        workspace_id="workspace",
    )


def _ingestor(
    provider: _Provider,
    *,
    cache,
    keyword=None,
    vector=None,
) -> tuple[SemanticRecordIngestor, _KeywordStore, _VectorStore]:
    keyword = keyword or _KeywordStore()
    vector = vector or _VectorStore()
    return (
        SemanticRecordIngestor(
            embedding_provider=provider,
            keyword_store=keyword,
            vector_store=vector,
            embedding_cache=cache,
        ),
        keyword,
        vector,
    )


@pytest.mark.asyncio
async def test_duplicate_effective_text_reuses_embedding_and_preserves_raw_body(
    tmp_path,
) -> None:
    provider = _Provider()
    cache = SQLiteEmbeddingCache(tmp_path / "embeddings.db", "test-namespace", 1)
    ingestor, keyword, vector = _ingestor(provider, cache=cache)
    first = _record("one", body="raw one", indexed_text="same indexed text")
    second = _record("two", body="raw two", indexed_text="same indexed text")

    receipt = await ingestor.index_records([first, second])

    assert receipt.records[0].source_id == "one"
    assert receipt.records[1].source_id == "two"
    assert provider.calls == [["same indexed text"]]
    assert cache.metrics.hits == 0
    assert cache.metrics.misses == 1
    assert first.embedding == second.embedding == [17.0]
    assert first.body == "raw one"
    assert second.body == "raw two"
    assert [record.storage_key for record in keyword.records] == [
        first.storage_key,
        second.storage_key,
    ]
    assert [record.storage_key for record in vector.records] == [
        first.storage_key,
        second.storage_key,
    ]


@pytest.mark.asyncio
async def test_batch_embedding_encodes_unique_texts_together(tmp_path) -> None:
    provider = _Provider()
    cache = SQLiteEmbeddingCache(tmp_path / "embeddings.db", "batch", 1)
    ingestor, _, _ = _ingestor(provider, cache=cache)

    receipt = await ingestor.index_records(
        [
            _record("one", indexed_text="first text"),
            _record("two", indexed_text="second text"),
            _record("three", indexed_text="first text"),
        ]
    )

    assert receipt.committed == 3
    assert len(provider.calls) == 1
    assert sorted(provider.calls[0]) == ["first text", "second text"]


@pytest.mark.asyncio
async def test_cache_hit_reuses_embedding_across_ingestor_instances(tmp_path) -> None:
    cache_path = tmp_path / "embeddings.db"
    first_provider = _Provider()
    first_ingestor, _, _ = _ingestor(
        first_provider,
        cache=SQLiteEmbeddingCache(cache_path, "shared", 1),
    )
    await first_ingestor.index_records([_record("one", indexed_text="cached text")])

    second_provider = _Provider()
    second_ingestor, _, _ = _ingestor(
        second_provider,
        cache=SQLiteEmbeddingCache(cache_path, "shared", 1),
    )
    receipt = await second_ingestor.index_records(
        [_record("two", indexed_text="cached text")]
    )

    assert receipt.committed == 1
    assert second_provider.calls == []


@pytest.mark.asyncio
async def test_active_and_archived_records_are_upserted_without_deletion() -> None:
    provider = _Provider()
    ingestor, keyword, vector = _ingestor(
        provider,
        cache=SQLiteEmbeddingCache(":memory:", "lifecycle", 1),
    )
    active = _record("active")
    archived = _record("archived", status=RecordStatus.ARCHIVED)

    receipt = await ingestor.index_records([active, archived])

    assert receipt.successful == 2
    assert [record.status for record in keyword.records] == [
        RecordStatus.ACTIVE,
        RecordStatus.ARCHIVED,
    ]
    assert [record.status for record in vector.records] == [
        RecordStatus.ACTIVE,
        RecordStatus.ARCHIVED,
    ]


@pytest.mark.asyncio
async def test_receipts_are_deterministic_and_do_not_advance_checkpoint() -> None:
    provider = _Provider()
    ingestor, _, _ = _ingestor(
        provider,
        cache=SQLiteEmbeddingCache(":memory:", "receipts", 1),
    )
    records = [_record("first"), _record("second")]

    receipt = await ingestor.index_records(
        records,
        checkpoint="source-cursor",
    )

    assert receipt.checkpoint == "source-cursor"
    assert receipt.workspace_id == "workspace"
    assert tuple(result.source_id for result in receipt.records) == (
        "first",
        "second",
    )
    assert tuple(result.status for result in receipt.records) == (
        "committed",
        "committed",
    )


@pytest.mark.asyncio
async def test_strict_failures_stop_at_failed_record() -> None:
    provider = _Provider()
    ingestor, keyword, _ = _ingestor(
        provider,
        cache=SQLiteEmbeddingCache(":memory:", "strict", 1),
        keyword=_KeywordStore({"bad"}),
    )
    records = [_record("first"), _record("bad"), _record("last")]

    receipt = await ingestor.index_records(records, failure_mode="strict")

    assert [result.source_id for result in receipt.records] == ["first", "bad"]
    assert [result.status for result in receipt.records] == ["committed", "failed"]
    assert receipt.failed == 1
    assert [record.source_id for record in keyword.records] == ["first"]


@pytest.mark.asyncio
async def test_lenient_failures_continue_and_report_each_record() -> None:
    provider = _Provider()
    ingestor, keyword, _ = _ingestor(
        provider,
        cache=SQLiteEmbeddingCache(":memory:", "lenient", 1),
        keyword=_KeywordStore({"bad"}),
    )

    receipt = await ingestor.index_records(
        [_record("first"), _record("bad"), _record("last")],
        failure_mode="lenient",
    )

    assert [result.source_id for result in receipt.records] == [
        "first",
        "bad",
        "last",
    ]
    assert [result.status for result in receipt.records] == [
        "committed",
        "failed",
        "committed",
    ]
    assert [record.source_id for record in keyword.records] == ["first", "last"]


def test_public_import_exposes_the_composable_ingestor() -> None:
    assert ApiRecordIngestor is SemanticRecordIngestor


@pytest.mark.asyncio
async def test_public_ingestor_composes_with_search_kernel() -> None:
    provider = _Provider()
    ingestor, _, _ = _ingestor(
        provider,
        cache=SQLiteEmbeddingCache(":memory:", "composition", 1),
    )

    class _Source:
        source_kind = "notes"

        def iter_records(self, since=None):
            async def stream():
                yield _record("composed")

            return stream()

        def change_signal(self):
            return {}

        def cursor_for(self, record):
            return record.source_id

    kernel = SearchKernel.build(content_sources=[_Source()], ingestor=ingestor)
    receipt = await kernel.ingest_source("notes")

    assert receipt.committed == 1
    assert receipt.records[0].source_id == "composed"
