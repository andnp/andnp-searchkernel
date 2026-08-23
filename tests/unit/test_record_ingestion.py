from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime

import pytest

from searchkernel.api import SearchKernel
from searchkernel.api import SemanticRecordIngestor as ApiRecordIngestor
from searchkernel.domain import Chunk, Record, RecordStatus
from searchkernel.indexing.embedding_cache import SQLiteEmbeddingCache
from searchkernel.indexing.semantic import semantic_input_for_record
from searchkernel.ingestion import SemanticRecordIngestor
from searchkernel.ingestion.records import (
    _merge_stage_outcomes,
    _RecordBatchMaterializer,
)
from searchkernel.ports.content_source import RecordIngestionResult


@dataclass
class _Provider:
    model_name: str = "test-model"
    dim: int = 1
    calls: list[list[str]] = field(default_factory=list)

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[float(len(text))] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


class _BlockingProvider(_Provider):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.started.set()
        self.release.wait(timeout=5)
        return super().embed(texts)


class _BatchFailingProvider(_Provider):
    def embed(self, texts: list[str]) -> list[list[float]]:
        if len(texts) > 1:
            raise RuntimeError("batch embedding failed")
        return super().embed(texts)


@dataclass
class _Chunker:
    def chunk_record(self, record: Record) -> list[Chunk]:
        return [
            Chunk(
                "child",
                record.source_id,
                "child body",
                {"header_path": "Child"},
                0,
            )
        ]


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


class _BlockingKeywordStore(_KeywordStore):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def index(self, records: list[Record]) -> None:
        self.started.set()
        self.release.wait(timeout=5)
        super().index(records)


class _VectorStore:
    def __init__(self, failures: set[str] | None = None) -> None:
        self.records: list[Record] = []
        self.failures = failures or set()

    def upsert(self, records, model_name, dim) -> None:
        if any(record.source_id in self.failures for record in records):
            raise RuntimeError("vector write failed")
        by_key = {record.storage_key: record for record in self.records}
        by_key.update({record.storage_key: record for record in records})
        self.records = list(by_key.values())

    def search(self, query_vector, k, *, model_name, dim, filters=None):
        return []

    def delete(self, record_ids):
        raise AssertionError("record ingestion must not delete source provenance")

    def epoch(self):
        return 0


def _record(
    source_id: str,
    *,
    title: str | None = None,
    body: str | None = None,
    indexed_text: str | None = None,
    status: RecordStatus = RecordStatus.ACTIVE,
) -> Record:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    return Record(
        source_kind="notes",
        source_id=source_id,
        title=source_id if title is None else title,
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
async def test_distinct_titled_texts_do_not_reuse_embedding_and_preserve_raw_body(
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
    expected_texts = [
        "Title: one\n\nsame indexed text",
        "Title: two\n\nsame indexed text",
    ]
    assert len(provider.calls) == 1
    assert sorted(provider.calls[0]) == sorted(expected_texts)
    assert cache.metrics.hits == 0
    assert cache.metrics.misses == 2
    assert first.embedding == [float(len(expected_texts[0]))]
    assert second.embedding == [float(len(expected_texts[1]))]
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
    assert sorted(provider.calls[0]) == sorted(
        [
            "Title: one\n\nfirst text",
            "Title: two\n\nsecond text",
            "Title: three\n\nfirst text",
        ]
    )


@pytest.mark.asyncio
async def test_keyword_stage_commits_while_embedding_is_blocked(tmp_path) -> None:
    provider = _BlockingProvider()
    ingestor, keyword, vector = _ingestor(
        provider,
        cache=SQLiteEmbeddingCache(tmp_path / "embeddings.db", "stages", 1),
    )
    task = asyncio.create_task(ingestor.index_records([_record("one")]))

    started = await asyncio.to_thread(provider.started.wait, 5)
    keyword_ids = [record.source_id for record in keyword.records]
    vector_ids = [record.source_id for record in vector.records]
    provider.release.set()
    receipt = await task

    assert started
    assert keyword_ids == ["one"]
    assert vector_ids == []
    assert receipt.committed == 1


@pytest.mark.asyncio
async def test_cache_hit_reuses_embedding_across_ingestor_instances(tmp_path) -> None:
    cache_path = tmp_path / "embeddings.db"
    first_provider = _Provider()
    first_ingestor, _, _ = _ingestor(
        first_provider,
        cache=SQLiteEmbeddingCache(cache_path, "shared", 1),
    )
    await first_ingestor.index_records(
        [_record("one", title="Shared title", indexed_text="cached text")]
    )

    second_provider = _Provider()
    second_ingestor, _, _ = _ingestor(
        second_provider,
        cache=SQLiteEmbeddingCache(cache_path, "shared", 1),
    )
    receipt = await second_ingestor.index_records(
        [_record("two", title="Shared title", indexed_text="cached text")]
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
    """Receipts preserve the caller checkpoint without advancing it.

    Checkpoint persistence remains the source caller's responsibility.
    """
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
async def test_empty_ingestion_preserves_checkpoint_and_has_no_outcomes() -> None:
    """Empty input returns a receipt without starting either indexing stage.

    An existing checkpoint is carried through unchanged for the caller.
    """
    provider = _Provider()
    ingestor, keyword, vector = _ingestor(
        provider,
        cache=SQLiteEmbeddingCache(":memory:", "empty", 1),
    )

    receipt = await ingestor.index_records([], checkpoint="cursor-1")

    assert receipt.source_kind == ""
    assert receipt.workspace_id is None
    assert receipt.checkpoint == "cursor-1"
    assert receipt.records == ()
    assert provider.calls == []
    assert keyword.records == []
    assert vector.records == []


def test_ingestor_rejects_invalid_batch_size() -> None:
    """The ingestor rejects a non-positive semantic batch size at construction.

    Invalid configuration must fail before any source records are processed.
    """
    with pytest.raises(ValueError, match="embedding_batch_size"):
        SemanticRecordIngestor(
            embedding_provider=_Provider(),
            keyword_store=_KeywordStore(),
            vector_store=_VectorStore(),
            embedding_cache=SQLiteEmbeddingCache(":memory:", "invalid", 1),
            embedding_batch_size=0,
        )


@pytest.mark.asyncio
async def test_semantic_batch_failure_falls_back_to_individual_records() -> None:
    """A failed semantic batch retries records individually.

    Successful per-record retries should produce a committed receipt.
    """
    provider = _BatchFailingProvider()
    ingestor, _, vector = _ingestor(
        provider,
        cache=SQLiteEmbeddingCache(":memory:", "semantic-fallback", 1),
    )

    receipt = await ingestor.index_records([_record("first"), _record("second")])

    assert receipt.committed == 2
    assert len(provider.calls) == 2
    assert [record.source_id for record in vector.records] == ["first", "second"]


@pytest.mark.asyncio
async def test_receipt_attributes_failures_to_the_stage_and_record() -> None:
    """Lenient receipts retain the failing stage for each source record.

    Independent keyword and semantic failures must not collapse into one error.
    """
    first = _record("keyword-failure")
    second = _record("vector-failure")
    ingestor, _, _ = _ingestor(
        _Provider(),
        cache=SQLiteEmbeddingCache(":memory:", "attribution", 1),
        keyword=_KeywordStore({first.source_id}),
        vector=_VectorStore({second.source_id}),
    )

    receipt = await ingestor.index_records([first, second], failure_mode="lenient")

    assert [result.source_id for result in receipt.records] == [
        "keyword-failure",
        "vector-failure",
    ]
    assert receipt.records[0].error is not None
    assert receipt.records[1].error is not None
    assert "keyword stage:" in receipt.records[0].error
    assert "semantic stage:" in receipt.records[1].error


@pytest.mark.asyncio
async def test_chunk_expansion_indexes_children_but_reports_parent_receipt() -> None:
    """Chunk expansion writes parent and child records under one parent outcome.

    The receipt remains source-oriented while both searchable records advance.
    """
    provider = _Provider()
    ingestor, keyword, vector = _ingestor(
        provider,
        cache=SQLiteEmbeddingCache(":memory:", "chunks", 1),
    )
    ingestor.chunker = _Chunker()
    parent = _record("parent")

    receipt = await ingestor.index_records([parent])

    assert receipt.committed == 1
    assert [record.source_id for record in keyword.records] == [
        "parent",
        "parent#chunk:child",
    ]
    assert [record.source_id for record in vector.records] == [
        "parent",
        "parent#chunk:child",
    ]
    assert provider.calls[0] == [
        "Title: parent\n\nraw body for parent",
        "Title: parent\n\nchild body",
    ]


def test_materializer_rejects_unknown_record_storage_keys() -> None:
    """Batch materialization rejects vectors for records outside its batch.

    Unknown identities must not silently attach embeddings to another record.
    """
    record = _record("known")
    semantic_input = semantic_input_for_record(record, "materializer")

    with pytest.raises(ValueError, match="does not belong to the record batch"):
        _RecordBatchMaterializer([record], "test-model").materialize(
            "missing-storage-key",
            [1.0],
            semantic_input,
        )


@pytest.mark.asyncio
async def test_cancelling_ingestion_cancels_both_sibling_stages() -> None:
    """Cancelling an in-flight ingestion propagates to both stage tasks.

    Blocking focused fakes make sibling cancellation deterministic and visible.
    """
    provider = _BlockingProvider()
    keyword = _BlockingKeywordStore()
    ingestor, _, _ = _ingestor(
        provider,
        cache=SQLiteEmbeddingCache(":memory:", "cancel", 1),
        keyword=keyword,
    )
    task = asyncio.create_task(ingestor.index_records([_record("one")]))

    await asyncio.to_thread(provider.started.wait, 5)
    await asyncio.to_thread(keyword.started.wait, 5)
    task.cancel()
    provider.release.set()
    keyword.release.set()

    with pytest.raises(asyncio.CancelledError):
        await task


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


@pytest.mark.asyncio
async def test_failed_record_can_be_retried_after_other_records_commit() -> None:
    provider = _Provider()
    keyword = _KeywordStore({"retry-me"})
    ingestor, keyword, vector = _ingestor(
        provider,
        cache=SQLiteEmbeddingCache(":memory:", "retry", 1),
        keyword=keyword,
    )
    records = [_record("kept"), _record("retry-me")]

    first = await ingestor.index_records(records, failure_mode="lenient")

    assert [result.status for result in first.records] == ["committed", "failed"]
    assert [record.source_id for record in keyword.records] == ["kept"]

    keyword.failures.clear()
    second = await ingestor.index_records([records[1]], failure_mode="lenient")

    assert second.committed == 1
    assert second.failures == ()
    assert [record.source_id for record in keyword.records] == ["kept", "retry-me"]
    assert [record.source_id for record in vector.records] == ["kept", "retry-me"]


@pytest.mark.asyncio
async def test_reconciliation_does_not_promote_cancelled_stage_to_committed() -> None:
    records = [_record("one")]
    receipt = _merge_stage_outcomes(
        records,
        checkpoint=None,
        failure_mode="lenient",
        keyword_outcomes=[
            RecordIngestionResult("notes", "one", "workspace", "committed")
        ],
        semantic_outcomes=[
            RecordIngestionResult(
                "notes",
                "one",
                "workspace",
                "cancelled",
                error="semantic worker stopped",
            )
        ],
    )

    assert receipt.committed == 0
    assert receipt.failed == 1
    assert receipt.records[0].error == (
        "semantic stage: semantic worker stopped"
    )


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
