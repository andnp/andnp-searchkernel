from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest

from searchkernel.domain import Chunk, Record, RecordStatus
from searchkernel.indexing import checkpoints as checkpoints_module
from searchkernel.indexing.bootstrap_checkpoint import (
    BootstrapCheckpoint,
    BootstrapFileStamp,
    get_bootstrap_availability,
    load_bootstrap_checkpoint,
    publish_bootstrap_availability,
    save_bootstrap_checkpoint,
)
from searchkernel.indexing.checkpoints import JsonCheckpointStore
from searchkernel.indexing.coordinator import ResumableSemanticCoordinator
from searchkernel.indexing.embedding_cache import SQLiteEmbeddingCache
from searchkernel.indexing.runtime_readiness import SearchAvailability
from searchkernel.indexing.semantic import (
    SemanticInput,
    SemanticWorkPlanner,
)
from searchkernel.indexing.stages import (
    GraphStage,
    IndexStage,
    KeywordStage,
    PreparedIndexBatch,
    PreparedIndexRecord,
    StageCounters,
    StageResult,
)
from searchkernel.indices import LocalRecordBackend
from searchkernel.ports.content_source import (
    IngestionError,
    IngestionFailureMode,
    IngestionReceipt,
    RecordIngestionResult,
    RecordIngestor,
)

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _record(
    source_id: str,
    *,
    cursor: str | None = None,
    body: str | None = None,
    status: RecordStatus = RecordStatus.ACTIVE,
    metadata: dict[str, object] | None = None,
) -> Record:
    return Record(
        source_kind="notes",
        source_id=source_id,
        title=source_id,
        body=body or f"body for {source_id}",
        created_at=_NOW,
        updated_at=_NOW,
        metadata={"cursor": cursor or source_id, **(metadata or {})},
        status=status,
        workspace_id="workspace",
    )


def _prepared(
    record: Record,
    *,
    content: str | None = None,
    chunk_count: int = 1,
) -> PreparedIndexRecord:
    chunks = [
        Chunk(
            chunk_id=f"{record.source_id}#chunk-{index}",
            record_id=record.source_id,
            content=content or record.body,
            metadata={
                "header_path": "",
                "start_pos": 0,
                "end_pos": len(content or record.body),
                "file_path": record.uri or f"{record.source_id}.md",
                "source_file": record.uri or f"{record.source_id}.md",
                "modified_time": _NOW.isoformat(),
                "title": record.title,
                "tags": [],
            },
            chunk_index=index,
        )
        for index in range(chunk_count)
    ]
    return PreparedIndexRecord(
        file_path=record.uri or f"{record.source_id}.md",
        parser=object(),
        record=record,
        chunks=chunks,
        graph_metadata={"source_id": record.source_id},
    )


class _Encoder:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.fail = fail

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        values = tuple(texts)
        self.calls.append(values)
        if self.fail:
            raise RuntimeError("encoder interrupted")
        return tuple((float(len(text)), 1.0) for text in values)


class _Materializer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def materialize(
        self,
        source_id: str,
        vector: Sequence[float],
        semantic_input: SemanticInput,
    ) -> None:
        del vector
        self.calls.append((source_id, semantic_input.content_hash))


class _RecordingChunkWriter:
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []

    def add_chunks(self, chunks: list[Chunk]) -> None:
        self.chunks.extend(chunks)


class _BulkGraph:
    def __init__(self) -> None:
        self.nodes: list[tuple[str, dict]] = []
        self.edges: list[tuple[str, str, str, str]] = []

    def add_nodes(self, nodes: list[tuple[str, dict]]) -> None:
        self.nodes.extend(nodes)

    def add_edges(self, edges: list[tuple[str, str, str, str]]) -> None:
        self.edges.extend(edges)


class _FailingStage:
    name = "failing"

    def __init__(self, failed_ids: set[str]) -> None:
        self.failed_ids = failed_ids
        self.calls: list[tuple[str, ...]] = []

    def apply(self, batch: PreparedIndexBatch) -> StageResult:
        ids = tuple(record.record.source_id for record in batch.records)
        self.calls.append(ids)
        if self.failed_ids.intersection(ids):
            raise RuntimeError("stage failed")
        return StageResult(self.name, StageCounters(records=len(ids)))


class _NoopStage:
    name = "noop"

    def apply(self, batch: PreparedIndexBatch) -> StageResult:
        return StageResult(self.name, StageCounters(records=len(batch.records)))


class _CursorSource:
    source_kind = "notes"

    def __init__(self, records: Sequence[Record]) -> None:
        self.records = tuple(records)
        self.since_values: list[str | None] = []

    def iter_records(self, since: str | None = None) -> AsyncIterator[Record]:
        self.since_values.append(since)
        start = int(since or "0")

        async def stream() -> AsyncIterator[Record]:
            for record in self.records:
                if int(str(record.metadata["cursor"])) > start:
                    yield record

        return stream()

    def change_signal(self) -> dict[str, int]:
        return {"poll_interval": 60}

    def cursor_for(self, record: Record) -> str:
        return str(record.metadata["cursor"])


class _LifecycleIngestor:
    def __init__(self, backend: LocalRecordBackend) -> None:
        self.backend = backend
        self.seen: list[str] = []

    async def index_records(
        self,
        records: Sequence[Record],
        *,
        checkpoint: str | None = None,
        failure_mode: IngestionFailureMode = "strict",
    ) -> IngestionReceipt:
        del checkpoint, failure_mode
        outcomes: list[RecordIngestionResult] = []
        for record in records:
            self.seen.append(record.source_id)
            moved_from = record.metadata.get("moved_from")
            if isinstance(moved_from, str):
                self.backend.delete([moved_from])
            if record.status is RecordStatus.ACTIVE:
                self.backend.index([record])
            else:
                self.backend.delete([record.storage_key])
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
            checkpoint=None,
            records=tuple(outcomes),
        )


def _coordinator(
    tmp_path: Path,
    *,
    namespace: str = "test-model",
    encoder: _Encoder | None = None,
    materializer: _Materializer | None = None,
    stages: Sequence[IndexStage] = (),
    checkpoint_store: JsonCheckpointStore | None = None,
    record_ingestor: RecordIngestor | None = None,
) -> tuple[
    ResumableSemanticCoordinator,
    SQLiteEmbeddingCache,
    _Encoder,
    _Materializer,
]:
    selected_encoder = encoder or _Encoder()
    selected_materializer = materializer or _Materializer()
    cache = SQLiteEmbeddingCache(
        tmp_path / "embeddings.db",
        namespace,
        dimension=2,
    )
    coordinator = ResumableSemanticCoordinator(
        planner=SemanticWorkPlanner(namespace, dimension=2),
        cache=cache,
        encoder=selected_encoder,
        materializer=selected_materializer,
        stages=stages,
        record_ingestor=record_ingestor,
        checkpoint_store=checkpoint_store,
    )
    return coordinator, cache, selected_encoder, selected_materializer


def _checkpoint_for_batch(batch: PreparedIndexBatch) -> str:
    return str(batch.records[-1].record.metadata["cursor"])


@pytest.mark.asyncio
async def test_prepared_records_are_bounded_and_stage_progress_is_ordered(
    tmp_path: Path,
) -> None:
    keyword = _RecordingChunkWriter()
    graph = _BulkGraph()
    availability = SearchAvailability(
        lexical="complete",
        graph="complete",
        semantic_coarse="backfilling",
        semantic_fine="unavailable",
    )
    coordinator, cache, encoder, materializer = _coordinator(
        tmp_path,
        stages=(KeywordStage(keyword), GraphStage(graph)),
        checkpoint_store=JsonCheckpointStore(tmp_path / "checkpoint.json"),
    )
    events: list[tuple[int, str]] = []
    records = [
        _prepared(_record(f"doc-{index}", cursor=str(index)), chunk_count=2)
        for index in range(1, 6)
    ]

    receipt = await coordinator.run_prepared_records(
        records,
        source_kind="notes",
        max_records=2,
        max_chunks=4,
        checkpoint_for_batch=_checkpoint_for_batch,
        availability=availability,
        progress=lambda event: events.append((event.batch_index, event.stage)),
    )

    assert [set(call) for call in encoder.calls] == [
        {"body for doc-1", "body for doc-2"},
        {"body for doc-3", "body for doc-4"},
        {"body for doc-5"},
    ]
    assert len(materializer.calls) == 10
    assert receipt.checkpoint == "5"
    assert receipt.checkpoint_persisted
    assert receipt.availability == availability
    assert events == [
        (0, "keyword"),
        (0, "graph"),
        (0, "semantic"),
        (0, "checkpoint"),
        (1, "keyword"),
        (1, "graph"),
        (1, "semantic"),
        (1, "checkpoint"),
        (2, "keyword"),
        (2, "graph"),
        (2, "semantic"),
        (2, "checkpoint"),
    ]
    assert all(
        event.checkpoint_persisted
        for event in receipt.progress
        if event.stage == "checkpoint"
    )
    assert cache.metrics.writes == 5


@pytest.mark.asyncio
async def test_duplicate_semantic_inputs_reuse_cache_across_coordinator_restart(
    tmp_path: Path,
) -> None:
    records = [
        _prepared(_record("one"), content="same semantic text"),
        _prepared(_record("two"), content="same semantic text"),
    ]
    first, first_cache, first_encoder, first_materializer = _coordinator(
        tmp_path,
        stages=(_NoopStage(),),
    )
    first_receipt = await first.run_prepared_records(
        records,
        source_kind="notes",
    )
    first_cache.close()

    second, second_cache, second_encoder, second_materializer = _coordinator(
        tmp_path,
        stages=(_NoopStage(),),
    )
    second_receipt = await second.run_prepared_records(
        records,
        source_kind="notes",
    )

    assert first_receipt.semantic_progress[0].total == 1
    assert first_receipt.semantic_progress[0].cache_misses == 1
    assert first_encoder.calls == [("same semantic text",)]
    assert len(first_materializer.calls) == 2
    assert second_receipt.semantic_progress[0].cache_hits == 1
    assert second_receipt.semantic_progress[0].cache_misses == 0
    assert second_encoder.calls == []
    assert len(second_materializer.calls) == 2
    second_cache.close()


@pytest.mark.asyncio
async def test_restart_uses_durable_checkpoint_without_repeating_completed_batches(
    tmp_path: Path,
) -> None:
    records = [
        _record("one", cursor="1"),
        _record("two", cursor="2"),
        _record("three", cursor="3"),
        _record("four", cursor="4"),
    ]
    source = _CursorSource(records)
    checkpoint_store = JsonCheckpointStore(tmp_path / "checkpoint.json")
    keyword = _RecordingChunkWriter()
    graph = _BulkGraph()
    first, first_cache, first_encoder, _ = _coordinator(
        tmp_path,
        stages=(KeywordStage(keyword), GraphStage(graph)),
        checkpoint_store=checkpoint_store,
    )

    async def interrupt_after_checkpoint(event) -> None:
        if event.batch_index == 0 and event.stage == "checkpoint":
            raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        await first.run_source(
            source,
            prepare_batch=lambda batch: PreparedIndexBatch.from_records(
                [_prepared(record) for record in batch]
            ),
            batch_size=2,
            progress=interrupt_after_checkpoint,
        )

    assert await checkpoint_store.load("notes") == "2"
    assert source.since_values == [None]
    assert first_encoder.calls == [("body for one", "body for two")]
    first_cache.close()

    second, second_cache, second_encoder, _ = _coordinator(
        tmp_path,
        stages=(KeywordStage(keyword), GraphStage(graph)),
        checkpoint_store=checkpoint_store,
    )
    receipt = await second.run_source(
        source,
        prepare_batch=lambda batch: PreparedIndexBatch.from_records(
            [_prepared(record) for record in batch]
        ),
        batch_size=2,
    )

    assert receipt.checkpoint == "4"
    assert source.since_values == [None, "2"]
    assert second_encoder.calls == [("body for three", "body for four")]
    assert len(keyword.chunks) == 4
    second_cache.close()


@pytest.mark.asyncio
async def test_strict_failure_raises_and_lenient_failure_continues_without_crossing_gap(
    tmp_path: Path,
) -> None:
    records = [
        _prepared(_record("one", cursor="1")),
        _prepared(_record("bad", cursor="2")),
        _prepared(_record("three", cursor="3")),
    ]
    strict_store = JsonCheckpointStore(tmp_path / "strict.json")
    strict_failure = _FailingStage({"bad"})
    strict, strict_cache, _, _ = _coordinator(
        tmp_path / "strict",
        stages=(strict_failure,),
        checkpoint_store=strict_store,
    )
    with pytest.raises(IngestionError) as error:
        await strict.run_prepared_records(
            records,
            source_kind="notes",
            max_records=1,
            checkpoint_for_batch=_checkpoint_for_batch,
        )
    assert error.value.receipt.failed == 1
    assert await strict_store.load("notes") == "1"
    strict_cache.close()

    lenient_store = JsonCheckpointStore(tmp_path / "lenient.json")
    lenient_failure = _FailingStage({"bad"})
    lenient, lenient_cache, _, _ = _coordinator(
        tmp_path / "lenient",
        stages=(lenient_failure,),
        checkpoint_store=lenient_store,
    )
    receipt = await lenient.run_prepared_records(
        records,
        source_kind="notes",
        max_records=1,
        checkpoint_for_batch=_checkpoint_for_batch,
        failure_mode="lenient",
    )
    assert [result.status for result in receipt.records] == [
        "committed",
        "failed",
        "committed",
    ]
    assert await lenient_store.load("notes") == "1"
    assert lenient_failure.calls == [("one",), ("bad",), ("three",)]
    lenient_cache.close()


@pytest.mark.asyncio
async def test_failed_batch_can_be_retried_without_crossing_checkpoint_gap(
    tmp_path: Path,
) -> None:
    records = [
        _record("one", cursor="1"),
        _record("bad", cursor="2"),
        _record("three", cursor="3"),
    ]
    source = _CursorSource(records)
    checkpoint_store = JsonCheckpointStore(tmp_path / "checkpoint.json")
    failing_stage = _FailingStage({"bad"})
    coordinator, cache, _, _ = _coordinator(
        tmp_path,
        stages=(failing_stage,),
        checkpoint_store=checkpoint_store,
    )
    prepare = lambda batch: PreparedIndexBatch.from_records(
        [_prepared(record) for record in batch]
    )

    first = await coordinator.run_source(
        source,
        prepare_batch=prepare,
        batch_size=1,
        failure_mode="lenient",
    )

    assert first.failed == 1
    assert first.committed == 2
    assert first.checkpoint == "1"
    assert await checkpoint_store.load("notes") == "1"

    failing_stage.failed_ids.clear()
    second = await coordinator.run_source(
        source,
        prepare_batch=prepare,
        batch_size=1,
        failure_mode="lenient",
    )

    assert [result.source_id for result in second.records] == ["bad", "three"]
    assert second.failed == 0
    assert second.committed == 2
    assert second.checkpoint == "3"
    assert await checkpoint_store.load("notes") == "3"
    assert source.since_values == [None, "1"]
    cache.close()


@pytest.mark.asyncio
async def test_checkpoint_file_preserves_last_value_when_persistence_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = JsonCheckpointStore(tmp_path / "checkpoint.json")
    coordinator, cache, _, _ = _coordinator(
        tmp_path,
        stages=(_NoopStage(),),
        checkpoint_store=store,
    )
    records = [
        _prepared(_record("first", cursor="1")),
        _prepared(_record("second", cursor="2")),
    ]
    original_atomic_write = checkpoints_module.atomic_write_json
    write_calls = 0

    def fail_atomic_write(*args, **kwargs) -> None:
        nonlocal write_calls
        write_calls += 1
        if write_calls == 2:
            raise OSError("disk full")
        original_atomic_write(*args, **kwargs)

    monkeypatch.setattr(checkpoints_module, "atomic_write_json", fail_atomic_write)
    with pytest.raises(OSError, match="disk full"):
        await coordinator.run_prepared_records(
            records,
            source_kind="notes",
            max_records=1,
            checkpoint_for_batch=_checkpoint_for_batch,
        )

    assert await store.load("notes") == "1"
    cache.close()


@pytest.mark.asyncio
async def test_corrupt_embedding_cache_recovers_and_recomputes(
    tmp_path: Path,
) -> None:
    cache_path = tmp_path / "embeddings.db"
    first, cache, first_encoder, _ = _coordinator(
        tmp_path,
        stages=(_NoopStage(),),
    )
    await first.run_prepared_records(
        [_prepared(_record("one"), content="recoverable text")],
        source_kind="notes",
    )
    cache.close()
    cache_path.write_bytes(b"not a sqlite database")

    second, recovered, second_encoder, _ = _coordinator(
        tmp_path,
        stages=(_NoopStage(),),
    )
    await second.run_prepared_records(
        [_prepared(_record("one"), content="recoverable text")],
        source_kind="notes",
    )

    assert first_encoder.calls == [("recoverable text",)]
    assert second_encoder.calls == [("recoverable text",)]
    assert recovered.metrics.invalidations >= 1
    recovered.close()


@pytest.mark.asyncio
async def test_encoder_namespace_change_forces_fresh_semantic_work(
    tmp_path: Path,
) -> None:
    record = _prepared(_record("one"), content="same body")
    first, first_cache, first_encoder, _ = _coordinator(
        tmp_path,
        namespace="model-v1",
        stages=(_NoopStage(),),
    )
    await first.run_prepared_records([record], source_kind="notes")
    first_cache.close()

    second, second_cache, second_encoder, _ = _coordinator(
        tmp_path,
        namespace="model-v2",
        stages=(_NoopStage(),),
    )
    await second.run_prepared_records([record], source_kind="notes")

    assert first_encoder.calls == [("same body",)]
    assert second_encoder.calls == [("same body",)]
    second_cache.close()


@pytest.mark.asyncio
async def test_generic_coordinator_passes_deletion_and_move_records_to_ingestor(
    tmp_path: Path,
) -> None:
    backend = LocalRecordBackend(tmp_path / "records.db")
    ingestor = _LifecycleIngestor(backend)
    old = _record("old", cursor="1")
    moved = _record(
        "new",
        cursor="2",
        metadata={"moved_from": old.storage_key},
    )
    deleted = _record("new", cursor="3", status=RecordStatus.ARCHIVED)
    source = _CursorSource([old, moved, deleted])
    coordinator, cache, _, _ = _coordinator(
        tmp_path,
        record_ingestor=ingestor,
        checkpoint_store=JsonCheckpointStore(tmp_path / "checkpoint.json"),
    )

    receipt = await coordinator.run_source(source, batch_size=1)

    assert receipt.successful == 3
    assert ingestor.seen == ["old", "new", "new"]
    assert backend.hydrate_record(old.storage_key) is None
    assert backend.hydrate_record(moved.storage_key) is None
    cache.close()


def test_readiness_transitions_remain_independent_and_round_trip_durably(
    tmp_path: Path,
) -> None:
    initial = SearchAvailability(
        lexical="complete",
        graph="complete",
        semantic_coarse="backfilling",
        semantic_fine="unavailable",
    )
    coarse_ready = SearchAvailability(
        lexical="complete",
        graph="complete",
        semantic_coarse="complete",
        semantic_fine="backfilling",
    )
    final = SearchAvailability(
        lexical="complete",
        graph="complete",
        semantic_coarse="complete",
        semantic_fine="complete",
    )
    assert initial.can_serve_queries() is True
    assert initial.is_fully_ready() is False
    assert coarse_ready.can_serve_queries() is True
    assert coarse_ready.is_fully_ready() is False
    assert final.is_fully_ready() is True

    checkpoint = BootstrapCheckpoint(
        schema_version="1.0.0",
        generation="generation",
        complete=False,
        targets={"doc.md": BootstrapFileStamp("doc.md", 1, 2)},
        completed={},
        availability=initial,
    )
    save_bootstrap_checkpoint(tmp_path, checkpoint)
    assert publish_bootstrap_availability(tmp_path, coarse_ready) is True
    assert get_bootstrap_availability(tmp_path) == coarse_ready
    loaded = load_bootstrap_checkpoint(tmp_path)
    assert loaded is not None
    assert loaded.to_dict()["availability"] == coarse_ready.to_dict()


@pytest.mark.asyncio
async def test_progress_callback_has_deterministic_bounded_stage_events(
    tmp_path: Path,
) -> None:
    coordinator, cache, _, _ = _coordinator(
        tmp_path,
        stages=(_NoopStage(),),
    )
    events: list[tuple[int, str]] = []
    records = [_prepared(_record(f"doc-{index}")) for index in range(4)]
    receipt = await coordinator.run_prepared_records(
        records,
        source_kind="notes",
        max_records=2,
        progress=lambda event: events.append((event.batch_index, event.stage)),
    )

    assert receipt.attempted == 4
    assert len(events) == 4
    assert events == [
        (0, "noop"),
        (0, "semantic"),
        (1, "noop"),
        (1, "semantic"),
    ]
    cache.close()
