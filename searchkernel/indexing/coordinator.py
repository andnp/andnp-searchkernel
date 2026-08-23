"""Source-agnostic coordination for resumable semantic indexing."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import (
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    Sequence,
)
from dataclasses import dataclass, field
from typing import Any, cast

from searchkernel.domain import Cursor, Record
from searchkernel.indexing.batches import PreparedIndexRecord
from searchkernel.indexing.batches import (
    iter_prepared_index_batches as iter_record_batches,
)
from searchkernel.indexing.runtime_readiness import SearchAvailability
from searchkernel.indexing.semantic import (
    EmbeddingCache,
    EmbeddingEncoder,
    SemanticProgress,
    SemanticWorkPlanner,
    VectorMaterializer,
    semantic_input_for_record,
)
from searchkernel.indexing.stages import (
    IndexStage,
    PreparedIndexBatch,
    StageResult,
)
from searchkernel.ports.content_source import (
    BatchContentSource,
    CheckpointStore,
    ContentSource,
    IngestionError,
    IngestionFailureMode,
    IngestionReceipt,
    RecordIngestionResult,
    RecordIngestor,
    SourceBatch,
)

type BatchPreparer = Callable[
    [Sequence[Record]],
    PreparedIndexBatch | Awaitable[PreparedIndexBatch],
]
type CheckpointForBatch = Callable[
    [PreparedIndexBatch],
    Cursor | Awaitable[Cursor],
]
type ProgressCallback = Callable[
    ["CoordinatorProgress"],
    object | Awaitable[object],
]


@dataclass(frozen=True, slots=True)
class CoordinatorProgress:
    """Progress evidence emitted after one durable indexing stage."""

    source_kind: str
    workspace_id: str | None
    batch_index: int
    stage: str
    checkpoint: Cursor = None
    checkpoint_persisted: bool = False
    stage_result: StageResult | None = None
    semantic: SemanticProgress | None = None
    ingestion: IngestionReceipt | None = None
    availability: SearchAvailability | None = None


@dataclass(frozen=True, slots=True)
class CoordinatorReceipt:
    """Indexing outcome composed from the existing ingestion receipt types."""

    ingestion: IngestionReceipt
    progress: tuple[CoordinatorProgress, ...] = ()
    stage_results: tuple[StageResult, ...] = ()
    semantic_progress: tuple[SemanticProgress, ...] = ()
    availability: SearchAvailability | None = None
    checkpoint_persisted: bool = False

    @property
    def source_kind(self) -> str:
        return self.ingestion.source_kind

    @property
    def workspace_id(self) -> str | None:
        return self.ingestion.workspace_id

    @property
    def checkpoint(self) -> Cursor:
        return self.ingestion.checkpoint

    @property
    def records(self) -> tuple[RecordIngestionResult, ...]:
        return self.ingestion.records

    @property
    def attempted(self) -> int:
        return self.ingestion.attempted

    @property
    def committed(self) -> int:
        return self.ingestion.committed

    @property
    def skipped(self) -> int:
        return self.ingestion.skipped

    @property
    def failed(self) -> int:
        return self.ingestion.failed

    @property
    def successful(self) -> int:
        return self.ingestion.successful

    @property
    def failures(self) -> tuple[RecordIngestionResult, ...]:
        return self.ingestion.failures


@dataclass
class _BatchOutcome:
    records: tuple[RecordIngestionResult, ...]
    stage_results: list[StageResult] = field(default_factory=list)
    semantic: SemanticProgress | None = None
    successful: bool = False


class _BatchStageRunner:
    """Execute one prepared batch without deciding durable replay policy."""

    def __init__(
        self,
        *,
        planner: SemanticWorkPlanner,
        cache: EmbeddingCache,
        encoder: EmbeddingEncoder,
        materializer: VectorMaterializer,
        record_ingestor: RecordIngestor | None,
        stages: Sequence[IndexStage],
    ) -> None:
        self._planner = planner
        self._cache = cache
        self._encoder = encoder
        self._materializer = materializer
        self._record_ingestor = record_ingestor
        self._stages = stages

    async def run(
        self,
        batch: PreparedIndexBatch,
        *,
        records: Sequence[Record],
        source_kind: str,
        workspace_id: str | None,
        batch_index: int,
        checkpoint: Cursor,
        failure_mode: IngestionFailureMode,
        progress: ProgressCallback | None,
        availability: SearchAvailability | None,
        progress_records: list[CoordinatorProgress],
    ) -> _BatchOutcome:
        outcome = _BatchOutcome(records=())

        if self._record_ingestor is not None:
            try:
                ingestion = await self._record_ingestor.index_records(
                    records,
                    checkpoint=checkpoint,
                    failure_mode=failure_mode,
                )
            except asyncio.CancelledError:
                raise
            except Exception as error:
                failed = _failed_receipt(
                    source_kind,
                    workspace_id,
                    checkpoint,
                    records,
                    error,
                )
                if failure_mode == "strict":
                    raise IngestionError(failed) from error
                return _BatchOutcome(
                    records=failed.records,
                    successful=False,
                )

            outcome.records = ingestion.records
            await _report_progress(
                progress,
                CoordinatorProgress(
                    source_kind=source_kind,
                    workspace_id=workspace_id,
                    batch_index=batch_index,
                    stage="records",
                    checkpoint=checkpoint,
                    ingestion=ingestion,
                    availability=availability,
                ),
                progress_records,
            )
            if ingestion.failed:
                if failure_mode == "strict":
                    raise IngestionError(ingestion)
                return outcome
        else:
            outcome.records = tuple(
                RecordIngestionResult(
                    source_kind=record.source_kind,
                    source_id=record.source_id,
                    workspace_id=record.workspace_id,
                    status="committed",
                )
                for record in records
            )
            for stage in self._stages:
                try:
                    result = await asyncio.to_thread(stage.apply, batch)
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    failed = tuple(
                        _failed_result(record, None, str(error))
                        for record in records
                    )
                    if failure_mode == "strict":
                        raise IngestionError(
                            IngestionReceipt(
                                source_kind=source_kind,
                                workspace_id=workspace_id,
                                checkpoint=checkpoint,
                                records=failed,
                            )
                        ) from error
                    return _BatchOutcome(records=failed, successful=False)
                if inspect.isawaitable(result):
                    result = await result
                if not isinstance(result, StageResult):
                    raise TypeError("index stages must return StageResult")
                outcome.stage_results.append(result)
                await _report_progress(
                    progress,
                    CoordinatorProgress(
                        source_kind=source_kind,
                        workspace_id=workspace_id,
                        batch_index=batch_index,
                        stage=result.stage,
                        checkpoint=checkpoint,
                        stage_result=result,
                        availability=availability,
                    ),
                    progress_records,
                )

        try:
            semantic = await self._planner.execute_async(
                batch.semantic_inputs,
                self._cache,
                self._encoder,
                self._materializer,
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            failed = tuple(
                _failed_result(record, None, str(error)) for record in records
            )
            if failure_mode == "strict":
                raise IngestionError(
                    IngestionReceipt(
                        source_kind=source_kind,
                        workspace_id=workspace_id,
                        checkpoint=checkpoint,
                        records=failed,
                    )
                ) from error
            return _BatchOutcome(records=failed, successful=False)

        outcome.semantic = semantic
        await _report_progress(
            progress,
            CoordinatorProgress(
                source_kind=source_kind,
                workspace_id=workspace_id,
                batch_index=batch_index,
                stage="semantic",
                checkpoint=checkpoint,
                semantic=semantic,
                availability=availability,
            ),
            progress_records,
        )
        outcome.successful = True
        return outcome

class ResumableSemanticCoordinator:
    """Coordinate bounded, commit-before-semantic indexing work.

    The coordinator deliberately accepts either an async ``RecordIngestor``
    or explicit lexical/graph stages.  The
    source and preparation functions remain caller-owned so no source-specific
    policy enters the library.
    """

    def __init__(
        self,
        *,
        planner: SemanticWorkPlanner | None = None,
        cache: EmbeddingCache | None = None,
        encoder: EmbeddingEncoder | None = None,
        materializer: VectorMaterializer | None = None,
        semantic_planner: SemanticWorkPlanner | None = None,
        embedding_cache: EmbeddingCache | None = None,
        embedding_encoder: EmbeddingEncoder | None = None,
        vector_materializer: VectorMaterializer | None = None,
        record_ingestor: RecordIngestor | None = None,
        stages: Sequence[IndexStage] = (),
        keyword_stage: IndexStage | None = None,
        graph_stage: IndexStage | None = None,
        checkpoint_store: CheckpointStore | None = None,
    ) -> None:
        self._planner = _select_dependency(
            planner, semantic_planner, "planner", "semantic_planner"
        )
        self._cache = _select_dependency(
            cache, embedding_cache, "cache", "embedding_cache"
        )
        self._encoder = _select_dependency(
            encoder, embedding_encoder, "encoder", "embedding_encoder"
        )
        self._materializer = _select_dependency(
            materializer,
            vector_materializer,
            "materializer",
            "vector_materializer",
        )
        selected_stages = tuple(stages)
        if keyword_stage is not None:
            selected_stages += (keyword_stage,)
        if graph_stage is not None:
            selected_stages += (graph_stage,)
        if record_ingestor is not None and selected_stages:
            raise ValueError(
                "record_ingestor cannot be combined with explicit indexing stages"
            )
        if record_ingestor is None and not selected_stages:
            raise ValueError(
                "configure a record_ingestor or at least one indexing stage"
            )
        self._record_ingestor = record_ingestor
        self._stages = selected_stages
        self._checkpoint_store = checkpoint_store
        self._batch_runner = _BatchStageRunner(
            planner=self._planner,
            cache=self._cache,
            encoder=self._encoder,
            materializer=self._materializer,
            record_ingestor=record_ingestor,
            stages=selected_stages,
        )

    async def run(
        self,
        source: ContentSource | BatchContentSource,
        *,
        prepare_batch: BatchPreparer | None = None,
        since: Cursor | None = None,
        workspace_id: str | None = None,
        batch_size: int = 100,
        failure_mode: IngestionFailureMode = "strict",
        checkpoint_store: CheckpointStore | None = None,
        progress: ProgressCallback | None = None,
        availability: SearchAvailability | None = None,
    ) -> CoordinatorReceipt:
        """Index a source in bounded batches and resume from its checkpoint."""
        return await self.run_source(
            source,
            prepare_batch=prepare_batch,
            since=since,
            workspace_id=workspace_id,
            batch_size=batch_size,
            failure_mode=failure_mode,
            checkpoint_store=checkpoint_store,
            progress=progress,
            availability=availability,
        )

    async def run_source(
        self,
        source: ContentSource | BatchContentSource,
        *,
        prepare_batch: BatchPreparer | None = None,
        since: Cursor | None = None,
        workspace_id: str | None = None,
        batch_size: int = 100,
        failure_mode: IngestionFailureMode = "strict",
        checkpoint_store: CheckpointStore | None = None,
        progress: ProgressCallback | None = None,
        availability: SearchAvailability | None = None,
    ) -> CoordinatorReceipt:
        """Index a source while advancing its cursor only after semantic work."""
        _validate_options(batch_size, failure_mode)
        if self._record_ingestor is None and prepare_batch is None:
            raise ValueError(
                "prepare_batch is required when using explicit indexing stages"
            )

        store = checkpoint_store or self._checkpoint_store
        current_checkpoint = since
        if current_checkpoint is None and store is not None:
            current_checkpoint = await store.load(source.source_kind, workspace_id)

        all_records: list[RecordIngestionResult] = []
        all_progress: list[CoordinatorProgress] = []
        all_stage_results: list[StageResult] = []
        all_semantic_progress: list[SemanticProgress] = []
        checkpoint_blocked = False

        async for batch_index, records, terminal_cursor in _iter_source_batches(
            source,
            since=current_checkpoint,
            batch_size=batch_size,
        ):
            if not records:
                if (
                    terminal_cursor is not None
                    and not checkpoint_blocked
                    and terminal_cursor != current_checkpoint
                ):
                    await _save_checkpoint(
                        store,
                        source.source_kind,
                        workspace_id,
                        terminal_cursor,
                    )
                    current_checkpoint = terminal_cursor
                    await _report_progress(
                        progress,
                        CoordinatorProgress(
                            source_kind=source.source_kind,
                            workspace_id=workspace_id,
                            batch_index=batch_index,
                            stage="checkpoint",
                            checkpoint=current_checkpoint,
                            checkpoint_persisted=store is not None,
                            availability=availability,
                        ),
                        all_progress,
                    )
                continue

            try:
                prepared = await self._prepare(records, prepare_batch)
                outcome = await self._batch_runner.run(
                    prepared,
                    records=records,
                    source_kind=source.source_kind,
                    workspace_id=workspace_id,
                    batch_index=batch_index,
                    checkpoint=current_checkpoint,
                    failure_mode=failure_mode,
                    progress=progress,
                    availability=availability,
                    progress_records=all_progress,
                )
            except asyncio.CancelledError:
                raise
            except IngestionError as error:
                if failure_mode == "strict":
                    raise
                outcome = _BatchOutcome(
                    records=error.receipt.records,
                    successful=False,
                )
            except Exception as error:
                failed_receipt = _failed_receipt(
                    source.source_kind,
                    workspace_id,
                    current_checkpoint,
                    records,
                    error,
                )
                if failure_mode == "strict":
                    raise IngestionError(failed_receipt) from error
                outcome = _BatchOutcome(
                    records=failed_receipt.records,
                    successful=False,
                )

            all_records.extend(outcome.records)
            all_stage_results.extend(outcome.stage_results)
            if outcome.semantic is not None:
                all_semantic_progress.append(outcome.semantic)

            if not outcome.successful:
                checkpoint_blocked = True
                continue

            if (
                terminal_cursor is not None
                and not checkpoint_blocked
                and terminal_cursor != current_checkpoint
            ):
                await _save_checkpoint(
                    store,
                    source.source_kind,
                    workspace_id,
                    terminal_cursor,
                )
                current_checkpoint = terminal_cursor
                await _report_progress(
                    progress,
                    CoordinatorProgress(
                        source_kind=source.source_kind,
                        workspace_id=workspace_id,
                        batch_index=batch_index,
                        stage="checkpoint",
                        checkpoint=current_checkpoint,
                        checkpoint_persisted=store is not None,
                        availability=availability,
                    ),
                    all_progress,
                )

        receipt = IngestionReceipt(
            source_kind=source.source_kind,
            workspace_id=workspace_id or _single_workspace(all_records),
            checkpoint=current_checkpoint,
            records=tuple(all_records),
        )
        return CoordinatorReceipt(
            ingestion=receipt,
            progress=tuple(all_progress),
            stage_results=tuple(all_stage_results),
            semantic_progress=tuple(all_semantic_progress),
            availability=availability,
            checkpoint_persisted=store is not None,
        )

    async def run_batches(
        self,
        batches: Iterable[PreparedIndexBatch] | AsyncIterable[PreparedIndexBatch],
        *,
        source_kind: str,
        workspace_id: str | None = None,
        since: Cursor | None = None,
        checkpoint_for_batch: CheckpointForBatch | None = None,
        checkpoint_store: CheckpointStore | None = None,
        failure_mode: IngestionFailureMode = "strict",
        progress: ProgressCallback | None = None,
        availability: SearchAvailability | None = None,
        max_records: int | None = None,
        max_chunks: int | None = None,
    ) -> CoordinatorReceipt:
        """Process caller-prepared batches with optional durable cursors."""
        _validate_options(1, failure_mode)
        if not source_kind:
            raise ValueError("source_kind must not be empty")

        store = checkpoint_store or self._checkpoint_store
        current_checkpoint = since
        if current_checkpoint is None and store is not None:
            current_checkpoint = await store.load(source_kind, workspace_id)

        all_records: list[RecordIngestionResult] = []
        all_progress: list[CoordinatorProgress] = []
        all_stage_results: list[StageResult] = []
        all_semantic_progress: list[SemanticProgress] = []
        checkpoint_blocked = False

        async for batch_index, batch in _as_async_batches(batches):
            if max_records is not None and len(batch.records) > max_records:
                raise ValueError("prepared batch exceeds max_records")
            if max_chunks is not None and len(batch.chunks) > max_chunks:
                raise ValueError("prepared batch exceeds max_chunks")

            records = tuple(prepared.record for prepared in batch.records)
            try:
                outcome = await self._batch_runner.run(
                    batch,
                    records=records,
                    source_kind=source_kind,
                    workspace_id=workspace_id,
                    batch_index=batch_index,
                    checkpoint=current_checkpoint,
                    failure_mode=failure_mode,
                    progress=progress,
                    availability=availability,
                    progress_records=all_progress,
                )
            except asyncio.CancelledError:
                raise
            except IngestionError:
                if failure_mode == "strict":
                    raise
                outcome = _BatchOutcome(
                    records=tuple(
                        _failed_result(record, None, "batch ingestion failed")
                        for record in records
                    ),
                    successful=False,
                )
            except Exception as error:
                failed_receipt = _failed_receipt(
                    source_kind,
                    workspace_id,
                    current_checkpoint,
                    records,
                    error,
                )
                if failure_mode == "strict":
                    raise IngestionError(failed_receipt) from error
                outcome = _BatchOutcome(
                    records=failed_receipt.records,
                    successful=False,
                )

            all_records.extend(outcome.records)
            all_stage_results.extend(outcome.stage_results)
            if outcome.semantic is not None:
                all_semantic_progress.append(outcome.semantic)

            if not outcome.successful:
                checkpoint_blocked = True
                continue

            if checkpoint_for_batch is not None and not checkpoint_blocked:
                if inspect.iscoroutinefunction(checkpoint_for_batch):
                    candidate = checkpoint_for_batch(batch)
                else:
                    candidate = await asyncio.to_thread(
                        checkpoint_for_batch, batch
                    )
                if inspect.isawaitable(candidate):
                    candidate = await candidate
                if candidate != current_checkpoint:
                    await _save_checkpoint(
                        store,
                        source_kind,
                        workspace_id,
                        candidate,
                    )
                    current_checkpoint = candidate
                    await _report_progress(
                        progress,
                        CoordinatorProgress(
                            source_kind=source_kind,
                            workspace_id=workspace_id,
                            batch_index=batch_index,
                            stage="checkpoint",
                            checkpoint=current_checkpoint,
                            checkpoint_persisted=store is not None,
                            availability=availability,
                        ),
                        all_progress,
                    )

        receipt = IngestionReceipt(
            source_kind=source_kind,
            workspace_id=workspace_id or _single_workspace(all_records),
            checkpoint=current_checkpoint,
            records=tuple(all_records),
        )
        return CoordinatorReceipt(
            ingestion=receipt,
            progress=tuple(all_progress),
            stage_results=tuple(all_stage_results),
            semantic_progress=tuple(all_semantic_progress),
            availability=availability,
            checkpoint_persisted=store is not None,
        )

    async def run_prepared_records(
        self,
        records: Iterable[PreparedIndexRecord],
        *,
        source_kind: str,
        workspace_id: str | None = None,
        since: Cursor | None = None,
        max_records: int = 100,
        max_chunks: int = 1000,
        checkpoint_for_batch: CheckpointForBatch | None = None,
        checkpoint_store: CheckpointStore | None = None,
        failure_mode: IngestionFailureMode = "strict",
        progress: ProgressCallback | None = None,
        availability: SearchAvailability | None = None,
    ) -> CoordinatorReceipt:
        """Bound prepared records with the existing stage batch helper."""
        record_batches = iter_record_batches(
            records,
            max_records=max_records,
            max_chunks=max_chunks,
        )
        return await self.run_batches(
            (
                PreparedIndexBatch.from_records(record_batch)
                for record_batch in record_batches
            ),
            source_kind=source_kind,
            workspace_id=workspace_id,
            since=since,
            checkpoint_for_batch=checkpoint_for_batch,
            checkpoint_store=checkpoint_store,
            failure_mode=failure_mode,
            progress=progress,
            availability=availability,
            max_records=max_records,
            max_chunks=max_chunks,
        )

    async def _prepare(
        self,
        records: Sequence[Record],
        prepare_batch: BatchPreparer | None,
    ) -> PreparedIndexBatch:
        if prepare_batch is None:
            return PreparedIndexBatch(
                records=[],
                semantic_inputs=[
                    semantic_input_for_record(
                        record,
                        self._planner.encoder_namespace,
                    )
                    for record in records
                ],
            )

        if inspect.iscoroutinefunction(prepare_batch):
            prepared = prepare_batch(records)
        else:
            prepared = await asyncio.to_thread(prepare_batch, records)
        if inspect.isawaitable(prepared):
            prepared = await prepared
        if not isinstance(prepared, PreparedIndexBatch):
            raise TypeError("prepare_batch must return PreparedIndexBatch")
        return prepared

async def _report_progress(
    callback: ProgressCallback | None,
    event: CoordinatorProgress,
    progress_records: list[CoordinatorProgress],
) -> None:
    progress_records.append(event)
    if callback is None:
        return
    if inspect.iscoroutinefunction(callback):
        await callback(event)
        return
    result = await asyncio.to_thread(callback, event)
    if inspect.isawaitable(result):
        await result


async def _iter_source_batches(
    source: ContentSource | BatchContentSource,
    *,
    since: Cursor,
    batch_size: int,
) -> AsyncIterator[tuple[int, tuple[Record, ...], Cursor]]:
    batch_iterator = getattr(source, "iter_batches", None)
    if callable(batch_iterator):
        stream = batch_iterator(since=since)
        if inspect.isawaitable(stream):
            stream = await stream
        if not hasattr(stream, "__aiter__"):
            raise TypeError(
                "BatchContentSource.iter_batches must return an async iterator"
            )
        index = 0
        async for source_batch in cast(AsyncIterator[SourceBatch], stream):
            if not isinstance(source_batch, SourceBatch):
                raise TypeError("source yielded an invalid SourceBatch")
            records = tuple(source_batch.records)
            _validate_records(records)
            for offset in range(0, len(records), batch_size) or (0,):
                bounded = records[offset : offset + batch_size]
                terminal = (
                    source_batch.terminal_cursor
                    if offset + len(bounded) == len(records)
                    else None
                )
                yield index, bounded, terminal
                index += 1
        return

    content_source = cast(ContentSource, source)
    stream = content_source.iter_records(since=since)
    if inspect.isawaitable(stream):
        stream = await stream
    if not hasattr(stream, "__aiter__"):
        raise TypeError("ContentSource.iter_records must return an async iterator")

    index = 0
    batch: list[Record] = []
    async for record in cast(AsyncIterator[Record], stream):
        if not isinstance(record, Record):
            raise TypeError("ContentSource yielded a non-Record value")
        batch.append(record)
        if len(batch) < batch_size:
            continue
        bounded = tuple(batch)
        yield index, bounded, content_source.cursor_for(bounded[-1])
        index += 1
        batch = []
    if batch:
        bounded = tuple(batch)
        yield index, bounded, content_source.cursor_for(bounded[-1])


async def _as_async_batches(
    batches: Iterable[PreparedIndexBatch] | AsyncIterable[PreparedIndexBatch],
) -> AsyncIterator[tuple[int, PreparedIndexBatch]]:
    if hasattr(batches, "__aiter__"):
        index = 0
        async for batch in cast(AsyncIterable[PreparedIndexBatch], batches):
            if not isinstance(batch, PreparedIndexBatch):
                raise TypeError("batches must contain PreparedIndexBatch values")
            yield index, batch
            index += 1
        return

    for index, batch in enumerate(cast(Iterable[PreparedIndexBatch], batches)):
        if not isinstance(batch, PreparedIndexBatch):
            raise TypeError("batches must contain PreparedIndexBatch values")
        yield index, batch


async def _save_checkpoint(
    store: CheckpointStore | None,
    source_kind: str,
    workspace_id: str | None,
    checkpoint: Cursor,
) -> None:
    if store is not None:
        await store.save(source_kind, workspace_id, checkpoint)


def _validate_options(
    batch_size: int,
    failure_mode: IngestionFailureMode,
) -> None:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if failure_mode not in {"strict", "lenient"}:
        raise ValueError("failure_mode must be 'strict' or 'lenient'")


def _validate_records(records: Sequence[Record]) -> None:
    for record in records:
        if not isinstance(record, Record):
            raise TypeError("source yielded a non-Record value")


def _failed_receipt(
    source_kind: str,
    workspace_id: str | None,
    checkpoint: Cursor,
    records: Sequence[Record],
    error: Exception,
) -> IngestionReceipt:
    return IngestionReceipt(
        source_kind=source_kind,
        workspace_id=workspace_id or _single_workspace(records),
        checkpoint=checkpoint,
        records=tuple(
            _failed_result(record, None, f"{type(error).__name__}: {error}")
            for record in records
        ),
    )


def _failed_result(
    record: Record,
    cursor: Cursor,
    error: str | None,
) -> RecordIngestionResult:
    return RecordIngestionResult(
        source_kind=record.source_kind,
        source_id=record.source_id,
        workspace_id=record.workspace_id,
        status="failed",
        cursor=cursor,
        error=error,
    )


def _single_workspace(
    records: Sequence[RecordIngestionResult] | Sequence[Record],
) -> str | None:
    values = {
        record.workspace_id
        for record in records
    }
    return next(iter(values)) if len(values) == 1 else None


def _select_dependency(
    primary: Any,
    alias: Any,
    primary_name: str,
    alias_name: str,
) -> Any:
    if primary is not None and alias is not None and primary is not alias:
        raise ValueError(f"{primary_name} and {alias_name} disagree")
    selected = primary if primary is not None else alias
    if selected is None:
        raise TypeError(f"{primary_name} is required")
    return selected


ProgressiveIndexCoordinator = ResumableSemanticCoordinator
SemanticIndexCoordinator = ResumableSemanticCoordinator

__all__ = [
    "BatchPreparer",
    "CheckpointForBatch",
    "CoordinatorProgress",
    "CoordinatorReceipt",
    "ProgressCallback",
    "ProgressiveIndexCoordinator",
    "ResumableSemanticCoordinator",
    "SemanticIndexCoordinator",
]
