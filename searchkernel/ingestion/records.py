"""Source-agnostic record ingestion over the kernel's storage ports."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Protocol

from searchkernel.domain import Chunk, Cursor, Record
from searchkernel.indexing.semantic import (
    EmbeddingCache,
    SemanticInput,
    SemanticWorkPlanner,
    semantic_input_for_record,
)
from searchkernel.ports.chunking import RecordChunker
from searchkernel.ports.content_source import (
    IngestionFailureMode,
    IngestionReceipt,
    RecordIngestionResult,
)
from searchkernel.ports.embedding import EmbeddingProvider
from searchkernel.ports.stores import KeywordStore, VectorStore


class _EmbeddingEncoder(Protocol):
    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        ...


class _ProviderEncoder:
    def __init__(self, provider: EmbeddingProvider) -> None:
        self._provider = provider

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return self._provider.embed(list(texts))


class _RecordMaterializer:
    def __init__(self, record: Record, model_name: str) -> None:
        self._record = record
        self._model_name = model_name

    def materialize(
        self,
        source_id: str,
        vector: Sequence[float],
        semantic_input: SemanticInput,
    ) -> None:
        if source_id != self._record.storage_key:
            raise ValueError("semantic input does not belong to the record")
        self._record.embedding = list(vector)
        self._record.embedding_model = self._model_name


class _RecordBatchMaterializer:
    def __init__(self, records: Sequence[Record], model_name: str) -> None:
        self._records = {record.storage_key: record for record in records}
        self._model_name = model_name

    def materialize(
        self,
        source_id: str,
        vector: Sequence[float],
        semantic_input: SemanticInput,
    ) -> None:
        record = self._records.get(source_id)
        if record is None:
            raise ValueError("semantic input does not belong to the record batch")
        record.embedding = list(vector)
        record.embedding_model = self._model_name


class SemanticRecordIngestor:
    """Index source-agnostic records through keyword and vector stores."""

    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider,
        keyword_store: KeywordStore,
        vector_store: VectorStore,
        embedding_cache: EmbeddingCache,
        chunker: RecordChunker | None = None,
        encoder_namespace: str | None = None,
        embedding_batch_size: int = 32,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.keyword_store = keyword_store
        self.vector_store = vector_store
        self.embedding_cache = embedding_cache
        self.chunker = chunker
        self.encoder_namespace = (
            encoder_namespace
            or getattr(embedding_cache, "encoder_namespace", None)
            or embedding_provider.model_name
        )
        self._planner = SemanticWorkPlanner(
            self.encoder_namespace,
            dimension=embedding_provider.dim,
        )
        if embedding_batch_size < 1:
            raise ValueError("embedding_batch_size must be >= 1")
        self.embedding_batch_size = embedding_batch_size
        self._encoder: _EmbeddingEncoder = _ProviderEncoder(embedding_provider)

    async def index_records(
        self,
        records: Sequence[Record],
        *,
        checkpoint: Cursor | None = None,
        failure_mode: IngestionFailureMode = "strict",
    ) -> IngestionReceipt:
        """Index a batch while leaving checkpoint persistence to the caller."""
        if failure_mode not in {"strict", "lenient"}:
            raise ValueError("failure_mode must be 'strict' or 'lenient'")

        source_records = tuple(records)
        if not source_records:
            return IngestionReceipt("", None, checkpoint, ())
        records = _expand_records(source_records, self.chunker)

        keyword_task = asyncio.create_task(
            self._index_keyword_stage(records, failure_mode=failure_mode)
        )
        semantic_task = asyncio.create_task(
            self._index_semantic_stage(records, failure_mode=failure_mode)
        )
        try:
            keyword_outcomes, semantic_outcomes = await asyncio.gather(
                keyword_task,
                semantic_task,
            )
        except asyncio.CancelledError:
            keyword_task.cancel()
            semantic_task.cancel()
            await asyncio.gather(
                keyword_task,
                semantic_task,
                return_exceptions=True,
            )
            raise

        return _merge_stage_outcomes(
            source_records,
            checkpoint=checkpoint,
            failure_mode=failure_mode,
            keyword_outcomes=_collapse_outcomes(
                source_records, records, keyword_outcomes
            ),
            semantic_outcomes=_collapse_outcomes(
                source_records, records, semantic_outcomes
            ),
        )

    async def _index_keyword_stage(
        self,
        records: Sequence[Record],
        *,
        failure_mode: IngestionFailureMode,
    ) -> list[RecordIngestionResult]:
        # Local stores can commit a whole batch in one SQLite transaction. If
        # a backend rejects the batch, fall back to per-record writes so the
        # receipt still reports precise failures.
        try:
            await asyncio.to_thread(self.keyword_store.index, list(records))
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - preserve per-record attribution
            return await self._index_keyword_records_individually(
                records,
                failure_mode=failure_mode,
            )

        return [
            RecordIngestionResult(
                source_kind=record.source_kind,
                source_id=record.source_id,
                workspace_id=record.workspace_id,
                status="committed",
            )
            for record in records
        ]

    async def _index_keyword_records_individually(
        self,
        records: Sequence[Record],
        *,
        failure_mode: IngestionFailureMode,
    ) -> list[RecordIngestionResult]:
        outcomes: list[RecordIngestionResult] = []
        for record in records:
            try:
                await asyncio.to_thread(self.keyword_store.index, [record])
            except asyncio.CancelledError:
                raise
            except Exception as error:  # noqa: BLE001
                outcomes.append(
                    RecordIngestionResult(
                        source_kind=record.source_kind,
                        source_id=record.source_id,
                        workspace_id=record.workspace_id,
                        status="failed",
                        error=f"{type(error).__name__}: {error}",
                    )
                )
                if failure_mode == "strict":
                    break
            else:
                outcomes.append(
                    RecordIngestionResult(
                        source_kind=record.source_kind,
                        source_id=record.source_id,
                        workspace_id=record.workspace_id,
                        status="committed",
                    )
                )

        return outcomes

    async def _index_semantic_stage(
        self,
        records: Sequence[Record],
        *,
        failure_mode: IngestionFailureMode,
    ) -> list[RecordIngestionResult]:
        try:
            await asyncio.to_thread(self._index_semantic_batch, records)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            # Preserve per-record failure attribution if batch embedding fails.
            return await self._index_semantic_records_individually(
                records,
                failure_mode=failure_mode,
            )

        return await self._index_vector_records(
            records,
            failure_mode=failure_mode,
        )

    async def _index_semantic_records_individually(
        self,
        records: Sequence[Record],
        *,
        failure_mode: IngestionFailureMode,
    ) -> list[RecordIngestionResult]:
        outcomes: list[RecordIngestionResult] = []
        for record in records:
            try:
                await asyncio.to_thread(self._index_record_semantic_and_vector, record)
            except asyncio.CancelledError:
                raise
            except Exception as error:  # noqa: BLE001
                outcomes.append(
                    RecordIngestionResult(
                        source_kind=record.source_kind,
                        source_id=record.source_id,
                        workspace_id=record.workspace_id,
                        status="failed",
                        error=f"{type(error).__name__}: {error}",
                    )
                )
                if failure_mode == "strict":
                    break
            else:
                outcomes.append(
                    RecordIngestionResult(
                        source_kind=record.source_kind,
                        source_id=record.source_id,
                        workspace_id=record.workspace_id,
                        status="committed",
                    )
                )

        return outcomes

    async def _index_vector_records(
        self,
        records: Sequence[Record],
        *,
        failure_mode: IngestionFailureMode,
    ) -> list[RecordIngestionResult]:
        try:
            await asyncio.to_thread(
                self.vector_store.upsert,
                list(records),
                self.embedding_provider.model_name,
                self.embedding_provider.dim,
            )
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - preserve per-record attribution
            return await self._index_vector_records_individually(
                records,
                failure_mode=failure_mode,
            )

        return [
            RecordIngestionResult(
                source_kind=record.source_kind,
                source_id=record.source_id,
                workspace_id=record.workspace_id,
                status="committed",
            )
            for record in records
        ]

    async def _index_vector_records_individually(
        self,
        records: Sequence[Record],
        *,
        failure_mode: IngestionFailureMode,
    ) -> list[RecordIngestionResult]:
        outcomes: list[RecordIngestionResult] = []
        for record in records:
            try:
                await asyncio.to_thread(self._index_record_vector, record)
            except asyncio.CancelledError:
                raise
            except Exception as error:  # noqa: BLE001
                outcomes.append(
                    RecordIngestionResult(
                        source_kind=record.source_kind,
                        source_id=record.source_id,
                        workspace_id=record.workspace_id,
                        status="failed",
                        error=f"{type(error).__name__}: {error}",
                    )
                )
                if failure_mode == "strict":
                    break
            else:
                outcomes.append(
                    RecordIngestionResult(
                        source_kind=record.source_kind,
                        source_id=record.source_id,
                        workspace_id=record.workspace_id,
                        status="committed",
                    )
                )

        return outcomes

    def _index_semantic_batch(self, records: Sequence[Record]) -> None:
        inputs = [
            semantic_input_for_record(record, self.encoder_namespace)
            for record in records
        ]
        self._planner.execute(
            inputs,
            self.embedding_cache,
            self._encoder,
            _RecordBatchMaterializer(
                records,
                self.embedding_provider.model_name,
            ),
            batch_size=self.embedding_batch_size,
        )

    def _index_record_semantic(self, record: Record) -> None:
        semantic_input = semantic_input_for_record(
            record,
            self.encoder_namespace,
        )
        self._planner.execute(
            [semantic_input],
            self.embedding_cache,
            self._encoder,
            _RecordMaterializer(record, self.embedding_provider.model_name),
            batch_size=self.embedding_batch_size,
        )
 
    def _index_record_semantic_and_vector(self, record: Record) -> None:
        self._index_record_semantic(record)
        self._index_record_vector(record)

    def _index_record_vector(self, record: Record) -> None:
        self.vector_store.upsert(
            [record],
            self.embedding_provider.model_name,
            self.embedding_provider.dim,
        )


def _receipt(
    records: Sequence[Record],
    checkpoint: Cursor | None,
    outcomes: list[RecordIngestionResult],
) -> IngestionReceipt:
    return IngestionReceipt(
        source_kind=records[0].source_kind if records else "",
        workspace_id=_workspace_id(records),
        checkpoint=checkpoint,
        records=tuple(outcomes),
    )


def _merge_stage_outcomes(
    records: Sequence[Record],
    *,
    checkpoint: Cursor | None,
    failure_mode: IngestionFailureMode,
    keyword_outcomes: Sequence[RecordIngestionResult],
    semantic_outcomes: Sequence[RecordIngestionResult],
) -> IngestionReceipt:
    keyword_by_id = {_result_key(result): result for result in keyword_outcomes}
    semantic_by_id = {_result_key(result): result for result in semantic_outcomes}
    outcomes: list[RecordIngestionResult] = []
    for record in records:
        record_key = _result_key(record)
        stage_results = (
            keyword_by_id.get(record_key),
            semantic_by_id.get(record_key),
        )
        errors: list[str] = []
        for stage_name, result in zip(
            ("keyword", "semantic"), stage_results, strict=True
        ):
            if result is None:
                errors.append(f"{stage_name} stage did not report an outcome")
            elif not result.successful:
                detail = result.error or f"status={result.status}"
                errors.append(f"{stage_name} stage: {detail}")
        if errors:
            outcomes.append(
                RecordIngestionResult(
                    source_kind=record.source_kind,
                    source_id=record.source_id,
                    workspace_id=record.workspace_id,
                    status="failed",
                    error="; ".join(errors),
                )
            )
            if failure_mode == "strict":
                break
        else:
            outcomes.append(
                RecordIngestionResult(
                    source_kind=record.source_kind,
                    source_id=record.source_id,
                    workspace_id=record.workspace_id,
                    status="committed",
                )
            )
    return _receipt(records, checkpoint, outcomes)


def _result_key(
    result: Record | RecordIngestionResult,
) -> tuple[str, str, str | None]:
    return result.source_kind, result.source_id, result.workspace_id


def _workspace_id(records: Sequence[Record]) -> str | None:
    values = {record.workspace_id for record in records}
    if len(values) == 1:
        return next(iter(values))
    return None


def _expand_records(
    records: Sequence[Record],
    chunker: RecordChunker | None,
) -> tuple[Record, ...]:
    if chunker is None:
        return tuple(records)
    expanded: list[Record] = []
    for record in records:
        expanded.append(record)
        expanded.extend(_chunk_record(record, chunk) for chunk in chunker.chunk_record(record))
    return tuple(expanded)


def _chunk_record(record: Record, chunk: Chunk) -> Record:
    metadata = dict(record.metadata)
    metadata.update(
        {
            "_searchkernel_chunk": True,
            "_chunk_id": chunk.chunk_id,
            "_chunk_index": chunk.chunk_index,
            "_chunk_parent_storage_key": record.storage_key,
            "_chunk_metadata": dict(chunk.metadata),
        }
    )
    return Record(
        workspace_id=record.workspace_id,
        source_kind=record.source_kind,
        source_id=f"{record.storage_key}#chunk:{chunk.chunk_id}",
        title=record.title,
        body=chunk.content,
        indexed_text=chunk.content,
        created_at=record.created_at,
        updated_at=record.updated_at,
        metadata=metadata,
        uri=record.uri,
        status=record.status,
    )


def _collapse_outcomes(
    source_records: Sequence[Record],
    indexed_records: Sequence[Record],
    outcomes: Sequence[RecordIngestionResult],
) -> tuple[RecordIngestionResult, ...]:
    by_parent: dict[str, list[RecordIngestionResult]] = {
        record.storage_key: [] for record in source_records
    }
    for record, outcome in zip(indexed_records, outcomes, strict=False):
        parent_key = record.metadata.get("_chunk_parent_storage_key", record.storage_key)
        by_parent.setdefault(parent_key, []).append(outcome)
    collapsed: list[RecordIngestionResult] = []
    for record in source_records:
        related = by_parent[record.storage_key]
        failed = next((result for result in related if not result.successful), None)
        if not related:
            failed = RecordIngestionResult(
                source_kind=record.source_kind,
                source_id=record.source_id,
                workspace_id=record.workspace_id,
                status="failed",
                error="chunk stage did not report an outcome",
            )
        collapsed.append(
            RecordIngestionResult(
                source_kind=record.source_kind,
                source_id=record.source_id,
                workspace_id=record.workspace_id,
                status="failed" if failed else "committed",
                error=failed.error if failed else None,
            )
        )
    return tuple(collapsed)


__all__ = ["SemanticRecordIngestor"]
