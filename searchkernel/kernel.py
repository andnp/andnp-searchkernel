"""Composable search-kernel composition root and driving facade."""

import asyncio
import inspect
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    Mapping,
    Sequence,
)
from typing import Any, cast

from searchkernel.domain import (
    Cursor,
    Record,
    RecordIdentity,
    Vector,
)
from searchkernel.ports.content_source import (
    CheckpointStore,
    ContentSource,
    IngestionError,
    IngestionFailureMode,
    IngestionReceipt,
    RecordIngestionResult,
    RecordIngestor,
    SourceBatch,
)
from searchkernel.ports.embedding import (
    AsyncEmbeddingProvider,
    EmbeddingProvider,
)
from searchkernel.ports.rerank import Reranker
from searchkernel.ports.stores import (
    AsyncGraphStore,
    AsyncKeywordStore,
    AsyncVectorStore,
    GraphStore,
    KeywordStore,
    VectorStore,
)
from searchkernel.search.orchestrator import SearchOrchestrator
from searchkernel.search.record_pipeline import (
    QueryEmbeddingProvider,
    RecordHydrator,
    RecordSearchConfig,
    RecordSearchOutcome,
    RecordSearchPolicy,
)


class SearchKernel:
    """Daemon-free, source-agnostic search composition root."""

    def __init__(
        self,
        *,
        orchestrator: SearchOrchestrator | None = None,
        ingestor: RecordIngestor | None = None,
        content_sources: Iterable[ContentSource] = (),
        config: object | None = None,
        embedder: EmbeddingProvider | None = None,
    ) -> None:
        self._orchestrator = orchestrator
        self._ingestor = ingestor
        self._content_sources: dict[str, ContentSource] = {}
        for source in content_sources:
            self.register_content_source(source)
        self._config = config
        self._embedder = embedder

    @classmethod
    def build(
        cls,
        config: object | None = None,
        *,
        content_sources: Iterable[ContentSource] = (),
        ingestor: RecordIngestor | None = None,
        embedder: EmbeddingProvider | None = None,
        reranker: Reranker | None = None,
        orchestrator: SearchOrchestrator | None = None,
        record_hydrator: (
            RecordHydrator
            | Callable[
                [RecordIdentity | str],
                Record | None | Awaitable[Record | None],
            ]
            | None
        ) = None,
        keyword_store: KeywordStore | AsyncKeywordStore | None = None,
        vector_store: VectorStore | AsyncVectorStore | None = None,
        graph_store: GraphStore | AsyncGraphStore | None = None,
        embedding_provider: (
            EmbeddingProvider
            | AsyncEmbeddingProvider
            | QueryEmbeddingProvider
            | Callable[[str], Vector | Awaitable[Vector]]
            | None
        ) = None,
        embedding_model_name: str | None = None,
        embedding_dim: int | None = None,
        search_policy: RecordSearchPolicy | None = None,
        search_config: RecordSearchConfig | None = None,
    ) -> "SearchKernel":
        """Compose a kernel from source adapters and provider instances.

        ``config`` and ``embedder`` are retained as composition dependencies for
        ingestion and administration. A reranker may be supplied by
        ``config.reranker``.

        Pass the record stores and ``record_hydrator`` to compose the canonical
        local record pipeline. ``orchestrator`` remains available for callers
        that already own a canonical ``SearchOrchestrator``.
        """
        effective_reranker = reranker
        if effective_reranker is None:
            if isinstance(config, Mapping):
                effective_reranker = config.get("reranker")
            elif config is not None:
                effective_reranker = getattr(config, "reranker", None)
        record_dependencies = (
            record_hydrator,
            keyword_store,
            vector_store,
            graph_store,
            embedding_provider,
            embedding_model_name,
            embedding_dim,
            search_policy,
            search_config,
        )
        if orchestrator is not None and any(
            dependency is not None for dependency in record_dependencies
        ):
            raise ValueError(
                "orchestrator cannot be combined with record search dependencies"
            )
        if orchestrator is None and any(
            dependency is not None for dependency in record_dependencies[1:]
        ):
            if record_hydrator is None:
                raise ValueError(
                    "record_hydrator is required when composing record stores"
                )
            orchestrator = SearchOrchestrator(
                hydrator=record_hydrator,
                keyword_store=keyword_store,
                vector_store=vector_store,
                graph_store=graph_store,
                embedding_provider=embedding_provider,
                embedding_model_name=embedding_model_name,
                embedding_dim=embedding_dim,
                reranker=effective_reranker,
                policy=search_policy,
                config=search_config,
            )
        elif orchestrator is None and record_hydrator is not None:
            orchestrator = SearchOrchestrator(
                hydrator=record_hydrator,
                reranker=effective_reranker,
                policy=search_policy,
                config=search_config,
            )
        effective_embedder = embedder
        if effective_embedder is None:
            if isinstance(config, Mapping):
                effective_embedder = config.get("embedder")
            elif config is not None:
                effective_embedder = getattr(config, "embedder", None)

        return cls(
            orchestrator=orchestrator,
            ingestor=ingestor,
            content_sources=content_sources,
            config=config,
            embedder=effective_embedder,
        )

    @property
    def config(self) -> object | None:
        return self._config

    @property
    def embedder(self) -> EmbeddingProvider | None:
        return self._embedder

    def register_content_source(self, source: ContentSource) -> None:
        """Register an ingestible source without affecting searchable sources."""
        self._content_sources[source.source_kind] = source

    async def ingest_source(
        self,
        source_kind: str,
        since: Cursor | None = None,
        *,
        workspace_id: str | None = None,
        batch_size: int = 100,
        failure_mode: IngestionFailureMode = "strict",
        checkpoint_store: CheckpointStore | None = None,
    ) -> IngestionReceipt:
        """Ingest a source in async batches with commit-then-checkpoint ordering."""
        source = self._content_sources.get(source_kind)
        if source is None:
            raise KeyError(f"No content source registered for {source_kind!r}")
        if self._ingestor is None:
            raise RuntimeError(
                "Cannot ingest content: no record ingestor was wired into SearchKernel"
            )
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if failure_mode not in {"strict", "lenient"}:
            raise ValueError("failure_mode must be 'strict' or 'lenient'")

        current_checkpoint = since
        if current_checkpoint is None and checkpoint_store is not None:
            current_checkpoint = await checkpoint_store.load(
                source_kind, workspace_id
            )

        all_results: list[RecordIngestionResult] = []
        checkpoint_blocked = False
        batch_iterator = getattr(source, "iter_batches", None)
        if callable(batch_iterator):
            stream = batch_iterator(since=current_checkpoint)
            if inspect.isawaitable(stream):
                stream = await stream
            if not hasattr(stream, "__aiter__"):
                raise TypeError(
                    "BatchContentSource.iter_batches must return an async iterator"
                )
            async for source_batch in cast(AsyncIterator[SourceBatch], stream):
                if not isinstance(source_batch, SourceBatch):
                    raise TypeError("BatchContentSource yielded an invalid SourceBatch")
                records = tuple(source_batch.records)
                for record in records:
                    if not isinstance(record, Record):
                        raise TypeError("SourceBatch contained a non-Record value")
                current_checkpoint, batch_results, checkpoint_blocked = (
                    await self._ingest_batch(
                        source,
                        records,
                        current_checkpoint=current_checkpoint,
                        checkpoint_blocked=checkpoint_blocked,
                        workspace_id=workspace_id,
                        failure_mode=failure_mode,
                        checkpoint_store=checkpoint_store,
                        terminal_cursor=source_batch.terminal_cursor,
                    )
                )
                all_results.extend(batch_results)
            return IngestionReceipt(
                source_kind=source_kind,
                workspace_id=workspace_id or _single_workspace(all_results),
                checkpoint=current_checkpoint,
                records=tuple(all_results),
            )

        stream = source.iter_records(since=current_checkpoint)
        if inspect.isawaitable(stream):
            stream = await stream
        if not hasattr(stream, "__aiter__"):
            raise TypeError(
                "ContentSource.iter_records must return an async iterator"
            )

        batch: list[Record] = []
        async for record in stream:
            if not isinstance(record, Record):
                raise TypeError("ContentSource yielded a non-Record value")
            batch.append(record)
            if len(batch) >= batch_size:
                current_checkpoint, batch_results, checkpoint_blocked = (
                    await self._ingest_batch(
                    source,
                    batch,
                    current_checkpoint=current_checkpoint,
                    checkpoint_blocked=checkpoint_blocked,
                    workspace_id=workspace_id,
                    failure_mode=failure_mode,
                    checkpoint_store=checkpoint_store,
                    )
                )
                all_results.extend(batch_results)
                batch = []

        if batch:
            current_checkpoint, batch_results, checkpoint_blocked = (
                await self._ingest_batch(
                source,
                batch,
                current_checkpoint=current_checkpoint,
                checkpoint_blocked=checkpoint_blocked,
                workspace_id=workspace_id,
                failure_mode=failure_mode,
                checkpoint_store=checkpoint_store,
                )
            )
            all_results.extend(batch_results)

        return IngestionReceipt(
            source_kind=source_kind,
            workspace_id=workspace_id or _single_workspace(all_results),
            checkpoint=current_checkpoint,
            records=tuple(all_results),
        )

    async def _ingest_batch(
        self,
        source: ContentSource,
        records: Sequence[Record],
        *,
        current_checkpoint: Cursor,
        checkpoint_blocked: bool,
        workspace_id: str | None,
        failure_mode: IngestionFailureMode,
        checkpoint_store: CheckpointStore | None,
        terminal_cursor: Cursor = None,
    ) -> tuple[Cursor, list[RecordIngestionResult], bool]:
        if not records:
            candidate = (
                current_checkpoint
                if checkpoint_blocked or terminal_cursor is None
                else terminal_cursor
            )
            if candidate != current_checkpoint and checkpoint_store is not None:
                await checkpoint_store.save(
                    source.source_kind,
                    workspace_id,
                    candidate,
                )
            return candidate, [], checkpoint_blocked

        try:
            ingestor = self._ingestor
            if ingestor is None:
                raise RuntimeError("record ingestor was not configured")
            receipt = await ingestor.index_records(
                records,
                checkpoint=current_checkpoint,
                failure_mode=failure_mode,
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            results = [
                RecordIngestionResult(
                    source_kind=record.source_kind,
                    source_id=record.source_id,
                    workspace_id=record.workspace_id,
                    status="failed",
                    cursor=_record_cursor(source, record),
                    error=f"{type(error).__name__}: {error}",
                )
                for record in records
            ]
            failed_receipt = IngestionReceipt(
                source_kind=records[0].source_kind,
                workspace_id=workspace_id or _single_workspace(results),
                checkpoint=current_checkpoint,
                records=tuple(results),
            )
            if failure_mode == "strict":
                raise IngestionError(failed_receipt) from error
            return current_checkpoint, results, True

        results = _normalize_results(source, records, receipt.records)
        batch_has_failure = any(not result.successful for result in results)
        if checkpoint_blocked:
            candidate = current_checkpoint
        elif terminal_cursor is not None and all(
            result.successful for result in results
        ):
            candidate = terminal_cursor
        else:
            candidate = _safe_batch_checkpoint(
                results,
                current_checkpoint=current_checkpoint,
                strict=failure_mode == "strict",
            )
        batch_receipt = IngestionReceipt(
            source_kind=records[0].source_kind,
            workspace_id=workspace_id or _single_workspace(results),
            checkpoint=candidate,
            records=tuple(results),
        )
        if failure_mode == "strict" and batch_receipt.failed:
            raise IngestionError(batch_receipt)
        if candidate != current_checkpoint and checkpoint_store is not None:
            await checkpoint_store.save(
                records[0].source_kind,
                workspace_id,
                candidate,
            )
        return candidate, results, checkpoint_blocked or batch_has_failure

    async def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        limit: int = 10,
    ) -> RecordSearchOutcome:
        """Search the composed canonical record pipeline."""
        if self._orchestrator is None:
            raise RuntimeError(
                "Cannot search records: no record orchestrator was wired into SearchKernel"
            )
        return await self._orchestrator.search(query, limit=limit, filters=filters)


def _record_cursor(source: ContentSource, record: Record) -> Cursor:
    cursor_for = getattr(source, "cursor_for", None)
    if callable(cursor_for):
        cursor = cursor_for(record)
        if cursor is not None:
            if not isinstance(cursor, str):
                raise TypeError("ContentSource.cursor_for must return a string or None")
            return cursor
    for key in ("cursor", "source_cursor"):
        value = record.metadata.get(key)
        if value is not None:
            if not isinstance(value, str):
                raise TypeError(f"record metadata {key!r} must be a string")
            return value
    return record.updated_at.isoformat()


def _normalize_results(
    source: ContentSource,
    records: Sequence[Record],
    outcomes: Sequence[RecordIngestionResult],
) -> list[RecordIngestionResult]:
    if len(outcomes) > len(records):
        raise ValueError("ingestor returned more outcomes than input records")

    normalized: list[RecordIngestionResult] = []
    for index, record in enumerate(records):
        if index >= len(outcomes):
            normalized.append(
                RecordIngestionResult(
                    source_kind=record.source_kind,
                    source_id=record.source_id,
                    workspace_id=record.workspace_id,
                    status="failed",
                    cursor=_record_cursor(source, record),
                    error="ingestor did not return an outcome for this record",
                )
            )
            continue
        outcome = outcomes[index]
        normalized.append(
            RecordIngestionResult(
                source_kind=record.source_kind,
                source_id=record.source_id,
                workspace_id=record.workspace_id,
                status=outcome.status,
                cursor=outcome.cursor or _record_cursor(source, record),
                error=outcome.error,
            )
        )
    return normalized


def _safe_batch_checkpoint(
    results: Sequence[RecordIngestionResult],
    *,
    current_checkpoint: Cursor,
    strict: bool,
) -> Cursor:
    if strict and any(not result.successful for result in results):
        return current_checkpoint

    candidate = current_checkpoint
    for result in results:
        if not result.successful:
            break
        candidate = result.cursor
    return candidate


def _single_workspace(
    results: Sequence[RecordIngestionResult],
) -> str | None:
    workspaces = {result.workspace_id for result in results}
    return next(iter(workspaces)) if len(workspaces) == 1 else None


__all__ = ["SearchKernel"]
