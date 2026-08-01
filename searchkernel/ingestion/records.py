"""Source-agnostic record ingestion over the kernel's storage ports."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Protocol

from searchkernel.domain import Cursor, Record
from searchkernel.indexing.semantic import (
    EmbeddingCache,
    SemanticInput,
    SemanticWorkPlanner,
    semantic_input_for_record,
)
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


class SemanticRecordIngestor:
    """Index source-agnostic records through keyword and vector stores."""

    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider,
        keyword_store: KeywordStore,
        vector_store: VectorStore,
        embedding_cache: EmbeddingCache,
        encoder_namespace: str | None = None,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.keyword_store = keyword_store
        self.vector_store = vector_store
        self.embedding_cache = embedding_cache
        self.encoder_namespace = (
            encoder_namespace
            or getattr(embedding_cache, "encoder_namespace", None)
            or embedding_provider.model_name
        )
        self._planner = SemanticWorkPlanner(
            self.encoder_namespace,
            dimension=embedding_provider.dim,
        )
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

        outcomes: list[RecordIngestionResult] = []
        for record in records:
            try:
                await asyncio.to_thread(self._index_record, record)
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

        return IngestionReceipt(
            source_kind=records[0].source_kind if records else "",
            workspace_id=_workspace_id(records),
            checkpoint=checkpoint,
            records=tuple(outcomes),
        )

    def _index_record(self, record: Record) -> None:
        semantic_input = semantic_input_for_record(
            record,
            self.encoder_namespace,
        )
        self._planner.execute(
            [semantic_input],
            self.embedding_cache,
            self._encoder,
            _RecordMaterializer(record, self.embedding_provider.model_name),
        )
        self.keyword_store.index([record])
        self.vector_store.upsert(
            [record],
            self.embedding_provider.model_name,
            self.embedding_provider.dim,
        )


def _workspace_id(records: Sequence[Record]) -> str | None:
    values = {record.workspace_id for record in records}
    if len(values) == 1:
        return next(iter(values))
    return None


__all__ = ["SemanticRecordIngestor"]
