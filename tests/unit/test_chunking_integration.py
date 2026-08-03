from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

from searchkernel.domain import Chunk, Record
from searchkernel.indexing.embedding_cache import SQLiteEmbeddingCache
from searchkernel.indices import LocalKeywordStore, LocalRecordBackend, LocalVectorStore
from searchkernel.ingestion import SemanticRecordIngestor
from searchkernel.search.record_pipeline import RecordSearchPipeline


@dataclass
class _Chunker:
    def chunk_record(self, record: Record) -> list[Chunk]:
        return [
            Chunk(
                f"{record.source_id}-0",
                record.source_id,
                "needle in the first section",
                {"header_path": "First", "start_pos": 0, "end_pos": 28},
                0,
            ),
            Chunk(
                f"{record.source_id}-1",
                record.source_id,
                "needle in the second section",
                {"header_path": "Second", "start_pos": 29, "end_pos": 58},
                1,
            ),
        ]


@dataclass
class _Embedder:
    model_name: str = "test"
    dim: int = 1

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text))] for text in texts]


def _record() -> Record:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    return Record(
        source_kind="note",
        source_id="note-1",
        title="Note",
        body="raw parent body",
        created_at=timestamp,
        updated_at=timestamp,
        workspace_id="workspace",
    )


@pytest.mark.asyncio
async def test_chunk_ingestion_search_returns_one_parent_with_excerpts(tmp_path) -> None:
    parent = _record()
    backend = LocalRecordBackend(tmp_path / "records.db")
    keyword = LocalKeywordStore(backend)
    vector = LocalVectorStore(backend)
    ingestor = SemanticRecordIngestor(
        embedding_provider=_Embedder(),
        keyword_store=keyword,
        vector_store=vector,
        embedding_cache=SQLiteEmbeddingCache(
            tmp_path / "embeddings.db",
            "test",
            1,
        ),
        chunker=_Chunker(),
    )

    receipt = await ingestor.index_records([parent])

    assert receipt.committed == 1
    assert len(backend.chunk_records(parent.identity)) == 2
    outcome = await RecordSearchPipeline(
        hydrator=backend,
        keyword_store=keyword,
    ).search("needle", limit=5, filters={"source_kind": "note"})

    assert len(outcome.results) == 1
    result = outcome.results[0]
    assert result.record.storage_key == parent.storage_key
    assert [match.chunk_id for match in result.excerpts] == [
        "note-1-0",
        "note-1-1",
    ]
    assert [match.metadata["header_path"] for match in result.excerpts] == [
        "First",
        "Second",
    ]
