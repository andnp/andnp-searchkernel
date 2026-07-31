from datetime import UTC, datetime

import pytest

from searchkernel.domain import Record, RecordStatus
from searchkernel.indices import (
    LocalGraphStore,
    LocalKeywordStore,
    LocalRecordBackend,
    LocalVectorStore,
)
from searchkernel.runtime.local import LocalSearchSource
from searchkernel.search.orchestrator import SearchOrchestrator


class _Embedder:
    model_name = "test"
    dim = 2

    def embed_query(self, query: str) -> list[float]:
        return [1.0, 0.0] if query == "first" else [0.0, 1.0]


def _record(
    source_kind: str,
    source_id: str,
    body: str,
    *,
    workspace_id: str | None = None,
    status: RecordStatus = RecordStatus.ACTIVE,
) -> Record:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    return Record(
        workspace_id=workspace_id,
        source_kind=source_kind,
        source_id=source_id,
        title=source_id,
        body=body,
        created_at=timestamp,
        updated_at=timestamp,
        status=status,
    )


def _backend(tmp_path):
    backend = LocalRecordBackend(tmp_path / "records.db")
    return (
        backend,
        LocalKeywordStore(backend),
        LocalVectorStore(backend),
        LocalGraphStore(backend),
    )


def test_local_store_uses_collision_safe_identity_and_filters(tmp_path) -> None:
    backend, keyword, vector, graph = _backend(tmp_path)
    records = [
        _record("note", "same", "first alpha", workspace_id="one"),
        _record("commit", "same", "first beta", workspace_id="one"),
        _record("note", "same", "first hidden", workspace_id="two"),
        _record("note", "archived", "first archived", status=RecordStatus.ARCHIVED),
    ]
    for record in records:
        record.embedding = [1.0, 0.0]
    vector.upsert(records, "test", 2)
    keyword.index(records)
    graph.upsert_edges([(records[0].storage_key, records[1].storage_key, "related", 0.5)])

    hits = keyword.search("first", 10, {"workspace_id": "one"})
    vector_hits = vector.search(
        [1.0, 0.0], 10, model_name="test", dim=2, filters={"workspace_id": "one"}
    )

    assert [hit.storage_key for hit in hits] == [
        records[1].storage_key,
        records[0].storage_key,
    ]
    assert [hit.storage_key for hit in vector_hits] == [
        records[1].storage_key,
        records[0].storage_key,
    ]
    hydrated = backend.hydrate_record(records[0].storage_key)
    assert hydrated is not None
    assert hydrated.storage_key == records[0].storage_key
    assert hydrated.body == records[0].body
    assert backend.hydrate_record("same") is None
    assert graph.neighbors(records[0].storage_key) == [
        (records[1].storage_key, "related", 0.5)
    ]


@pytest.mark.asyncio
async def test_local_source_matches_record_pipeline_contract_deterministically(tmp_path) -> None:
    backend, keyword, vector, _graph = _backend(tmp_path)
    first = _record("note", "one", "first alpha", workspace_id="workspace")
    second = _record("note", "two", "first beta", workspace_id="workspace")
    first.embedding = [1.0, 0.0]
    second.embedding = [0.9, 0.1]
    vector.upsert([first, second], "test", 2)
    keyword.index([first, second])

    source = LocalSearchSource(
        SearchOrchestrator(
            hydrator=backend,
            keyword_store=keyword,
            vector_store=vector,
            embedding_provider=_Embedder(),
        )
    )
    results = list(
        await source.search("first", 2, {"workspace_id": "workspace"})
    )

    assert [result.storage_key for result in results] == sorted(
        (first.storage_key, second.storage_key)
    )
    assert all(result.metadata["text"].startswith("first") for result in results)
