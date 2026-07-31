import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from searchkernel.domain import GraphNeighbor, Record, RecordIdentity, RecordStatus
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
    title: str | None = None,
    uri: str | None = None,
    metadata: dict | None = None,
) -> Record:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    return Record(
        workspace_id=workspace_id,
        source_kind=source_kind,
        source_id=source_id,
        title=title or source_id,
        body=body,
        created_at=timestamp,
        updated_at=timestamp,
        status=status,
        uri=uri,
        metadata=metadata or {},
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
        GraphNeighbor(RecordIdentity("one", "commit", "same"), "related", 0.5)
    ]
    assert graph.neighbors(
        RecordIdentity("one", "note", "same")
    ) == [GraphNeighbor(RecordIdentity("one", "commit", "same"), "related", 0.5)]


def test_keyword_search_supports_tokens_phrases_prefixes_artifacts_and_symbols(
    tmp_path,
) -> None:
    backend = LocalRecordBackend(tmp_path / "records.db")
    records = [
        _record(
            "note",
            "guide",
            "alpha beta phrase",
            title="Alpha Guide",
            uri="docs/guide.py",
        ),
        _record("note", "prefix", "alphabet soup", title="Alphabet"),
        _record("note", "symbol", "parse_record implementation", title="Parser"),
        _record("note", "substring", "concatenate only", title="Other"),
    ]
    backend.index(records)

    assert [hit.source_id for hit in backend.search_keyword("alpha", 10)] == ["guide"]
    assert [
        hit.source_id for hit in backend.search_keyword('"alpha beta"', 10)
    ] == ["guide"]
    assert {
        hit.source_id for hit in backend.search_keyword("alph*", 10)
    } == {"guide", "prefix"}
    assert [hit.source_id for hit in backend.search_keyword("guide.py", 10)] == [
        "guide"
    ]
    assert [hit.source_id for hit in backend.search_keyword("parse_record", 10)] == [
        "symbol"
    ]


def test_keyword_search_handles_case_sanitization_and_keyword_metadata(tmp_path) -> None:
    backend = LocalRecordBackend(tmp_path / "records.db")
    record = _record(
        "note",
        "tagged",
        "Alpha body",
        metadata={"tags": ["Important"], "keywords": ["needle"]},
    )
    backend.index([record])

    assert backend.search_keyword("ALPHA", 10)
    assert [hit.source_id for hit in backend.search_keyword("NEEDLE", 10)] == [
        "tagged"
    ]


def test_keyword_search_applies_all_sql_filters(tmp_path) -> None:
    backend = LocalRecordBackend(tmp_path / "records.db")
    records = [
        _record("note", "one", "common", workspace_id="one"),
        _record("commit", "two", "common", workspace_id="one"),
        _record(
            "note",
            "three",
            "common",
            workspace_id="two",
            status=RecordStatus.STALE,
        ),
        _record(
            "note",
            "four",
            "common",
            workspace_id="one",
            status=RecordStatus.ARCHIVED,
        ),
    ]
    backend.index(records)

    assert {
        hit.source_id for hit in backend.search_keyword(
            "common", 10, {"workspace_id": "one"}
        )
    } == {"one", "two"}
    assert [hit.source_id for hit in backend.search_keyword(
        "common", 10, {"statuses": ["stale"], "workspace_id": "two"}
    )] == ["three"]
    assert [hit.source_id for hit in backend.search_keyword(
        "common", 10, {"status": RecordStatus.ARCHIVED, "include_inactive": False}
    )] == ["four"]
    assert [hit.source_id for hit in backend.search_keyword(
        "common", 10, {"source_kinds": ["commit"], "include_inactive": True}
    )] == ["two"]
    assert [hit.source_id for hit in backend.search_keyword(
        "common",
        10,
        {"candidate_storage_keys": [records[2].storage_key], "include_inactive": True},
    )] == ["three"]


def test_keyword_update_delete_and_rebuild_consistency(tmp_path) -> None:
    backend = LocalRecordBackend(tmp_path / "records.db")
    record = _record("note", "one", "before")
    backend.index([record])
    assert backend.check_keyword_index()
    assert backend.search_keyword("before", 10)

    record.body = "after"
    backend.index([record])
    assert not backend.search_keyword("before", 10)
    assert backend.search_keyword("after", 10)

    conn = backend.db_manager.get_connection()
    conn.execute(
        "DELETE FROM local_records_fts WHERE rowid = "
        "(SELECT rowid FROM local_records WHERE storage_key = ?)",
        (record.storage_key,),
    )
    conn.commit()
    assert not backend.check_keyword_index()
    backend.rebuild_keyword_index()
    assert backend.check_keyword_index()

    backend.delete([record.storage_key])
    assert not backend.search_keyword("after", 10)
    assert backend.check_keyword_index()


def test_keyword_ties_are_ordered_by_storage_key() -> None:
    backend = LocalRecordBackend()
    first = _record("note", "a", "same")
    second = _record("note", "b", "same")
    backend.index([second, first])

    hits = backend.search_keyword("same", 10)
    assert [hit.storage_key for hit in hits] == sorted(
        (first.storage_key, second.storage_key)
    )


def test_keyword_migrates_existing_local_records_schema(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE local_records (
            storage_key TEXT PRIMARY KEY,
            workspace_id TEXT,
            source_kind TEXT NOT NULL,
            source_id TEXT NOT NULL,
            title TEXT NOT NULL,
            body TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            metadata TEXT NOT NULL,
            uri TEXT,
            status TEXT NOT NULL
        )
        """
    )
    timestamp = datetime(2026, 1, 1, tzinfo=UTC).isoformat()
    conn.execute(
        "INSERT INTO local_records VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            'record:["workspace","note","legacy"]',
            "workspace",
            "note",
            "legacy",
            "Legacy",
            "migration body",
            timestamp,
            timestamp,
            '{"tags": ["migration"]}',
            "legacy.md",
            "active",
        ),
    )
    conn.commit()
    conn.close()

    backend = LocalRecordBackend(db_path)
    assert [hit.source_id for hit in backend.search_keyword("migration", 10)] == [
        "legacy"
    ]
    assert backend.check_keyword_index()


def test_scalar_and_batched_keyword_ingestion_are_equivalent() -> None:
    records = [
        _record("note", "one", "alpha"),
        _record("note", "two", "beta alpha"),
        _record("note", "three", "gamma"),
    ]
    scalar = LocalRecordBackend()
    batched = LocalRecordBackend()
    for record in records:
        scalar.index([record])
    batched.index(records)

    scalar_hits = scalar.search_keyword("alpha", 10)
    batch_hits = batched.search_keyword("alpha", 10)
    assert [(hit.storage_key, hit.score) for hit in scalar_hits] == [
        (hit.storage_key, hit.score) for hit in batch_hits
    ]


@pytest.mark.slow
def test_marked_keyword_scale_smoke() -> None:
    backend = LocalRecordBackend()
    records = [
        _record("note", f"record-{index}", f"common token {index}")
        for index in range(1_000)
    ]
    backend.index(records)

    hits = backend.search_keyword("common token", 10)
    assert len(hits) == 10
    assert backend.check_keyword_index()


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
