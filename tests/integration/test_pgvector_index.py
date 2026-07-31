"""Integration tests for PGVectorIndex, the VectorIndex-shaped adapter over
PGVectorStore that lets the live index/search path run on pgvector.

Uses a stub embedder (no real model load) against a live Postgres, so these
tests are fast while still exercising the real SQL. The integration conftest
starts a temporary pgvector container when SEARCHKERNEL_PG_DSN is absent.

Each test gets its own uuid-suffixed embedding-model name and doc/chunk id
prefix so tests stay isolated from each other under the repo's default
`--dist worksteal` parallel test execution (`pytest.mark.xdist_group`,
used elsewhere in this suite for "serial" tests, has no effect under
worksteal -- it is only honored under `--dist loadgroup`), rather than
relying on wiping shared Postgres state between tests.
"""

import os
import uuid
from datetime import UTC, datetime

import pytest

from searchkernel.adapters.stores.pgvector_index import PGVectorIndex
from searchkernel.domain import Chunk


def _with_hash(chunk):
    """Finalize a freshly-built domain.Chunk (test helper).

    domain.Chunk, unlike the legacy models.Chunk, does not auto-compute
    content_hash in __post_init__, and its metadata dict must stay JSON
    serializable (it flows into index/docstore persistence), so a raw
    datetime `modified_time` is normalized to ISO text.
    """
    if not chunk.content_hash:
        chunk.content_hash = chunk.compute_content_hash()
    modified_time = chunk.metadata.get("modified_time")
    if hasattr(modified_time, "isoformat"):
        chunk.metadata["modified_time"] = modified_time.isoformat()
    return chunk


class _StubEmbedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.dim = 4

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t) % 7 + 1), 0.0, 0.0, 0.0] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text) % 7 + 1), 0.0, 0.0, 0.0]


class _CountingEmbedder(_StubEmbedder):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.embed_calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls += 1
        return super().embed(texts)


@pytest.fixture
def pg_dsn():
    dsn = os.environ.get("SEARCHKERNEL_PG_DSN")
    if not dsn:
        pytest.skip("SEARCHKERNEL_PG_DSN not set")
    return dsn


@pytest.fixture
def prefix() -> str:
    return uuid.uuid4().hex[:12]


@pytest.fixture
def index(pg_dsn, prefix):
    return PGVectorIndex(pg_dsn=pg_dsn, embedder=_StubEmbedder(f"stub-{prefix}"))


def _chunk(chunk_id: str, doc_id: str, content: str, chunk_index: int = 0) -> Chunk:
    return _with_hash(Chunk(chunk_id=chunk_id, record_id=doc_id, content=content, metadata={ "header_path": "Intro", "start_pos": 0, "end_pos": len(content), "file_path": f"{doc_id}.md", "modified_time": datetime.now(UTC)}, chunk_index=chunk_index))


def test_add_chunks_then_get_chunk_by_id(index, prefix):
    chunk_id = f"{prefix}_doc1_chunk_0"
    index.add_chunks([_chunk(chunk_id, f"{prefix}_doc1", "hello world")])

    result = index.get_chunk_by_id(chunk_id)

    assert result is not None
    assert result["chunk_id"] == chunk_id
    assert result["doc_id"] == f"{prefix}_doc1"
    assert result["content"] == "hello world"
    assert result["header_path"] == "Intro"


def test_lifecycle_methods_are_safe_no_ops(index, tmp_path):
    index.add_chunks([])
    index.warm_up()
    index.persist(tmp_path / "index")
    index.load(tmp_path / "index")
    index.clear()
    assert index.is_ready() is True


def test_get_chunk_by_id_missing_returns_none(index, prefix):
    assert index.get_chunk_by_id(f"{prefix}_nope") is None


def test_search_returns_added_chunk(index, prefix):
    chunk_id_1 = f"{prefix}_doc1_chunk_0"
    chunk_id_2 = f"{prefix}_doc2_chunk_0"
    index.add_chunks(
        [
            _chunk(chunk_id_1, f"{prefix}_doc1", "hello world"),
            _chunk(chunk_id_2, f"{prefix}_doc2", "goodbye"),
        ]
    )

    results = index.search("hello world", top_k=5)

    chunk_ids = {r["chunk_id"] for r in results}
    assert chunk_id_1 in chunk_ids
    assert chunk_id_2 in chunk_ids


def test_search_empty_query_returns_empty(index):
    assert index.search("   ") == []


def test_search_excludes_matching_files(index, prefix, tmp_path):
    index.add_chunks(
        [
            _chunk(f"{prefix}_excluded_chunk_0", f"{prefix}_excluded", "hello world"),
            _chunk(f"{prefix}_kept_chunk_0", f"{prefix}_kept", "goodbye"),
        ]
    )

    results = index.search(
        "hello world",
        top_k=2,
        excluded_files={f"{prefix}_excluded"},
        docs_root=tmp_path,
    )

    assert results
    assert all(result["file_path"] != f"{prefix}_excluded.md" for result in results)


def test_get_chunk_ids_for_document_and_get_document_ids(index, prefix):
    doc1 = f"{prefix}_doc1"
    doc2 = f"{prefix}_doc2"
    index.add_chunks(
        [
            _chunk(f"{doc1}_chunk_0", doc1, "a", chunk_index=0),
            _chunk(f"{doc1}_chunk_1", doc1, "b", chunk_index=1),
            _chunk(f"{doc2}_chunk_0", doc2, "c", chunk_index=0),
        ]
    )

    assert set(index.get_chunk_ids_for_document(doc1)) == {
        f"{doc1}_chunk_0",
        f"{doc1}_chunk_1",
    }
    assert {doc1, doc2} <= set(index.get_document_ids())


def test_get_parent_content(index, prefix):
    chunk_id = f"{prefix}_doc1_chunk_0"
    index.add_chunks([_chunk(chunk_id, f"{prefix}_doc1", "parent text")])

    assert index.get_parent_content(chunk_id) == "parent text"
    assert index.get_parent_content(f"{prefix}_missing") is None


def test_get_embedding_for_empty_chunk_returns_none(index, prefix):
    chunk = _with_hash(
        Chunk(
            chunk_id=f"{prefix}_empty_chunk_0",
            record_id=f"{prefix}_empty",
            content="",
            metadata={},
        )
    )
    index.add_chunk(chunk)

    assert index.get_embedding_for_chunk(chunk.chunk_id) is None


def test_remove_deletes_all_chunks_for_document(index, prefix):
    doc1 = f"{prefix}_doc1"
    index.add_chunks(
        [
            _chunk(f"{doc1}_chunk_0", doc1, "a"),
            _chunk(f"{doc1}_chunk_1", doc1, "b"),
        ]
    )

    index.remove(doc1)

    assert index.get_chunk_ids_for_document(doc1) == []
    assert index.get_chunk_by_id(f"{doc1}_chunk_0") is None


def test_remove_chunk_deletes_one_chunk(index, prefix):
    doc1 = f"{prefix}_doc1"
    index.add_chunks(
        [
            _chunk(f"{doc1}_chunk_0", doc1, "a"),
            _chunk(f"{doc1}_chunk_1", doc1, "b"),
        ]
    )

    index.remove_chunk(f"{doc1}_chunk_0")

    assert index.get_chunk_by_id(f"{doc1}_chunk_0") is None
    assert index.get_chunk_by_id(f"{doc1}_chunk_1") is not None


def test_prune_document_returns_removed_count(index, prefix):
    doc1 = f"{prefix}_doc1"
    index.add_chunks(
        [
            _chunk(f"{doc1}_chunk_0", doc1, "a"),
            _chunk(f"{doc1}_chunk_1", doc1, "b"),
        ]
    )

    removed = index.prune_document(doc1)

    assert removed == 2
    assert index.get_chunk_ids_for_document(doc1) == []


def test_update_chunk_path_renames_chunk(index, prefix):
    old_chunk_id = f"{prefix}_old_chunk_0"
    new_chunk_id = f"{prefix}_new_chunk_0"
    index.add_chunks([_chunk(old_chunk_id, f"{prefix}_old", "moved content")])

    ok = index.update_chunk_path(
        old_chunk_id,
        new_chunk_id,
        {"doc_id": f"{prefix}_new", "header_path": "Intro", "file_path": "new.md"},
    )

    assert ok is True
    assert index.get_chunk_by_id(old_chunk_id) is None
    new_chunk = index.get_chunk_by_id(new_chunk_id)
    assert new_chunk is not None
    assert new_chunk["content"] == "moved content"
    assert new_chunk["doc_id"] == f"{prefix}_new"


def test_update_chunk_path_missing_source_returns_false(index, prefix):
    assert index.update_chunk_path(f"{prefix}_missing", f"{prefix}_new", {}) is False


def test_clear_removes_only_this_models_chunks(index, prefix):
    doc1 = f"{prefix}_doc1"
    index.add_chunks([_chunk(f"{doc1}_chunk_0", doc1, "a")])

    index.clear()

    assert index.get_chunk_ids_for_document(doc1) == []
    assert index.get_chunk_by_id(f"{doc1}_chunk_0") is None


def test_clear_preserves_same_chunk_in_other_model(index, prefix):
    chunk = _chunk(f"{prefix}_shared_chunk_0", f"{prefix}_shared", "shared content")
    other = PGVectorIndex(
        pg_dsn=index._conn_pool.dsn,
        embedder=_StubEmbedder(f"other-{prefix}"),
    )
    index.add_chunk(chunk)
    other.add_chunk(chunk)

    index.clear()

    assert index.get_chunk_by_id(chunk.chunk_id) is None
    assert other.get_chunk_by_id(chunk.chunk_id) is not None
    other.clear()


def test_is_ready_is_always_true(index):
    assert index.is_ready() is True


def test_expand_query_is_a_passthrough(index):
    assert index.expand_query("hello world") == "hello world"


def test_get_embedding_for_chunk_uses_stored_vector(pg_dsn, prefix):
    embedder = _CountingEmbedder(f"stored-{prefix}")
    index = PGVectorIndex(pg_dsn=pg_dsn, embedder=embedder)
    chunk_id = f"{prefix}_doc1_chunk_0"
    index.add_chunks([_chunk(chunk_id, f"{prefix}_doc1", "hello")])
    calls_after_insert = embedder.embed_calls

    embedding = index.get_embedding_for_chunk(chunk_id)

    assert embedding == [6.0, 0.0, 0.0, 0.0]
    assert embedder.embed_calls == calls_after_insert


def test_get_embedding_for_chunk_missing_returns_none(index, prefix):
    assert index.get_embedding_for_chunk(f"{prefix}_missing") is None


def test_search_skips_missing_record_rows(index, prefix):
    chunk = _chunk(f"{prefix}_orphan_chunk_0", f"{prefix}_orphan", "orphan content")
    index.add_chunk(chunk)

    conn = index._conn_pool.get_connection()
    cursor = None
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM records WHERE record_id = %s;", (chunk.storage_key,))
        conn.commit()
    finally:
        if cursor is not None:
            cursor.close()
        index._conn_pool.put_connection(conn)

    assert index.search("orphan content") == []


def test_workspace_scopes_hydration_and_delete(pg_dsn, prefix):
    embedder = _StubEmbedder(f"workspace-{prefix}")
    first = PGVectorIndex(
        pg_dsn=pg_dsn,
        embedder=embedder,
        workspace_id="workspace-a",
    )
    second = PGVectorIndex(
        pg_dsn=pg_dsn,
        embedder=_StubEmbedder(f"workspace-{prefix}"),
        workspace_id="workspace-b",
    )
    chunk = _chunk(f"{prefix}_shared_chunk_0", f"{prefix}_shared", "shared")

    first.add_chunk(chunk)
    second.add_chunk(chunk)

    assert first.get_chunk_by_id(chunk.chunk_id) is not None
    assert second.get_chunk_by_id(chunk.chunk_id) is not None

    first.remove_chunk(chunk.chunk_id)

    assert first.get_chunk_by_id(chunk.chunk_id) is None
    assert second.get_chunk_by_id(chunk.chunk_id) is not None
    first.add_chunk(chunk)
    first.clear()
    assert first.get_chunk_by_id(chunk.chunk_id) is None
    second.clear()
