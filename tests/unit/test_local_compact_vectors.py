import asyncio
import threading
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from searchkernel.domain import Record, RecordStatus
from searchkernel.indices import (
    FAISSLocalVectorStore,
    LocalRecordBackend,
    LocalVectorStore,
)
from searchkernel.indices.local_vectors import PackedVectorCodec


def _record(
    source_id: str,
    embedding: list[float],
    *,
    workspace_id: str | None = "workspace",
    status: RecordStatus = RecordStatus.ACTIVE,
    metadata: dict[str, object] | None = None,
) -> Record:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    return Record(
        workspace_id=workspace_id,
        source_kind="note",
        source_id=source_id,
        title=source_id,
        body=source_id,
        created_at=timestamp,
        updated_at=timestamp,
        status=status,
        metadata=metadata or {},
        embedding=embedding,
    )


def test_packed_vector_round_trip_normalizes_little_endian_float32() -> None:
    payload = PackedVectorCodec.encode([3.0, 4.0], 2)

    assert len(payload) == 8
    assert np.frombuffer(payload, dtype="<f4").tolist() == pytest.approx([0.6, 0.8])
    assert PackedVectorCodec.decode(payload, 2).tolist() == pytest.approx([0.6, 0.8])


@pytest.mark.parametrize(
    ("values", "dim", "message"),
    [
        ([1.0], 2, "dimension"),
        ([1.0, float("nan")], 2, "finite"),
        ([1.0, float("inf")], 2, "finite"),
        ([0.0, 0.0], 2, "non-zero"),
    ],
)
def test_packed_vector_validation_is_explicit(
    values: list[float], dim: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        PackedVectorCodec.encode(values, dim)


def test_local_vector_store_round_trip_uses_packed_schema(tmp_path: Path) -> None:
    backend = LocalRecordBackend(tmp_path / "records.db")
    store = LocalVectorStore(backend)
    record = _record("current", [3.0, 4.0])

    store.upsert([record], "model", 2)

    hits = store.search([3.0, 4.0], 1, model_name="model", dim=2)
    row = backend.db_manager.get_connection().execute(
        """
        SELECT embedding, format_version, normalization_policy
        FROM local_vectors_v2
        WHERE storage_key = ? AND encoder_namespace = ?
        """,
        (record.storage_key, "model"),
    ).fetchone()

    assert hits[0].storage_key == record.storage_key
    assert row[0] is not None
    assert len(row[0]) == 8
    assert row[1:] == (2, "l2")


def test_exact_search_has_cosine_parity_deterministic_ties_and_filters() -> None:
    backend = LocalRecordBackend()
    records = [
        _record("b", [3.0, 4.0]),
        _record("a", [0.6, 0.8]),
        _record("hidden", [3.0, 4.0], workspace_id="other"),
        _record("archived", [3.0, 4.0], status=RecordStatus.ARCHIVED),
    ]
    backend.upsert(records, "model:v1", 2)

    hits = backend.search_vector(
        [3.0, 4.0],
        2,
        model_name="model:v1",
        dim=2,
        filters={"workspace_id": "workspace"},
    )

    assert [hit.source_id for hit in hits] == ["a", "b"]
    assert hits[0].score == pytest.approx(1.0)
    assert backend.search_vector(
        [3.0, 4.0],
        10,
        model_name="model:v1",
        dim=2,
        filters={"statuses": ["archived"]},
    )[0].source_id == "archived"


def test_vector_metadata_and_snapshot_cache_follow_vector_epoch() -> None:
    backend = LocalRecordBackend()
    vector = LocalVectorStore(backend, engine="auto")
    record = _record("one", [1.0, 0.0])
    backend.upsert([record], "model", 2)

    queries: list[str] = []
    connection = backend.db_manager.get_connection()
    connection.set_trace_callback(queries.append)

    def stats_query_count() -> int:
        return sum(
            "COALESCE(SUM(length(embedding))" in query for query in queries
        )

    vector.search([1.0, 0.0], 1, model_name="model", dim=2)
    snapshot = backend._vector_snapshots[("model", 2)]
    vector.search([1.0, 0.0], 1, model_name="model", dim=2)

    assert stats_query_count() == 1
    assert backend._vector_snapshots[("model", 2)] is snapshot

    keyword_only = _record("keyword-only", [0.0, 1.0])
    backend.index([keyword_only])
    vector.search([1.0, 0.0], 1, model_name="model", dim=2)
    assert stats_query_count() == 1
    assert backend._vector_snapshots[("model", 2)] is snapshot

    record.embedding = [0.0, 1.0]
    backend.upsert([record], "model", 2)
    vector.search([0.0, 1.0], 1, model_name="model", dim=2)
    assert stats_query_count() == 2
    assert backend._vector_snapshots[("model", 2)] is not snapshot
    assert connection.execute(
        "SELECT COUNT(*) FROM local_vectors_v2"
    ).fetchone()[0] == 1
    vector.delete([record.storage_key])
    assert backend.vector_count("model", 2) == 0
    vector.search([0.0, 1.0], 1, model_name="model", dim=2)
    assert stats_query_count() == 3
    assert backend._vector_snapshots[("model", 2)] is not snapshot
    assert vector.search([0.0, 1.0], 1, model_name="model", dim=2) == []
    connection.set_trace_callback(None)


def test_snapshot_reload_corruption_and_deletion(tmp_path: Path) -> None:
    db_path = tmp_path / "records.db"
    backend = LocalRecordBackend(db_path)
    records = [_record("one", [1.0, 0.0]), _record("two", [0.0, 1.0])]
    backend.upsert(records, "model", 2)
    assert backend.search_vector([1.0, 0.0], 1, model_name="model", dim=2)

    conn = backend.db_manager.get_connection()
    conn.execute(
        """
        UPDATE local_vectors_v2
        SET embedding = ?
        WHERE storage_key = ?
        """,
        (b"corrupt", records[0].storage_key),
    )
    conn.commit()
    corrupted = LocalRecordBackend(db_path)
    with pytest.raises(ValueError, match="byte length mismatch"):
        corrupted.search_vector([1.0, 0.0], 1, model_name="model", dim=2)

    restored = LocalRecordBackend(db_path)
    restored.upsert([records[0]], "model", 2)
    restored.delete([records[0].storage_key])
    assert [
        hit.source_id
        for hit in restored.search_vector(
            [1.0, 0.0], 2, model_name="model", dim=2
        )
    ] == ["two"]


def test_optional_faiss_recall_reload_and_corruption_fallback(tmp_path: Path) -> None:
    pytest.importorskip("faiss")
    backend = LocalRecordBackend(tmp_path / "records.db")
    records = [
        _record("one", [1.0, 0.0]),
        _record("two", [0.9, 0.1]),
        _record("three", [0.0, 1.0]),
    ]
    store = FAISSLocalVectorStore(backend, index_path=tmp_path / "faiss")
    store.upsert(records, "model", 2)

    assert [hit.source_id for hit in store.search(
        [1.0, 0.0], 2, model_name="model", dim=2
    )] == ["one", "two"]
    assert store.verify_recall([1.0, 0.0], 2, model_name="model", dim=2) == 1.0

    index_files = list((tmp_path / "faiss").glob("*.faiss"))
    assert len(index_files) == 1
    index_files[0].write_bytes(b"corrupt")
    fallback = FAISSLocalVectorStore(backend, index_path=tmp_path / "faiss")
    assert fallback.search(
        [1.0, 0.0], 2, model_name="model", dim=2
    )[0].source_id == "one"


def test_faiss_batches_candidate_validation_and_preserves_filters(
    tmp_path: Path,
) -> None:
    pytest.importorskip("faiss")
    backend = LocalRecordBackend(tmp_path / "records.db")
    records = [
        _record("blocked", [1.0, 0.0], workspace_id="other"),
        _record("allowed", [0.9, 0.1], metadata={"project_id": "project-a"}),
        _record("other-project", [0.8, 0.2], metadata={"project_id": "project-b"}),
    ]
    store = FAISSLocalVectorStore(backend, index_path=tmp_path / "faiss")
    store.upsert(records, "model", 2)

    # Build and persist the state before tracing the search itself.
    assert store.search([1.0, 0.0], 1, model_name="model", dim=2)
    queries: list[str] = []
    connection = backend.db_manager.get_connection()
    connection.set_trace_callback(queries.append)

    hits = store.search(
        [1.0, 0.0],
        1,
        model_name="model",
        dim=2,
        filters={
            "workspace_id": "workspace",
            "metadata_equals": {"project_id": "project-a"},
        },
    )

    connection.set_trace_callback(None)
    assert [hit.source_id for hit in hits] == ["allowed"]
    assert not any("WHERE r.storage_key = ?" in query for query in queries)


@pytest.mark.asyncio
async def test_async_vector_search_offloads_blocking_work() -> None:
    store = LocalVectorStore(LocalRecordBackend())
    started = threading.Event()
    release = threading.Event()

    def blocked(*args: object, **kwargs: object) -> list[object]:
        started.set()
        release.wait(timeout=1.0)
        return []

    store.search = blocked  # type: ignore[method-assign]
    task = asyncio.create_task(
        store.async_search([1.0], 1, model_name="model", dim=1)
    )
    await asyncio.sleep(0)
    assert started.is_set()
    release.set()
    assert await task == []
