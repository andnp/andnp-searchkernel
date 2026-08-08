import asyncio
import json
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
from searchkernel.indices.faiss_local import FAISSConfiguration
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


def test_auto_vector_engine_reuses_selection_until_vector_epoch_changes(
    monkeypatch,
) -> None:
    backend = LocalRecordBackend()
    vector = LocalVectorStore(backend, engine="auto")
    record = _record("one", [1.0, 0.0])
    backend.upsert([record], "model", 2)

    calls = 0
    vector_count = backend.vector_count

    def counted_vector_count(model_name: str, dim: int) -> int:
        nonlocal calls
        calls += 1
        return vector_count(model_name, dim)

    monkeypatch.setattr(backend, "vector_count", counted_vector_count)
    vector.search([1.0, 0.0], 1, model_name="model", dim=2)
    vector.search([1.0, 0.0], 1, model_name="model", dim=2)

    assert calls == 1


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
    assert fallback.last_search_diagnostics["fallback"] is False
    assert fallback.last_search_diagnostics["persistence"] == "rebuilt"
    assert "persistence_reason" in fallback.last_search_diagnostics


def test_faiss_execution_fallback_reports_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = LocalRecordBackend()
    record = _record("one", [1.0, 0.0])
    backend.upsert([record], "model", 2)
    store = FAISSLocalVectorStore(backend)

    def fail_to_load(*args: object, **kwargs: object) -> object:
        raise RuntimeError("index unavailable")

    monkeypatch.setattr(store, "_get_state", fail_to_load)

    hits = store.search([1.0, 0.0], 1, model_name="model", dim=2)

    assert [hit.source_id for hit in hits] == ["one"]
    assert store.last_search_diagnostics["fallback"] is True
    assert store.last_search_diagnostics["fallback_reason"] == (
        "RuntimeError: index unavailable"
    )


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("hnsw_m", True, TypeError),
        ("hnsw_ef_construction", 0, ValueError),
        ("hnsw_ef_search", "16", TypeError),
        ("overfetch_multiplier", float("inf"), ValueError),
        ("max_scan_rounds", 0, ValueError),
        ("max_scan_candidates", False, TypeError),
    ],
)
def test_faiss_configuration_validation(
    field: str, value: object, error: type[Exception]
) -> None:
    kwargs = {field: value}

    with pytest.raises(error):
        FAISSConfiguration(**kwargs)


def test_faiss_configuration_fingerprint_persists_hnsw_settings(
    tmp_path: Path,
) -> None:
    pytest.importorskip("faiss")
    backend = LocalRecordBackend(tmp_path / "records.db")
    backend.upsert(
        [_record("one", [1.0, 0.0]), _record("two", [0.0, 1.0])],
        "model",
        2,
    )
    store = FAISSLocalVectorStore(
        backend,
        index_path=tmp_path / "faiss",
        search_strategy="approximate",
        hnsw_m=12,
        hnsw_ef_construction=27,
        hnsw_ef_search=73,
    )
    store.search([1.0, 0.0], 1, model_name="model", dim=2)

    metadata_path = next((tmp_path / "faiss").glob("*.json"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["configuration"] == store.configuration.as_dict()
    assert metadata["configuration_fingerprint"] == store.configuration.fingerprint

    reloaded = FAISSLocalVectorStore(
        backend,
        index_path=tmp_path / "faiss",
        search_strategy="approximate",
        hnsw_m=12,
        hnsw_ef_construction=27,
        hnsw_ef_search=73,
    )
    state = reloaded._get_state("model", 2)
    import faiss

    hnsw = faiss.downcast_index(state.index.index).hnsw
    assert hnsw.efConstruction == 27
    assert hnsw.efSearch == 73
    assert reloaded.last_search_diagnostics["persistence"] == "loaded"


def test_faiss_approximate_search_enforces_filtered_candidate_budget(
    tmp_path: Path,
) -> None:
    pytest.importorskip("faiss")
    backend = LocalRecordBackend(tmp_path / "records.db")
    backend.upsert(
        [
            _record("blocked-1", [1.0, 0.0], workspace_id="other"),
            _record("blocked-2", [0.99, 0.1], workspace_id="other"),
            _record("eligible-1", [0.8, 0.6]),
            _record("eligible-2", [0.6, 0.8]),
        ],
        "model",
        2,
    )
    store = FAISSLocalVectorStore(
        backend,
        index_path=tmp_path / "faiss",
        search_strategy="approximate",
        max_scan_candidates=2,
    )

    hits = store.search(
        [1.0, 0.0],
        2,
        model_name="model",
        dim=2,
        filters={"workspace_id": "workspace"},
    )

    diagnostics = store.last_search_diagnostics
    assert len(hits) < 2
    assert diagnostics["candidate_budget"] == 2
    assert diagnostics["scan_limit"] <= 2
    assert diagnostics["candidate_budget_hit"] is True
    assert diagnostics["under_returned"] is True


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


def test_faiss_exact_filtered_search_does_not_under_return_candidates(
    tmp_path: Path,
) -> None:
    pytest.importorskip("faiss")
    backend = LocalRecordBackend(tmp_path / "records.db")
    records = [
        _record("blocked-1", [1.0, 0.0], workspace_id="other"),
        _record("blocked-2", [0.99, 0.1], workspace_id="other"),
        _record("blocked-3", [0.97, 0.24], workspace_id="other"),
        _record("eligible-1", [0.9, 0.4358899]),
        _record("eligible-2", [0.8, 0.6]),
    ]
    backend.upsert(records, "model", 2)
    store = FAISSLocalVectorStore(
        backend,
        index_path=tmp_path / "faiss",
        overfetch_multiplier=1.0,
        max_scan_rounds=1,
    )

    hits = store.search(
        [1.0, 0.0],
        2,
        model_name="model",
        dim=2,
        filters={"workspace_id": "workspace"},
    )

    assert len(hits) == 2
    assert [hit.source_id for hit in hits] == ["eligible-1", "eligible-2"]


@pytest.mark.parametrize(
    "filters",
    [
        {},
        {"workspace_id": "workspace"},
        {"project_id": "project-a"},
        {"statuses": [RecordStatus.ARCHIVED]},
        {"paths": ["docs/a.md"]},
    ],
)
def test_faiss_exact_search_matches_local_filter_parity(
    tmp_path: Path,
    filters: dict[str, object],
) -> None:
    pytest.importorskip("faiss")
    backend = LocalRecordBackend(tmp_path / "records.db")
    records = [
        _record(
            "a",
            [1.0, 0.0],
            metadata={"project_id": "project-a", "file_path": "docs/a.md"},
        ),
        _record(
            "b",
            [0.9, 0.4358899],
            metadata={"project_id": "project-b", "file_path": "docs/b.md"},
        ),
        _record(
            "c",
            [0.8, 0.6],
            workspace_id="other",
            metadata={"project_id": "project-a", "file_path": "docs/c.md"},
        ),
        _record(
            "archived",
            [0.7, 0.7141428],
            status=RecordStatus.ARCHIVED,
            metadata={"project_id": "project-a", "file_path": "docs/a.md"},
        ),
    ]
    backend.upsert(records, "model", 2)
    store = FAISSLocalVectorStore(backend, index_path=tmp_path / "faiss")

    expected = backend.search_vector(
        [1.0, 0.0], 10, model_name="model", dim=2, filters=filters
    )
    actual = store.search(
        [1.0, 0.0], 10, model_name="model", dim=2, filters=filters
    )

    assert [hit.source_id for hit in actual] == [
        hit.source_id for hit in expected
    ]
    assert [hit.score for hit in actual] == pytest.approx(
        [hit.score for hit in expected]
    )


def test_vector_document_exclusion_does_not_apply_path_variants() -> None:
    backend = LocalRecordBackend()
    kept = _record(
        "kept",
        [1.0, 0.0],
        metadata={"doc_id": "guide"},
    )
    excluded = _record(
        "excluded",
        [0.9, 0.1],
        metadata={"doc_id": "guide.md"},
    )
    backend.upsert([kept, excluded], "model", 2)

    hits = backend.search_vector(
        [1.0, 0.0],
        10,
        model_name="model",
        dim=2,
        filters={"excluded_documents": ["guide.md"]},
    )

    assert [hit.source_id for hit in hits] == ["kept"]


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
