"""Behavioral contracts for local-backend performance paths."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

import searchkernel.indices.local as local_indices
from searchkernel.domain import Record, RecordHit, RecordStatus
from searchkernel.indices import LocalRecordBackend, LocalVectorStore
from searchkernel.indices.local_vectors import PackedVectorCodec

_TIMESTAMP = datetime(2026, 1, 1, tzinfo=UTC)


def _record(
    source_id: str,
    embedding: list[float],
    *,
    metadata: dict[str, object] | None = None,
    indexed_text: str | None = None,
) -> Record:
    return Record(
        workspace_id="workspace",
        source_kind="note",
        source_id=source_id,
        title=source_id,
        body=f"raw body for {source_id}",
        indexed_text=indexed_text,
        created_at=_TIMESTAMP,
        updated_at=_TIMESTAMP,
        metadata=metadata or {},
        status=RecordStatus.ACTIVE,
        embedding=embedding,
    )


def _vector_records() -> list[Record]:
    return [
        _record("plain", [1.0, 0.0], metadata={"group": "a"}),
        _record("missing", [0.9, 0.1], metadata={}),
        _record("none", [0.8, 0.2], metadata={"group": None}),
        _record("list", [0.7, 0.3], metadata={"group": ["a"]}),
        _record("other", [0.0, 1.0], metadata={"group": "b"}),
    ]


def _load_vectors(backend: LocalRecordBackend, records: list[Record]) -> None:
    backend.upsert(records, "model", 2)


def _hit_signature(hits: list[RecordHit]) -> list[tuple[str, float]]:
    return [(hit.storage_key, hit.score) for hit in hits]


def test_vector_snapshot_and_block_paths_preserve_filter_semantics() -> None:
    """Filtered and unfiltered block searches match snapshot search.

    Missing metadata, explicit nulls, arrays, and ordinary values must retain
    their existing string-comparison semantics in both retrieval paths.
    """
    records = _vector_records()
    snapshot = LocalRecordBackend()
    block = LocalRecordBackend(vector_snapshot_max_rows=2, vector_snapshot_max_bytes=16)
    _load_vectors(snapshot, records)
    _load_vectors(block, records)

    filters_to_check = [
        None,
        {"metadata_equals": {"group": "a"}},
        {"metadata_equals": {"group": "None"}},
        {"metadata_equals": {"missing": "None"}},
        {"metadata_in": {"group": ["a", "b"]}},
    ]
    for filters in filters_to_check:
        expected = snapshot.search_vector(
            [1.0, 0.0], 10, model_name="model", dim=2, filters=filters
        )
        actual = block.search_vector(
            [1.0, 0.0], 10, model_name="model", dim=2, filters=filters
        )
        assert [hit.storage_key for hit in actual] == [
            hit.storage_key for hit in expected
        ]
        assert [hit.score for hit in actual] == pytest.approx(
            [hit.score for hit in expected]
        )


def test_batch_vector_decode_matches_scalar_validation() -> None:
    """Batch decoding preserves normalized values and numerical scores."""
    payloads = [
        PackedVectorCodec.encode([3.0, 4.0], 2),
        PackedVectorCodec.encode([5.0, 12.0], 2),
    ]

    batch = PackedVectorCodec.decode_batch(payloads, 2)
    scalar = np.vstack([PackedVectorCodec.decode(payload, 2) for payload in payloads])

    assert batch.dtype == np.dtype("<f4")
    assert batch @ np.asarray([0.6, 0.8], dtype=np.float32) == pytest.approx(
        scalar @ np.asarray([0.6, 0.8], dtype=np.float32)
    )


def test_fts_projection_keeps_query_variants_correct_and_narrow(tmp_path: Path) -> None:
    """Plain, artifact, and metadata-filtered FTS queries keep their contracts.

    The plain path should not hydrate large text or metadata columns, while
    artifact reranking and Python-side filters must still request what they use.
    """
    backend = LocalRecordBackend(tmp_path / "records.db")
    records = [
        _record(
            "plain",
            [1.0, 0.0],
            metadata={"kind": "plain"},
            indexed_text="needle indexed text",
        ),
        _record(
            "artifact",
            [0.0, 1.0],
            metadata={"kind": "artifact"},
            indexed_text="guide.py needle",
        ),
    ]
    backend.index(records)
    connection = backend.db_manager.get_connection()

    traces: list[str] = []
    connection.set_trace_callback(traces.append)
    assert backend.search_keyword("needle", 10)
    connection.set_trace_callback(None)
    plain_sql = next(sql for sql in traces if "local_records_fts MATCH" in sql)
    assert "r.body" not in plain_sql
    assert "r.indexed_text" not in plain_sql
    assert "r.metadata" not in plain_sql

    traces.clear()
    connection.set_trace_callback(traces.append)
    assert backend.search_keyword("guide.py", 10)
    connection.set_trace_callback(None)
    artifact_sql = next(sql for sql in traces if "local_records_fts MATCH" in sql)
    assert "r.body" in artifact_sql
    assert "r.indexed_text" in artifact_sql

    traces.clear()
    connection.set_trace_callback(traces.append)
    filtered = backend.search_keyword(
        "needle", 10, {"metadata_equals": {"kind": "plain"}}
    )
    connection.set_trace_callback(None)
    filtered_sql = next(sql for sql in traces if "local_records_fts MATCH" in sql)
    assert [hit.source_id for hit in filtered] == ["plain"]
    assert "r.metadata" in filtered_sql


@pytest.mark.parametrize("database_kind", ["file", "memory"])
def test_concurrent_vector_reads_and_writes_remain_correct(
    tmp_path: Path,
    database_kind: str,
) -> None:
    """Concurrent reads and a committing write do not corrupt local results."""
    db_path = tmp_path / "records.db" if database_kind == "file" else None
    backend = LocalRecordBackend(
        db_path,
        vector_snapshot_max_rows=10,
        vector_snapshot_max_bytes=80,
    )
    records = [
        _record(f"record-{index}", [1.0, 0.0], metadata={"n": index})
        for index in range(40)
    ]
    _load_vectors(backend, records)
    backend.search_vector([1.0, 0.0], 5, model_name="model", dim=2)

    def read() -> list[str]:
        return [
            hit.source_id
            for hit in backend.search_vector(
                [1.0, 0.0], 5, model_name="model", dim=2
            )
        ]

    def write() -> None:
        backend.upsert(
            [_record("record-0", [0.0, 1.0], metadata={"n": "updated"})],
            "model",
            2,
        )

    def run_write() -> list[str] | None:
        write()
        return None

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures: list[Future[list[str] | None]] = [
            executor.submit(read) for _ in range(12)
        ]
        futures.append(executor.submit(run_write))
        results = [future.result() for future in futures]

    assert all(isinstance(result, (list, type(None))) for result in results)
    assert backend.search_vector(
        [0.0, 1.0], 1, model_name="model", dim=2
    )[0].source_id == "record-0"


def test_adaptive_routing_reuses_stable_warmed_selection(monkeypatch) -> None:
    """Adaptive routing calibrates once and preserves the selected result path."""
    backend = LocalRecordBackend(faiss_threshold=1)
    record = _record("one", [1.0, 0.0])
    _load_vectors(backend, [record])
    vector = LocalVectorStore(backend, engine="auto")

    class _FakeFAISS:
        calls = 0

        def __init__(self, *args: object, **kwargs: object) -> None:
            self.last_search_diagnostics = {"fallback": False}

        def search(
            self,
            query_vector: list[float],
            k: int,
            *,
            model_name: str,
            dim: int,
            filters: dict[str, object] | None = None,
        ) -> list[RecordHit]:
            type(self).calls += 1
            return [RecordHit(record.identity, 1.0)]

    monkeypatch.setattr(local_indices, "FAISSLocalVectorStore", _FakeFAISS)
    clock_values = iter((0.0, 0.010, 0.010, 0.015))
    monkeypatch.setattr(local_indices.time, "perf_counter", lambda: next(clock_values))

    first = vector.search([1.0, 0.0], 1, model_name="model", dim=2)
    measurement = vector.last_routing_measurement
    second = vector.search([1.0, 0.0], 1, model_name="model", dim=2)

    assert _hit_signature(first) == _hit_signature(second)
    assert measurement is vector.last_routing_measurement
    assert vector.engine_name == "faiss"
    assert _FakeFAISS.calls == 3
