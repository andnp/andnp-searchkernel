import asyncio
import json
import sqlite3
import threading
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

import searchkernel.indices.local as local_indices
from searchkernel.domain import Record, RecordHit, RecordIdentity, RecordStatus
from searchkernel.domain.vector_filters import compile_vector_filters
from searchkernel.indices import (
    FAISSLocalVectorStore,
    LocalRecordBackend,
    LocalVectorStore,
)
from searchkernel.indices.faiss_local import FAISSConfiguration
from searchkernel.indices.local_vectors import PackedVectorCodec, VectorSnapshot
from searchkernel.indices.vector_revision import record_embedding_revision


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


def test_vector_upsert_rejects_malformed_batch_without_mutation(tmp_path: Path) -> None:
    """A malformed vector rejects the whole batch before any row is stored."""
    backend = LocalRecordBackend(tmp_path / "records.db")
    good = _record("good", [1.0, 0.0])
    malformed = _record("malformed", [float("nan"), 0.0])

    with pytest.raises(ValueError, match="embedding for .*malformed.*finite"):
        backend.upsert([good, malformed], "model", 2)

    connection = backend.db_manager.get_connection()
    assert connection.execute("SELECT COUNT(*) FROM local_records").fetchone()[0] == 0
    assert connection.execute("SELECT COUNT(*) FROM local_vectors_v2").fetchone()[0] == 0
    assert backend.vector_epoch() == 0


def test_vector_upsert_preserves_duplicate_last_write_and_idempotent_epoch() -> None:
    """Duplicate keys keep the last vector and unchanged writes keep the epoch."""
    backend = LocalRecordBackend()
    first = _record("duplicate", [1.0, 0.0])
    last = _record("duplicate", [0.0, 1.0])

    backend.upsert([first, last], "model", 2)

    connection = backend.db_manager.get_connection()
    assert last.embedding is not None
    stored = connection.execute(
        "SELECT embedding FROM local_vectors_v2 WHERE storage_key = ?",
        (last.storage_key,),
    ).fetchone()[0]
    assert stored == PackedVectorCodec.encode(last.embedding, 2)
    assert connection.execute("SELECT COUNT(*) FROM local_vectors_v2").fetchone()[0] == 1
    epoch = backend.vector_epoch()

    backend.upsert([first, last], "model", 2)

    assert backend.vector_epoch() == epoch
    assert backend.search_vector(
        [0.0, 1.0], 1, model_name="model", dim=2
    )[0].storage_key == last.storage_key


def test_vector_upsert_rejects_model_dimension_change_without_mutation() -> None:
    """A model dimension change leaves its existing vector rows untouched."""
    backend = LocalRecordBackend()
    original = _record("original", [1.0, 0.0])
    backend.upsert([original], "model", 2)
    epoch = backend.vector_epoch()

    with pytest.raises(ValueError, match="Dimension mismatch"):
        backend.upsert([_record("new", [1.0, 0.0, 0.0])], "model", 3)

    connection = backend.db_manager.get_connection()
    assert connection.execute("SELECT COUNT(*) FROM local_vectors_v2").fetchone()[0] == 1
    assert backend.vector_epoch() == epoch


def test_local_vector_store_round_trip_uses_packed_schema(tmp_path: Path) -> None:
    backend = LocalRecordBackend(tmp_path / "records.db")
    store = LocalVectorStore(backend)
    record = _record("current", [3.0, 4.0])

    store.upsert([record], "model", 2)

    hits = store.search([3.0, 4.0], 1, model_name="model", dim=2)
    row = backend.db_manager.get_connection().execute(
        """
        SELECT embedding, revision, format_version, normalization_policy
        FROM local_vectors_v2
        WHERE storage_key = ? AND encoder_namespace = ?
        """,
        (record.storage_key, "model"),
    ).fetchone()

    assert hits[0].storage_key == record.storage_key
    assert row[0] is not None
    assert len(row[0]) == 8
    assert row[1] == record_embedding_revision(record, "model", 2)
    assert row[2:] == (2, "l2")


def test_record_embedding_revision_tracks_identity_content_model_and_dimension() -> None:
    """Revision changes when any embedding-relevant input changes."""
    record = _record("revision", [1.0, 0.0])

    base_revision = record_embedding_revision(record, "model", 2)
    revisions = {
        base_revision,
        record_embedding_revision(record, "other-model", 2),
        record_embedding_revision(record, "model", 3),
    }
    record.body = "changed content"
    changed_revision = record_embedding_revision(record, "model", 2)
    revisions.add(changed_revision)

    assert len(revisions) == 4
    assert changed_revision != base_revision


def test_local_vector_schema_repairs_legacy_rows_on_replacement(tmp_path: Path) -> None:
    """A replacement embedding repairs a legacy row's missing revision."""
    db_path = tmp_path / "legacy-vectors.db"
    backend = LocalRecordBackend(db_path)
    record = _record("legacy", [1.0, 0.0])
    backend.index([record])
    backend.close()

    conn = sqlite3.connect(db_path)
    conn.execute("DROP TABLE local_vectors_v2")
    conn.execute(
        """
        CREATE TABLE local_vectors_v2 (
            storage_key TEXT NOT NULL,
            encoder_namespace TEXT NOT NULL,
            dim INTEGER NOT NULL,
            embedding BLOB NOT NULL,
            format_version INTEGER NOT NULL,
            normalization_policy TEXT NOT NULL,
            PRIMARY KEY (storage_key, encoder_namespace, dim)
        )
        """
    )
    assert record.embedding is not None
    conn.execute(
        """
        INSERT INTO local_vectors_v2
            (storage_key, encoder_namespace, dim, embedding, format_version,
             normalization_policy)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            record.storage_key,
            "model",
            2,
            PackedVectorCodec.encode(record.embedding, 2),
            2,
            "l2",
        ),
    )
    conn.commit()
    conn.close()

    restored = LocalRecordBackend(db_path)
    columns = {
        row[1]
        for row in restored.db_manager.get_connection().execute(
            "PRAGMA table_info(local_vectors_v2)"
        )
    }
    revision = restored.db_manager.get_connection().execute(
        "SELECT revision FROM local_vectors_v2"
    ).fetchone()[0]

    assert "revision" in columns
    assert revision is None
    assert restored.search_vector([1.0, 0.0], 1, model_name="model", dim=2)

    restored.upsert([record], "model", 2)

    row = restored.db_manager.get_connection().execute(
        """
        SELECT revision, format_version, normalization_policy
        FROM local_vectors_v2
        """
    ).fetchone()
    assert tuple(row) == (record_embedding_revision(record, "model", 2), 2, "l2")


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


def _generic_snapshot_mask(
    snapshot: VectorSnapshot,
    filters: dict[str, object] | None,
) -> np.ndarray:
    predicate = compile_vector_filters(filters)
    return np.asarray(
        [
            predicate.matches(
                storage_key=storage_key,
                source_id=str(source_id),
                workspace_id=(
                    str(workspace_id) if workspace_id is not None else None
                ),
                source_kind=str(source_kind),
                status=str(status),
                metadata=metadata,
                uri=uri,
            )
            for storage_key, source_id, workspace_id, source_kind, status, metadata, uri in zip(
                snapshot.storage_keys,
                snapshot.source_ids,
                snapshot.workspace_ids,
                snapshot.source_kinds,
                snapshot.statuses,
                snapshot.metadata,
                snapshot.uris,
                strict=True,
            )
        ],
        dtype=bool,
    )


def _filter_snapshot() -> VectorSnapshot:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    rows = []
    for index, (workspace_id, source_kind, status) in enumerate(
        [
            ("workspace-a", "note", "active"),
            ("workspace-a", "commit", "stale"),
            ("workspace-b", "note", "active"),
            ("workspace-b", "note", "archived"),
        ]
    ):
        record = Record(
            workspace_id=workspace_id,
            source_kind=source_kind,
            source_id=f"record-{index}",
            title=f"record-{index}",
            body="body",
            created_at=timestamp,
            updated_at=timestamp,
            status=RecordStatus(status),
            metadata={"project_id": workspace_id},
            uri=f"/docs/{index}.md",
            embedding=[1.0, 0.0],
        )
        assert record.embedding is not None
        rows.append(
            {
                "storage_key": record.storage_key,
                "source_id": record.source_id,
                "workspace_id": record.workspace_id,
                "source_kind": record.source_kind,
                "status": record.status.value,
                "metadata": record.metadata,
                "uri": record.uri,
                "embedding": PackedVectorCodec.encode(record.embedding, 2),
                "format_version": 2,
                "normalization_policy": "l2",
            }
        )
    return VectorSnapshot.from_rows(
        rows,
        encoder_namespace="model",
        dim=2,
        epoch=1,
        materialize_metadata=True,
    )


@pytest.mark.parametrize(
    "filters",
    [
        None,
        {},
        {"status": RecordStatus.ACTIVE},
        {"statuses": ["active", "stale"]},
        {"workspace_id": "workspace-a"},
        {"source_kinds": ["note"]},
        {
            "candidate_ids": [
                RecordIdentity("workspace-a", "note", "record-0")
            ]
        },
        {
            "statuses": ["active"],
            "workspace_id": "workspace-a",
            "source_kind": "note",
            "candidate_storage_keys": [
                RecordIdentity("workspace-a", "note", "record-0").storage_key
            ],
        },
        {"candidate_ids": []},
    ],
)
def test_snapshot_scalar_filter_mask_matches_generic_filter(
    filters: dict[str, object] | None,
) -> None:
    """Scalar filters preserve the generic predicate's eligibility mask."""
    snapshot = _filter_snapshot()

    actual = snapshot.filter_mask(
        filters,
        status_values=set(),
        filter_values=None,
    )

    assert actual.tolist() == _generic_snapshot_mask(snapshot, filters).tolist()


@pytest.mark.parametrize(
    "filters",
    [
        {"metadata_equals": {"project_id": "workspace-a"}},
        {"paths": ["1.md"]},
        {"excluded_files": ["2.md"]},
        {
            "source_scoped_filters": {
                "note": {"workspace_ids": ["workspace-a"]}
            }
        },
        {"statuses": ["active"], "metadata_equals": {"project_id": "workspace-a"}},
        {"regex": "record-[01]"},
    ],
)
def test_snapshot_custom_filter_mask_matches_generic_filter(
    filters: dict[str, object],
) -> None:
    """Custom and mixed filters retain generic metadata and authorization semantics."""
    snapshot = _filter_snapshot()

    actual = snapshot.filter_mask(
        filters,
        status_values=set(),
        filter_values=None,
    )

    assert actual.tolist() == _generic_snapshot_mask(snapshot, filters).tolist()


def test_snapshot_filter_mask_preserves_empty_results_and_storage_order() -> None:
    """Filtering an empty snapshot stays empty and matches remain deterministic."""
    snapshot = _filter_snapshot()
    empty = VectorSnapshot.from_rows(
        [],
        encoder_namespace="model",
        dim=2,
        epoch=1,
    )

    assert empty.filter_mask(None, status_values=set(), filter_values=None).tolist() == []
    assert np.flatnonzero(
        snapshot.filter_mask(
            {"statuses": ["active", "stale"]},
            status_values=set(),
            filter_values=None,
        )
    ).tolist() == [0, 1, 2]


def test_filter_mask_raises_when_metadata_not_materialized() -> None:
    """A non-scalar filter still demands materialized metadata to score it."""
    snapshot = VectorSnapshot.from_rows(
        [
            {
                "storage_key": _record("only", [1.0, 0.0]).storage_key,
                "source_id": "only",
                "workspace_id": "workspace",
                "source_kind": "note",
                "status": "active",
                "embedding": PackedVectorCodec.encode([1.0, 0.0], 2),
                "format_version": 2,
                "normalization_policy": "l2",
            }
        ],
        encoder_namespace="model",
        dim=2,
        epoch=1,
    )

    with pytest.raises(ValueError, match="metadata must be materialized"):
        snapshot.filter_mask(
            {"metadata_equals": {"project_id": "keep"}},
            status_values=set(),
            filter_values=None,
        )


def test_metadata_materialization_reuses_vector_and_scalar_arrays() -> None:
    """Materializing metadata preserves the immutable vector search state."""
    record = _record("only", [1.0, 0.0], metadata={"project_id": "keep"})
    row = {
        "storage_key": record.storage_key,
        "source_id": record.source_id,
        "workspace_id": record.workspace_id,
        "source_kind": record.source_kind,
        "status": record.status.value,
        "metadata": record.metadata,
        "uri": "/docs/only.md",
        "embedding": PackedVectorCodec.encode([1.0, 0.0], 2),
        "format_version": 2,
        "normalization_policy": "l2",
    }

    cold = VectorSnapshot.from_rows(
        [row], encoder_namespace="model", dim=2, epoch=1
    )
    warm = cold.with_materialized_metadata([row])

    assert warm.matrix is cold.matrix
    assert warm.storage_keys is cold.storage_keys
    assert warm.source_ids is cold.source_ids
    assert warm.workspace_ids is cold.workspace_ids
    assert warm.source_kinds is cold.source_kinds
    assert warm.statuses is cold.statuses
    assert warm.metadata == ({"project_id": "keep"},)
    assert warm.uris == ("/docs/only.md",)


def test_metadata_materialization_rejects_mismatched_rows() -> None:
    """Metadata materialization rejects reordered and incomplete row sets."""
    record = _record("only", [1.0, 0.0], metadata={"project_id": "keep"})
    row = {
        "storage_key": record.storage_key,
        "metadata": record.metadata,
        "uri": "/docs/only.md",
        "embedding": PackedVectorCodec.encode([1.0, 0.0], 2),
        "format_version": 2,
        "normalization_policy": "l2",
    }
    snapshot = VectorSnapshot.from_rows(
        [
            {
                **row,
                "source_id": record.source_id,
                "workspace_id": record.workspace_id,
                "source_kind": record.source_kind,
                "status": record.status.value,
            }
        ],
        encoder_namespace="model",
        dim=2,
        epoch=1,
    )

    reordered = {**row, "storage_key": "other"}
    with pytest.raises(ValueError, match="ordering"):
        snapshot.with_materialized_metadata([reordered])
    with pytest.raises(ValueError, match="length"):
        snapshot.with_materialized_metadata([])


def _normalized_vectors(count: int, dim: int, *, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((count, dim))
    return [vector / np.linalg.norm(vector) for vector in raw]


def test_candidate_storage_keys_matches_unfiltered_scan_restricted_to_keys() -> None:
    """Candidate-key filtering agrees with trimming an unfiltered ranking."""
    backend = LocalRecordBackend()
    dim = 6
    vectors = _normalized_vectors(30, dim, seed=1)
    records = [
        _record(f"record-{index:02d}", vector.tolist())
        for index, vector in enumerate(vectors)
    ]
    backend.upsert(records, "model", dim)
    query = _normalized_vectors(1, dim, seed=2)[0].tolist()
    candidate_keys = [records[i].storage_key for i in range(0, 30, 3)]

    unfiltered = backend.search_vector(query, 30, model_name="model", dim=dim)
    filtered = backend.search_vector(
        query,
        10,
        model_name="model",
        dim=dim,
        filters={"candidate_storage_keys": candidate_keys},
    )

    candidate_set = set(candidate_keys)
    expected = [hit for hit in unfiltered if hit.storage_key in candidate_set][:10]
    assert [hit.storage_key for hit in filtered] == [hit.storage_key for hit in expected]
    assert [hit.score for hit in filtered] == pytest.approx(
        [hit.score for hit in expected]
    )


def test_candidate_storage_keys_and_metadata_filter_return_intersection() -> None:
    """Combining candidate keys with a metadata filter intersects both."""
    backend = LocalRecordBackend()
    dim = 6
    vectors = _normalized_vectors(30, dim, seed=3)
    records = [
        _record(f"record-{index:02d}", vector.tolist())
        for index, vector in enumerate(vectors)
    ]
    for index, record in enumerate(records):
        record.metadata = {"project_id": "keep" if index % 2 == 0 else "drop"}
    backend.upsert(records, "model", dim)
    query = _normalized_vectors(1, dim, seed=4)[0].tolist()
    candidate_keys = [records[i].storage_key for i in range(0, 30, 3)]
    metadata_filter = {"metadata_equals": {"project_id": "keep"}}

    metadata_only = backend.search_vector(
        query, 30, model_name="model", dim=dim, filters=metadata_filter
    )
    combined = backend.search_vector(
        query,
        10,
        model_name="model",
        dim=dim,
        filters={**metadata_filter, "candidate_storage_keys": candidate_keys},
    )

    candidate_set = set(candidate_keys)
    expected = [hit for hit in metadata_only if hit.storage_key in candidate_set][:10]
    assert [hit.storage_key for hit in combined] == [hit.storage_key for hit in expected]


def test_empty_candidate_storage_keys_return_no_vector_hits() -> None:
    """An explicitly empty candidate set excludes every local vector."""
    backend = LocalRecordBackend()
    records = [
        _record("first", [1.0, 0.0]),
        _record("second", [0.0, 1.0]),
    ]
    backend.upsert(records, "model", 2)

    hits = backend.search_vector(
        [1.0, 0.0],
        10,
        model_name="model",
        dim=2,
        filters={"candidate_storage_keys": []},
    )

    assert hits == []


@pytest.mark.parametrize(
    "filters",
    [
        {"workspace_id": "workspace"},
        {"statuses": ["active"]},
        {"metadata_equals": {"project_id": "keep"}},
        {"metadata_in": {"project_id": ["keep", "also-keep"]}},
        {
            "candidate_storage_keys": [
                RecordIdentity("workspace", "note", "middle").storage_key,
                RecordIdentity("workspace", "note", "first").storage_key,
            ]
        },
        {"candidate_storage_keys": []},
    ],
)
def test_local_vector_filters_match_generic_predicate_and_order(
    filters: dict[str, object],
) -> None:
    """Local filtered search matches generic eligibility and score ordering.

    Candidate keys, scalar constraints, metadata constraints, and an empty
    candidate set must all produce the same eligible ranking as the predicate.
    """
    backend = LocalRecordBackend()
    records = [
        _record("first", [1.0, 0.0], metadata={"project_id": "keep"}),
        _record("middle", [0.8, 0.6], metadata={"project_id": "also-keep"}),
        _record("last", [0.6, 0.8], metadata={"project_id": "drop"}),
        _record(
            "archived",
            [1.0, 0.0],
            status=RecordStatus.ARCHIVED,
            metadata={"project_id": "keep"},
        ),
    ]
    backend.upsert(records, "model", 2)

    actual = backend.search_vector(
        [1.0, 0.0], 10, model_name="model", dim=2, filters=filters
    )
    predicate = compile_vector_filters(filters)
    expected_records = [
        record
        for record in records
        if predicate.matches(
            storage_key=record.storage_key,
            source_id=record.source_id,
            workspace_id=record.workspace_id,
            source_kind=record.source_kind,
            status=record.status,
            metadata=record.metadata,
            uri=record.uri,
        )
    ]
    expected_scores: dict[str, float] = {}
    for record in expected_records:
        assert record.embedding is not None
        expected_scores[record.storage_key] = float(record.embedding[0])
    expected_records.sort(
        key=lambda record: (-expected_scores[record.storage_key], record.storage_key)
    )

    assert [hit.storage_key for hit in actual] == [
        record.storage_key for record in expected_records
    ]
    assert [hit.score for hit in actual] == pytest.approx(
        [expected_scores[record.storage_key] for record in expected_records]
    )


@pytest.mark.parametrize("candidate_fraction", [0.1, 0.8])
def test_search_vector_scoring_agrees_across_selectivity_branches(
    candidate_fraction: float,
) -> None:
    """Full-matvec and gathered scoring pick the same top-k in the same order."""
    backend = LocalRecordBackend()
    dim = 6
    record_count = 40
    vectors = _normalized_vectors(record_count, dim, seed=5)
    records = [
        _record(f"record-{index:02d}", vector.tolist())
        for index, vector in enumerate(vectors)
    ]
    backend.upsert(records, "model", dim)
    query = _normalized_vectors(1, dim, seed=6)[0]
    candidate_count = max(1, int(record_count * candidate_fraction))
    candidate_keys = [records[i].storage_key for i in range(candidate_count)]

    hits = backend.search_vector(
        query.tolist(),
        10,
        model_name="model",
        dim=dim,
        filters={"candidate_storage_keys": candidate_keys},
    )

    key_to_score = {
        records[i].storage_key: float(np.dot(vectors[i], query))
        for i in range(candidate_count)
    }
    expected_order = sorted(
        key_to_score,
        key=lambda key: (-key_to_score[key], key),
    )[:10]
    assert [hit.storage_key for hit in hits] == expected_order
    assert [hit.score for hit in hits] == pytest.approx(
        [key_to_score[key] for key in expected_order], rel=1e-5
    )


def test_vector_storage_stats_cache_follows_vector_epoch() -> None:
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
    assert backend.vector_storage_stats("model", 2) == (1, 8)
    vector.search([1.0, 0.0], 1, model_name="model", dim=2)

    assert stats_query_count() == 1

    keyword_only = _record("keyword-only", [0.0, 1.0])
    backend.index([keyword_only])
    vector.search([1.0, 0.0], 1, model_name="model", dim=2)
    assert stats_query_count() == 1

    record.embedding = [0.0, 1.0]
    backend.upsert([record], "model", 2)
    vector.search([0.0, 1.0], 1, model_name="model", dim=2)
    assert stats_query_count() == 2
    assert connection.execute(
        "SELECT COUNT(*) FROM local_vectors_v2"
    ).fetchone()[0] == 1
    vector.delete([record.storage_key])
    assert backend.vector_count("model", 2) == 0
    vector.search([0.0, 1.0], 1, model_name="model", dim=2)
    assert stats_query_count() == 3
    assert vector.search([0.0, 1.0], 1, model_name="model", dim=2) == []
    connection.set_trace_callback(None)


def test_vector_snapshot_is_invalidated_after_vector_epoch_changes() -> None:
    """A changed vector epoch replaces the snapshot's pre-mutation vectors.

    Public searches must rank against the updated embedding after an upsert.
    """
    backend = LocalRecordBackend()
    store = LocalVectorStore(backend)
    first = _record("first", [1.0, 0.0])
    second = _record("second", [0.0, 1.0])
    store.upsert([first, second], "model", 2)

    assert store.search([1.0, 0.0], 1, model_name="model", dim=2)[0].source_id == (
        "first"
    )
    before = store.vector_epoch()

    first.embedding = [-1.0, 0.0]
    store.upsert([first], "model", 2)

    assert store.vector_epoch() == before + 1
    assert store.search([1.0, 0.0], 1, model_name="model", dim=2)[0].source_id == (
        "second"
    )


def test_concurrent_snapshot_publication_stays_coherent_with_epoch(monkeypatch) -> None:
    """Concurrent publication cannot make a later search use an old epoch.

    Barriers place the mutation after row capture and before snapshot
    materialization, then verify a search started after that mutation sees it.
    """
    backend = LocalRecordBackend()
    store = LocalVectorStore(backend)
    first = _record("first", [1.0, 0.0])
    second = _record("second", [0.0, 1.0])
    store.upsert([first, second], "model", 2)
    assert store.search([1.0, 0.0], 1, model_name="model", dim=2)

    first.embedding = [0.0, 1.0]
    second.embedding = [1.0, 0.0]
    store.upsert([first, second], "model", 2)

    rows_ready = threading.Event()
    publish = threading.Event()
    from_rows_calls = 0
    from_rows_lock = threading.Lock()
    original_from_rows = VectorSnapshot.from_rows

    def gated_from_rows(
        rows: Sequence[object],
        *,
        encoder_namespace: str,
        dim: int,
        epoch: int,
        materialize_metadata: bool = False,
    ) -> VectorSnapshot:
        nonlocal from_rows_calls
        with from_rows_lock:
            from_rows_calls += 1
            should_pause = from_rows_calls == 1
        if should_pause:
            rows_ready.set()
            if not publish.wait(timeout=5):
                raise AssertionError("snapshot publication was not released")
        return original_from_rows(
            rows,
            encoder_namespace=encoder_namespace,
            dim=dim,
            epoch=epoch,
            materialize_metadata=materialize_metadata,
        )

    monkeypatch.setattr(VectorSnapshot, "from_rows", gated_from_rows)
    with ThreadPoolExecutor(max_workers=2) as executor:
        publisher = executor.submit(
            store.search, [1.0, 0.0], 1, model_name="model", dim=2
        )
        observer: Future[list[RecordHit]] | None = None
        try:
            assert rows_ready.wait(timeout=5)
            first.embedding = [1.0, 0.0]
            second.embedding = [0.0, 1.0]
            store.upsert([first, second], "model", 2)
            observer = executor.submit(
                store.search, [1.0, 0.0], 1, model_name="model", dim=2
            )
            publish.set()
        finally:
            publish.set()

        assert observer is not None
        publisher_hits = publisher.result(timeout=5)
        observer_hits = observer.result(timeout=5)

    assert [hit.source_id for hit in publisher_hits] == ["second"]
    assert [hit.source_id for hit in observer_hits] == ["first"]


def test_vector_snapshot_keys_isolate_models_and_dimensions() -> None:
    """Snapshot entries remain isolated by encoder model and vector dimension.

    Alternating public searches must not reuse another model's or dimension's
    immutable vector rows.
    """
    backend = LocalRecordBackend()
    store = LocalVectorStore(backend)
    model_two = _record("model-two", [1.0, 0.0])
    other_two = _record("other-two", [0.0, 1.0])
    model_three = _record("model-three", [1.0, 0.0, 0.0])
    store.upsert([model_two], "model-two", 2)
    store.upsert([other_two], "other-two", 2)
    store.upsert([model_three], "model-three", 3)

    assert [
        hit.source_id
        for hit in store.search([1.0, 0.0], 1, model_name="model-two", dim=2)
    ] == ["model-two"]
    assert [
        hit.source_id
        for hit in store.search([0.0, 1.0], 1, model_name="other-two", dim=2)
    ] == ["other-two"]
    assert [
        hit.source_id
        for hit in store.search(
            [1.0, 0.0, 0.0], 1, model_name="model-three", dim=3
        )
    ] == ["model-three"]
    assert [
        hit.source_id
        for hit in store.search([1.0, 0.0], 1, model_name="model-two", dim=2)
    ] == ["model-two"]


def test_vector_search_materializes_metadata_only_for_metadata_filters() -> None:
    """Normal searches defer row metadata while metadata filters stay exact."""
    backend = LocalRecordBackend()
    vector = LocalVectorStore(backend)
    record = _record("metadata", [1.0, 0.0])
    record.metadata = {"category": "keep"}
    backend.upsert([record], "model", 2)

    assert vector.search([1.0, 0.0], 1, model_name="model", dim=2)

    hits = vector.search(
        [1.0, 0.0],
        1,
        model_name="model",
        dim=2,
        filters={"metadata_equals": {"category": "keep"}},
    )

    assert [hit.storage_key for hit in hits] == [record.storage_key]


def test_cached_vector_snapshot_reuses_cold_state_for_metadata_filters(monkeypatch) -> None:
    """A warm metadata filter reuses the cold snapshot build and its ordering."""
    records = [
        _record("drop", [1.0, 0.0], metadata={"category": "drop"}),
        _record("keep-first", [0.8, 0.6], metadata={"category": "keep"}),
        _record("keep-second", [0.6, 0.8], metadata={"category": "keep"}),
    ]
    backend = LocalRecordBackend()
    vector = LocalVectorStore(backend)
    backend.upsert(records, "model", 2)

    original_from_rows = VectorSnapshot.from_rows
    materialization_modes: list[bool] = []

    def tracking_from_rows(
        rows: Sequence[object],
        *,
        encoder_namespace: str,
        dim: int,
        epoch: int,
        materialize_metadata: bool = False,
    ) -> VectorSnapshot:
        materialization_modes.append(materialize_metadata)
        return original_from_rows(
            rows,
            encoder_namespace=encoder_namespace,
            dim=dim,
            epoch=epoch,
            materialize_metadata=materialize_metadata,
        )

    monkeypatch.setattr(VectorSnapshot, "from_rows", tracking_from_rows)
    assert [
        hit.source_id
        for hit in vector.search([1.0, 0.0], 3, model_name="model", dim=2)
    ] == ["drop", "keep-first", "keep-second"]

    filters = {"metadata_equals": {"category": "keep"}}
    warm_hits = vector.search(
        [1.0, 0.0], 3, model_name="model", dim=2, filters=filters
    )
    repeated_hits = vector.search(
        [1.0, 0.0], 3, model_name="model", dim=2, filters=filters
    )

    assert [hit.source_id for hit in warm_hits] == ["keep-first", "keep-second"]
    assert [hit.storage_key for hit in repeated_hits] == [
        hit.storage_key for hit in warm_hits
    ]
    assert [hit.score for hit in repeated_hits] == pytest.approx(
        [hit.score for hit in warm_hits]
    )
    assert materialization_modes == [False]


def test_large_mixed_vector_upsert_writes_only_changed_rows(tmp_path: Path) -> None:
    """Large batches skip unchanged rows while repairing changed vector state."""
    backend = LocalRecordBackend(tmp_path / "records.db")
    records = [_record(f"record-{index}", [1.0, 0.0]) for index in range(901)]
    backend.upsert(records, "model", 2)
    before = backend.vector_epoch()

    original_revision = record_embedding_revision(records[0], "model", 2)
    records[0].body = "changed semantic input"
    records[1].embedding = [0.0, 1.0]
    connection = backend.db_manager.get_connection()
    connection.execute(
        """
        UPDATE local_vectors_v2
        SET embedding = ?, revision = ?, format_version = ?,
            normalization_policy = ?
        WHERE storage_key = ?
        """,
        (
            b"corrupt",
            record_embedding_revision(records[2], "model", 2),
            1,
            "legacy",
            records[2].storage_key,
        ),
    )
    connection.commit()
    queries: list[str] = []
    connection.set_trace_callback(queries.append)

    backend.upsert(records, "model", 2)

    vector_writes = [
        query for query in queries if "local_vectors_v2" in query and "INSERT" in query
    ]
    assert len(vector_writes) == 3
    assert backend.vector_epoch() == before + 1
    assert backend.search_vector(
        [0.0, 1.0], 1, model_name="model", dim=2
    )[0].source_id == "record-1"
    changed_revision = connection.execute(
        "SELECT revision FROM local_vectors_v2 WHERE storage_key = ?",
        (records[0].storage_key,),
    ).fetchone()[0]
    assert changed_revision != original_revision
    repaired = connection.execute(
        """
        SELECT embedding, revision, format_version, normalization_policy
        FROM local_vectors_v2
        WHERE storage_key = ?
        """,
        (records[2].storage_key,),
    ).fetchone()
    assert records[2].embedding is not None
    assert tuple(repaired) == (
        PackedVectorCodec.encode(records[2].embedding, 2),
        record_embedding_revision(records[2], "model", 2),
        2,
        "l2",
    )
    connection.set_trace_callback(None)


def test_public_vector_batches_preserve_bounds_order_and_required_rows() -> None:
    """Public vector batches stay bounded, ordered, and fully projected.

    The iterator must expose the rows needed by optional index builders while
    preserving the existing deterministic keyset pagination contract.
    """
    backend = LocalRecordBackend(
        vector_snapshot_max_rows=2,
        vector_snapshot_max_bytes=2 * 4 * 2,
    )
    records = [_record(f"record-{index}", [1.0, 0.0]) for index in range(5)]
    backend.upsert(records, "model", 2)
    backend.upsert([_record("other-model", [0.0, 1.0])], "other", 2)

    batches = list(backend.iter_vector_batches("model", 2))
    row_fields = {
        "storage_key",
        "workspace_id",
        "source_kind",
        "source_id",
        "status",
        "metadata",
        "uri",
        "embedding",
        "format_version",
        "normalization_policy",
    }

    assert [len(batch) for batch in batches] == [2, 2, 1]
    assert [
        row["storage_key"] for batch in batches for row in batch
    ] == sorted(record.storage_key for record in records)
    assert all(set(row.keys()) == row_fields for batch in batches for row in batch)


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


def test_auto_vector_engine_calibrates_exact_engines_once(monkeypatch) -> None:
    """Auto routing selects the faster exact engine after one calibration."""
    backend = LocalRecordBackend(faiss_threshold=1)
    record = _record("one", [1.0, 0.0])
    backend.upsert([record], "model", 2)
    vector = LocalVectorStore(backend, engine="auto")

    class _FakeFAISS:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.last_search_diagnostics = {"fallback": False}
            self.calls = 0

        def search(
            self,
            query_vector: list[float],
            k: int,
            *,
            model_name: str,
            dim: int,
            filters: dict[str, object] | None = None,
            compiled_filter: object | None = None,
        ) -> list[RecordHit]:
            self.calls += 1
            return [RecordHit(record.identity, 1.0)]

    monkeypatch.setattr(local_indices, "FAISSLocalVectorStore", _FakeFAISS)
    clock_values = iter((0.0, 0.010, 0.010, 0.015))
    monkeypatch.setattr(local_indices.time, "perf_counter", lambda: next(clock_values))

    first = vector.search([1.0, 0.0], 1, model_name="model", dim=2)
    second = vector.search([1.0, 0.0], 1, model_name="model", dim=2)

    assert [hit.source_id for hit in first] == ["one"]
    assert [hit.source_id for hit in second] == ["one"]
    assert vector.engine_name == "faiss"
    assert vector.last_routing_measurement is not None
    assert vector.last_routing_measurement.selected == "faiss"
    assert vector.last_routing_measurement.faiss_ms < (
        vector.last_routing_measurement.sqlite_ms
    )


def test_auto_vector_engine_pins_sqlite_when_exact_results_differ(monkeypatch) -> None:
    """Auto routing rejects a FAISS probe that changes exact result identity."""
    backend = LocalRecordBackend(faiss_threshold=1)
    records = [_record("one", [1.0, 0.0]), _record("two", [0.0, 1.0])]
    backend.upsert(records, "model", 2)
    vector = LocalVectorStore(backend, engine="auto")

    class _MismatchedFAISS:
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
            compiled_filter: object | None = None,
        ) -> list[RecordHit]:
            return [RecordHit(records[1].identity, 1.0)]

    monkeypatch.setattr(local_indices, "FAISSLocalVectorStore", _MismatchedFAISS)
    clock_values = iter((0.0, 0.010, 0.010, 0.015))
    monkeypatch.setattr(local_indices.time, "perf_counter", lambda: next(clock_values))

    hits = vector.search([1.0, 0.0], 1, model_name="model", dim=2)

    assert [hit.source_id for hit in hits] == ["one"]
    assert vector.engine_name == "sqlite-exact"
    assert vector.last_routing_measurement is not None
    assert vector.last_routing_measurement.selected == "sqlite-exact"


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
    assert restored.search_vector(
        [1.0, 0.0], 1, model_name="model", dim=2
    )[0].source_id == "one"
    restored.delete([records[0].storage_key])
    assert [
        hit.source_id
        for hit in restored.search_vector(
            [1.0, 0.0], 2, model_name="model", dim=2
        )
    ] == ["two"]


def test_optional_faiss_recall_reload_and_corruption_fallback(tmp_path: Path) -> None:
    """FAISS rebuilds from persisted vectors when its artifact is corrupt.

    A fresh store must recover searchable state from the durable vector rows.
    """
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
    with pytest.raises(error):
        _invalid_faiss_configuration(field, value)


def _invalid_faiss_configuration(field: str, value: object) -> None:
    if field == "hnsw_m":
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError("hnsw_m must be an integer")
        FAISSConfiguration(hnsw_m=value)
    elif field == "hnsw_ef_construction":
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError("hnsw_ef_construction must be an integer")
        FAISSConfiguration(hnsw_ef_construction=value)
    elif field == "hnsw_ef_search":
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError("hnsw_ef_search must be an integer")
        FAISSConfiguration(hnsw_ef_search=value)
    elif field == "overfetch_multiplier":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError("overfetch_multiplier must be numeric")
        FAISSConfiguration(overfetch_multiplier=value)
    elif field == "max_scan_rounds":
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError("max_scan_rounds must be an integer")
        FAISSConfiguration(max_scan_rounds=value)
    elif field == "max_scan_candidates":
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError("max_scan_candidates must be an integer")
        FAISSConfiguration(max_scan_candidates=value)
    else:
        raise AssertionError(f"unsupported FAISS configuration field: {field}")


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
    assert metadata["build_fingerprint"] == store.configuration.build_fingerprint
    assert metadata["query_policy_fingerprint"] == (
        store.configuration.query_policy_fingerprint
    )

    reloaded = FAISSLocalVectorStore(
        backend,
        index_path=tmp_path / "faiss",
        search_strategy="approximate",
        hnsw_m=12,
        hnsw_ef_construction=27,
        hnsw_ef_search=73,
    )
    assert reloaded.search(
        [1.0, 0.0], 1, model_name="model", dim=2
    )[0].source_id == "one"
    assert reloaded.last_search_diagnostics["persistence"] == "loaded"


def test_faiss_query_policy_reload_reuses_index_and_updates_diagnostics(
    tmp_path: Path,
) -> None:
    """Query-policy changes reload the persisted artifact without rebuilding it."""
    pytest.importorskip("faiss")
    backend = LocalRecordBackend(tmp_path / "records.db")
    backend.upsert([_record("one", [1.0, 0.0])], "model", 2)
    index_path = tmp_path / "faiss"
    original = FAISSLocalVectorStore(backend, index_path=index_path)
    original.search([1.0, 0.0], 1, model_name="model", dim=2)

    reloaded = FAISSLocalVectorStore(
        backend,
        index_path=index_path,
        overfetch_multiplier=8.0,
    )
    assert reloaded.search([1.0, 0.0], 1, model_name="model", dim=2)

    diagnostics = reloaded.last_search_diagnostics
    assert diagnostics["persistence"] == "loaded"
    assert diagnostics["build_fingerprint"] == original.configuration.build_fingerprint
    assert diagnostics["query_policy_fingerprint"] == (
        reloaded.configuration.query_policy_fingerprint
    )
    assert diagnostics["query_policy_fingerprint"] != (
        original.configuration.query_policy_fingerprint
    )


def test_faiss_persistence_compacts_candidate_metadata_and_round_trips_filters(
    tmp_path: Path,
) -> None:
    """Reloaded FAISS state preserves metadata filtering and ordering."""
    pytest.importorskip("faiss")
    backend = LocalRecordBackend(tmp_path / "records.db")
    backend.upsert(
        [
            _record("allowed", [1.0, 0.0], metadata={"project": "a"}),
            _record("blocked", [0.9, 0.1], metadata={"project": "b"}),
        ],
        "model",
        2,
    )
    index_path = tmp_path / "faiss"
    original = FAISSLocalVectorStore(backend, index_path=index_path)
    filters = {"metadata_equals": {"project": "a"}}

    expected = original.search(
        [1.0, 0.0], 2, model_name="model", dim=2, filters=filters
    )
    metadata_path = next(index_path.glob("*.json"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    reloaded = FAISSLocalVectorStore(backend, index_path=index_path)
    actual = reloaded.search(
        [1.0, 0.0], 2, model_name="model", dim=2, filters=filters
    )

    assert isinstance(metadata["candidate_metadata"], list)
    assert len(metadata["candidate_metadata"]) == len(metadata["storage_keys"])
    assert all("storage_key" not in value for value in metadata["candidate_metadata"])
    assert [hit.storage_key for hit in actual] == [hit.storage_key for hit in expected]
    assert [hit.score for hit in actual] == pytest.approx(
        [hit.score for hit in expected]
    )
    assert reloaded.last_search_diagnostics["persistence"] == "loaded"

    legacy_metadata = dict(metadata)
    legacy_metadata["candidate_metadata"] = dict(
        zip(metadata["storage_keys"], metadata["candidate_metadata"], strict=True)
    )
    metadata_path.write_text(json.dumps(legacy_metadata), encoding="utf-8")
    rebuilt = FAISSLocalVectorStore(backend, index_path=index_path)
    rebuilt.search([1.0, 0.0], 2, model_name="model", dim=2, filters=filters)

    assert rebuilt.last_search_diagnostics["persistence"] == "rebuilt"


def test_faiss_configuration_fingerprints_separate_build_and_query_policy() -> None:
    """Build and query-policy fingerprints change only for their own inputs."""
    base = FAISSConfiguration()
    query_policy = FAISSConfiguration(overfetch_multiplier=8.0)
    build = FAISSConfiguration(hnsw_m=12)

    assert base.build_fingerprint == query_policy.build_fingerprint
    assert base.query_policy_fingerprint != query_policy.query_policy_fingerprint
    assert base.build_fingerprint != build.build_fingerprint
    assert base.query_policy_fingerprint == build.query_policy_fingerprint
    assert len({base.fingerprint, query_policy.fingerprint, build.fingerprint}) == 3


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


def test_faiss_authorization_filters_use_exact_path_before_limit(
    tmp_path: Path,
) -> None:
    """Approximate FAISS search cannot discard authorized low-score records."""
    pytest.importorskip("faiss")
    backend = LocalRecordBackend(tmp_path / "records.db")
    records = [
        _record(
            f"blocked-{index}",
            [1.0, 0.0],
            metadata={"acl": ["blocked"]},
        )
        for index in range(3)
    ]
    records.extend(
        [
            _record("eligible-1", [0.8, 0.6], metadata={"acl": ["allowed"]}),
            _record("eligible-2", [0.6, 0.8], metadata={"acl": ["allowed"]}),
        ]
    )
    backend.upsert(records, "model", 2)
    store = FAISSLocalVectorStore(
        backend,
        index_path=tmp_path / "faiss",
        search_strategy="approximate",
        max_scan_candidates=2,
    )
    filters = {
        "source_scoped_filters": {
            "note": {"metadata_contains_any": {"acl": ["allowed"]}}
        }
    }

    hits = store.search(
        [1.0, 0.0], 2, model_name="model", dim=2, filters=filters
    )

    assert [hit.source_id for hit in hits] == ["eligible-1", "eligible-2"]
    assert store.last_search_diagnostics["strategy"] == "exact_filtered"


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


def test_snapshot_names_the_offending_row_for_a_later_corrupt_vector(
    tmp_path: Path,
) -> None:
    """A non-finite vector past the first row is still attributed correctly.

    Bulk decoding validates the whole matrix at once, so the row-to-key
    mapping in the error path is the part that can silently drift.
    """
    db_path = tmp_path / "records.db"
    backend = LocalRecordBackend(db_path)
    records = [_record("one", [1.0, 0.0]), _record("two", [0.0, 1.0])]
    backend.upsert(records, "model", 2)
    later = max(records, key=lambda record: record.storage_key)

    conn = backend.db_manager.get_connection()
    conn.execute(
        "UPDATE local_vectors_v2 SET embedding = ? WHERE storage_key = ?",
        (
            np.asarray([np.nan, 0.0], dtype="<f4").tobytes(),
            later.storage_key,
        ),
    )
    conn.commit()

    reopened = LocalRecordBackend(db_path)
    with pytest.raises(ValueError, match="must contain only finite values") as error:
        reopened.search_vector([1.0, 0.0], 1, model_name="model", dim=2)
    assert later.storage_key in str(error.value)


def test_bounded_vector_batch_matches_scalar_order_scores_and_ties() -> None:
    """Batch search preserves scalar ordering, scores, and deterministic ties."""
    backend = LocalRecordBackend()
    records = [
        _record("b", [1.0, 0.0]),
        _record("a", [1.0, 0.0]),
        _record("c", [0.0, 1.0]),
    ]
    backend.upsert(records, "model", 2)
    queries = [[1.0, 0.0], [0.0, 1.0]]

    actual = backend.search_vector_batch(queries, 3, model_name="model", dim=2)
    expected = [
        backend.search_vector(query, 3, model_name="model", dim=2)
        for query in queries
    ]

    assert [[hit.storage_key for hit in hits] for hits in actual] == [
        [hit.storage_key for hit in hits] for hits in expected
    ]
    assert [[hit.score for hit in hits] for hits in actual] == [
        pytest.approx([hit.score for hit in hits]) for hits in expected
    ]


def test_bounded_vector_batch_handles_empty_and_non_positive_k() -> None:
    """Empty input and non-positive k return one result per requested query."""
    backend = LocalRecordBackend()
    backend.upsert([_record("one", [1.0, 0.0])], "model", 2)

    assert backend.search_vector_batch([], 1, model_name="model", dim=2) == []
    assert backend.search_vector_batch(
        [[1.0, 0.0], [0.0, 1.0]], 0, model_name="model", dim=2
    ) == [[], []]


def test_bounded_vector_batch_falls_back_for_typed_filter_and_oversized_batch() -> None:
    """Unsupported filters and query counts retain scalar filtering behavior."""
    backend = LocalRecordBackend()
    records = [
        _record("kept", [1.0, 0.0], workspace_id="workspace"),
        _record("hidden", [1.0, 0.0], workspace_id="other"),
    ]
    backend.upsert(records, "model", 2)
    filters = {"workspace_id": "workspace"}

    typed = backend.search_vector_batch(
        [[1.0, 0.0]], 2, model_name="model", dim=2, filters=filters
    )
    oversized = backend.search_vector_batch(
        [[1.0, 0.0]] * 65, 2, model_name="model", dim=2, filters=filters
    )

    assert [hit.source_id for hit in typed[0]] == ["kept"]
    assert all([hit.source_id for hit in hits] == ["kept"] for hits in oversized)


def test_bounded_vector_batch_rejects_malformed_query() -> None:
    """Malformed query vectors retain scalar validation errors with context."""
    backend = LocalRecordBackend()
    backend.upsert([_record("one", [1.0, 0.0])], "model", 2)

    with pytest.raises(ValueError, match="query vector 1"):
        backend.search_vector_batch(
            [[1.0, 0.0], [float("nan"), 0.0]],
            1,
            model_name="model",
            dim=2,
        )
