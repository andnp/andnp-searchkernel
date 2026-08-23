"""Fallback and validation tests for local vector batch scoring."""

from datetime import UTC, datetime

import pytest

from searchkernel.domain import Record
from searchkernel.indices import LocalRecordBackend


def _record(source_id: str, embedding: list[float], *, project: str = "one") -> Record:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    return Record(
        workspace_id="workspace",
        source_kind="note",
        source_id=source_id,
        title=source_id,
        body=source_id,
        created_at=timestamp,
        updated_at=timestamp,
        metadata={"project_id": project},
        embedding=embedding,
    )


def _assert_batch_matches_scalar(
    backend: LocalRecordBackend,
    queries: list[list[float]],
    *,
    filters: dict[str, object] | None = None,
) -> None:
    batched = backend.search_vector_batch(
        queries, 2, model_name="model", dim=2, filters=filters
    )
    scalar = [
        backend.search_vector(
            query, 2, model_name="model", dim=2, filters=filters
        )
        for query in queries
    ]
    assert [[hit.storage_key for hit in result] for result in batched] == [
        [hit.storage_key for hit in result] for result in scalar
    ]
    for batched_result, scalar_result in zip(batched, scalar, strict=True):
        assert [hit.score for hit in batched_result] == pytest.approx(
            [hit.score for hit in scalar_result], abs=2e-6
        )


def test_batch_search_matches_scalar_results_for_shared_scalar_filter() -> None:
    """A shared scalar filter uses the same eligible records for every query."""
    backend = LocalRecordBackend()
    backend.upsert(
        [_record("first", [1.0, 0.0]), _record("second", [0.0, 1.0])],
        "model",
        2,
    )

    _assert_batch_matches_scalar(
        backend,
        [[1.0, 0.0], [0.0, 1.0]],
        filters={"workspace_id": "workspace"},
    )


def test_batch_search_falls_back_for_typed_filters() -> None:
    """Typed metadata filters preserve scalar behavior instead of using GEMM."""
    backend = LocalRecordBackend()
    backend.upsert(
        [
            _record("included", [1.0, 0.0], project="included"),
            _record("excluded", [0.0, 1.0], project="excluded"),
        ],
        "model",
        2,
    )

    _assert_batch_matches_scalar(
        backend,
        [[1.0, 0.0], [0.0, 1.0]],
        filters={"project_ids": ["included"]},
    )


def test_batch_search_falls_back_for_oversized_batches() -> None:
    """Batches above the bound remain correct through scalar execution."""
    backend = LocalRecordBackend()
    backend.upsert([_record("one", [1.0, 0.0])], "model", 2)
    queries = [[1.0, 0.0] for _ in range(65)]

    _assert_batch_matches_scalar(backend, queries)


def test_batch_search_falls_back_for_oversized_snapshots() -> None:
    """Snapshot-ineligible stores preserve results through scalar block search."""
    backend = LocalRecordBackend(vector_snapshot_max_rows=1)
    backend.upsert(
        [_record("one", [1.0, 0.0]), _record("two", [0.0, 1.0])],
        "model",
        2,
    )

    _assert_batch_matches_scalar(backend, [[1.0, 0.0], [0.0, 1.0]])


@pytest.mark.parametrize(
    "queries",
    [[[1.0]], [[1.0, 0.0, 3.0]]],
)
def test_batch_search_validates_query_dimensions(
    queries: list[list[float]],
) -> None:
    """Invalid query dimensions use the scalar validation contract."""
    backend = LocalRecordBackend()

    with pytest.raises(ValueError, match="dimension mismatch"):
        backend.search_vector_batch(queries, 1, model_name="model", dim=2)


def test_batch_search_returns_empty_results_for_nonpositive_k() -> None:
    """Nonpositive k returns one empty result per supplied query."""
    backend = LocalRecordBackend()

    assert backend.search_vector_batch(
        [[1.0, 0.0], [0.0, 1.0]], 0, model_name="model", dim=2
    ) == [[], []]
