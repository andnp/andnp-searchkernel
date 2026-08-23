"""Core behavior tests for local vector batch scoring."""

from datetime import UTC, datetime

import pytest

from searchkernel.domain import Record
from searchkernel.indices import LocalRecordBackend


def _record(source_id: str, embedding: list[float]) -> Record:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    return Record(
        workspace_id="workspace",
        source_kind="note",
        source_id=source_id,
        title=source_id,
        body=source_id,
        created_at=timestamp,
        updated_at=timestamp,
        embedding=embedding,
    )


def test_batch_search_matches_scalar_results_for_unfiltered_queries() -> None:
    """The local batch entry point preserves scalar hit identity and scores."""
    backend = LocalRecordBackend()
    records = [
        _record("first", [1.0, 0.0]),
        _record("second", [0.0, 1.0]),
        _record("diagonal", [1.0, 1.0]),
    ]
    backend.upsert(records, "model", 2)
    queries = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]

    batched = backend.search_vector_batch(queries, 2, model_name="model", dim=2)
    scalar = [
        backend.search_vector(query, 2, model_name="model", dim=2)
        for query in queries
    ]

    assert [[hit.storage_key for hit in result] for result in batched] == [
        [hit.storage_key for hit in result] for result in scalar
    ]
    for batched_result, scalar_result in zip(batched, scalar, strict=True):
        assert [hit.score for hit in batched_result] == pytest.approx(
            [hit.score for hit in scalar_result], abs=2e-6
        )


def test_batch_search_preserves_storage_key_order_for_score_ties() -> None:
    """Equal scores use the same deterministic storage-key ordering as scalar search."""
    backend = LocalRecordBackend()
    records = [
        _record("zulu", [1.0, 0.0]),
        _record("alpha", [1.0, 0.0]),
        _record("middle", [0.0, 1.0]),
    ]
    backend.upsert(records, "model", 2)

    batched = backend.search_vector_batch(
        [[1.0, 0.0], [0.0, 1.0]], 2, model_name="model", dim=2
    )
    scalar = [
        backend.search_vector(query, 2, model_name="model", dim=2)
        for query in ([1.0, 0.0], [0.0, 1.0])
    ]

    assert [[hit.storage_key for hit in result] for result in batched] == [
        [hit.storage_key for hit in result] for result in scalar
    ]
