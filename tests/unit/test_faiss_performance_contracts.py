from datetime import UTC, datetime
from pathlib import Path

from searchkernel.domain import Record, RecordStatus
from searchkernel.indices import FAISSLocalVectorStore, LocalRecordBackend


def _record(
    source_id: str,
    embedding: list[float],
    *,
    metadata: dict[str, object] | None = None,
    status: RecordStatus = RecordStatus.ACTIVE,
) -> Record:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    return Record(
        source_kind="note",
        source_id=source_id,
        title=source_id,
        body=source_id,
        created_at=timestamp,
        updated_at=timestamp,
        metadata=metadata or {},
        status=status,
        embedding=embedding,
    )


def test_exact_unfiltered_search_returns_active_top_k_in_order(tmp_path: Path) -> None:
    """Return exact active results in descending score order.

    The default ACTIVE status contract remains intact when inactive vectors
    have higher scores than some active results.
    """
    backend = LocalRecordBackend(tmp_path / "records.db")
    backend.upsert(
        [
            _record("archived", [1.0, 0.0], status=RecordStatus.ARCHIVED),
            _record("best", [0.95, 0.3122499]),
            _record("next", [0.8, 0.6]),
            _record("last", [0.6, 0.8]),
        ],
        "model",
        2,
    )

    hits = FAISSLocalVectorStore(backend).search(
        [1.0, 0.0], 2, model_name="model", dim=2
    )

    assert [hit.source_id for hit in hits] == ["best", "next"]


def test_exact_filtered_search_preserves_filtering_and_order(tmp_path: Path) -> None:
    """Apply metadata filters before returning exact top-k results.

    A filtered result below an ineligible higher-scoring vector must still be
    found and ordered by its exact score.
    """
    backend = LocalRecordBackend(tmp_path / "records.db")
    backend.upsert(
        [
            _record("excluded", [1.0, 0.0], metadata={"keep": False}),
            _record("first", [0.9, 0.4358899], metadata={"keep": True}),
            _record("second", [0.7, 0.7141421], metadata={"keep": True}),
        ],
        "model",
        2,
    )

    hits = FAISSLocalVectorStore(backend).search(
        [1.0, 0.0],
        2,
        model_name="model",
        dim=2,
        filters={"metadata_equals": {"keep": True}},
    )

    assert [hit.source_id for hit in hits] == ["first", "second"]


def test_faiss_fallback_matches_local_for_compound_filters(
    tmp_path: Path, monkeypatch
) -> None:
    """FAISS fallback preserves local exact filtering and ordering.

    A forced index failure exercises the fallback contract without depending
    on FAISS availability or internal compilation details.
    """
    backend = LocalRecordBackend(tmp_path / "records.db")
    records = [
        _record("first", [1.0, 0.0], metadata={"keep": True}),
        _record("second", [0.8, 0.6], metadata={"keep": True}),
        _record("blocked", [0.99, 0.1], metadata={"keep": False}),
    ]
    backend.upsert(records, "model", 2)
    filters = {
        "candidate_storage_keys": [
            records[0].storage_key,
            records[1].storage_key,
            records[2].storage_key,
        ],
        "metadata_equals": {"keep": True},
    }
    store = FAISSLocalVectorStore(backend, index_path=tmp_path / "faiss")
    monkeypatch.setattr(
        store,
        "_get_state",
        lambda model_name, dim: (_ for _ in ()).throw(RuntimeError("index failed")),
    )

    expected = backend.search_vector(
        [1.0, 0.0], 2, model_name="model", dim=2, filters=filters
    )
    actual = store.search(
        [1.0, 0.0], 2, model_name="model", dim=2, filters=filters
    )

    assert [(hit.storage_key, hit.score) for hit in actual] == [
        (hit.storage_key, hit.score) for hit in expected
    ]
    assert store.last_search_diagnostics["fallback"] is True


def test_exact_search_returns_empty_for_empty_corpus(tmp_path: Path) -> None:
    """Return no hits when the exact FAISS corpus is empty.

    An empty index must not produce placeholder or sentinel results.
    """
    backend = LocalRecordBackend(tmp_path / "records.db")

    hits = FAISSLocalVectorStore(backend).search(
        [1.0, 0.0], 5, model_name="model", dim=2
    )

    assert hits == []


def test_exact_search_returns_short_corpus_without_padding(tmp_path: Path) -> None:
    """Return every available exact hit when k exceeds the corpus size.

    The result count must be bounded by the available records.
    """
    backend = LocalRecordBackend(tmp_path / "records.db")
    backend.upsert(
        [_record("first", [1.0, 0.0]), _record("second", [0.0, 1.0])],
        "model",
        2,
    )

    hits = FAISSLocalVectorStore(backend).search(
        [1.0, 0.0], 5, model_name="model", dim=2
    )

    assert [hit.source_id for hit in hits] == ["first", "second"]


def test_faiss_persistence_round_trip_preserves_search_results(tmp_path: Path) -> None:
    """Persisted FAISS state returns the same public search results.

    A new store instance must recover both the index ranking and the persisted
    candidate metadata needed by a filtered search.
    """
    backend = LocalRecordBackend(tmp_path / "records.db")
    records = [
        _record("first", [1.0, 0.0], metadata={"keep": True}),
        _record("second", [0.8, 0.6], metadata={"keep": False}),
    ]
    backend.upsert(records, "model", 2)
    index_path = tmp_path / "index.faiss"

    first_store = FAISSLocalVectorStore(backend, index_path=index_path)
    first_hits = first_store.search([1.0, 0.0], 2, model_name="model", dim=2)
    restored_hits = FAISSLocalVectorStore(backend, index_path=index_path).search(
        [1.0, 0.0], 2, model_name="model", dim=2
    )
    filtered_hits = FAISSLocalVectorStore(backend, index_path=index_path).search(
        [1.0, 0.0], 2, model_name="model", dim=2,
        filters={"metadata_equals": {"keep": True}},
    )

    assert [hit.source_id for hit in restored_hits] == [hit.source_id for hit in first_hits]
    assert [hit.score for hit in restored_hits] == [hit.score for hit in first_hits]
    assert [hit.source_id for hit in filtered_hits] == ["first"]
