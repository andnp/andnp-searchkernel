from datetime import UTC, datetime
from pathlib import Path

import pytest

from searchkernel.domain import Record, RecordStatus
from searchkernel.indices import FAISSLocalVectorStore, LocalRecordBackend


def _record(
    source_id: str,
    embedding: list[float],
    *,
    metadata: dict[str, object] | None = None,
    status: RecordStatus = RecordStatus.ACTIVE,
    workspace_id: str | None = "workspace",
    source_kind: str = "note",
) -> Record:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    return Record(
        workspace_id=workspace_id,
        source_kind=source_kind,
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


def test_exact_canonical_scalar_filter_matches_local_order(tmp_path: Path) -> None:
    """Canonical scalar filtering preserves local identity and score order."""
    backend = LocalRecordBackend(tmp_path / "records.db")
    records = [
        _record("best", [1.0, 0.0]),
        _record("commit", [0.99, 0.1], source_kind="commit"),
        _record("other-workspace", [0.98, 0.2], workspace_id="other"),
        _record("next", [0.8, 0.6]),
    ]
    backend.upsert(records, "model", 2)
    filters = {"workspace_id": "workspace", "source_kind": "note"}

    expected = backend.search_vector(
        [1.0, 0.0], 10, model_name="model", dim=2, filters=filters
    )
    actual = FAISSLocalVectorStore(backend).search(
        [1.0, 0.0], 10, model_name="model", dim=2, filters=filters
    )

    assert [hit.storage_key for hit in actual] == [
        hit.storage_key for hit in expected
    ]
    assert [hit.score for hit in actual] == pytest.approx(
        [hit.score for hit in expected]
    )


def test_canonical_scalar_filter_refreshes_after_vector_epoch_change(
    tmp_path: Path,
) -> None:
    """A changed vector epoch removes records from the prior eligibility state."""
    backend = LocalRecordBackend(tmp_path / "records.db")
    target = _record("target", [1.0, 0.0])
    remaining = _record("remaining", [0.8, 0.6])
    backend.upsert([target, remaining], "model", 2)
    store = FAISSLocalVectorStore(backend, index_path=tmp_path / "faiss")
    filters = {"workspace_id": "workspace", "source_kind": "note"}

    assert [
        hit.source_id
        for hit in store.search(
            [1.0, 0.0], 10, model_name="model", dim=2, filters=filters
        )
    ] == ["target", "remaining"]

    backend.upsert(
        [_record("target", [0.9, 0.4358899], status=RecordStatus.ARCHIVED)],
        "model",
        2,
    )

    assert [
        hit.source_id
        for hit in store.search(
            [1.0, 0.0], 10, model_name="model", dim=2, filters=filters
        )
    ] == ["remaining"]


def test_persisted_reload_rebuilds_canonical_scalar_eligibility(
    tmp_path: Path,
) -> None:
    """A loaded FAISS state preserves canonical filtering without persisted masks."""
    backend = LocalRecordBackend(tmp_path / "records.db")
    backend.upsert(
        [
            _record("allowed", [1.0, 0.0]),
            _record("wrong-kind", [0.9, 0.1], source_kind="commit"),
        ],
        "model",
        2,
    )
    index_path = tmp_path / "faiss"
    filters = {"workspace_id": "workspace", "source_kind": "note"}
    original = FAISSLocalVectorStore(backend, index_path=index_path)
    expected = original.search(
        [1.0, 0.0], 10, model_name="model", dim=2, filters=filters
    )

    restored = FAISSLocalVectorStore(backend, index_path=index_path)
    actual = restored.search(
        [1.0, 0.0], 10, model_name="model", dim=2, filters=filters
    )

    assert [hit.storage_key for hit in actual] == [
        hit.storage_key for hit in expected
    ]
    assert restored.last_search_diagnostics["persistence"] == "loaded"


def test_approximate_canonical_filter_keeps_scan_budget_semantics(
    tmp_path: Path,
) -> None:
    """Canonical filtering does not expand approximate scan limits."""
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
        filters={"workspace_id": "workspace", "source_kind": "note"},
    )

    diagnostics = store.last_search_diagnostics
    assert len(hits) < 2
    assert diagnostics["candidate_budget"] == 2
    assert diagnostics["candidate_budget_hit"] is True
    assert diagnostics["under_returned"] is True


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


class _EmbeddingReadCountingBackend(LocalRecordBackend):
    """Count the stored embeddings a FAISS state refresh actually reads."""

    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.embeddings_read = 0

    def iter_vector_batches(self, model_name: str, dim: int):  # type: ignore[no-untyped-def]
        for rows in super().iter_vector_batches(model_name, dim):
            self.embeddings_read += len(rows)
            yield rows

    def iter_vector_embedding_batches(  # type: ignore[no-untyped-def]
        self, model_name: str, dim: int, storage_keys
    ):
        for rows in super().iter_vector_embedding_batches(
            model_name, dim, storage_keys
        ):
            self.embeddings_read += len(rows)
            yield rows


def test_a_single_new_vector_does_not_reread_the_whole_corpus(
    tmp_path: Path,
) -> None:
    """Refreshing a stale FAISS state costs the change, not the corpus.

    Re-adding every vector on each advanced vector epoch is the dominant query
    cost on a continuously indexed corpus, so a one-record change must read a
    bounded number of stored embeddings while returning rebuild-equal results.
    """
    backend = _EmbeddingReadCountingBackend(tmp_path / "records.db")
    corpus = [
        _record(f"seed-{index}", [1.0 - index / 100.0, index / 100.0])
        for index in range(40)
    ]
    backend.upsert(corpus, "model", 2)
    store = FAISSLocalVectorStore(backend, index_path=tmp_path / "index.faiss")
    store.search([1.0, 0.0], 5, model_name="model", dim=2)
    backend.upsert([_record("added", [0.0, 1.0])], "model", 2)
    backend.embeddings_read = 0

    hits = store.search([0.0, 1.0], 5, model_name="model", dim=2)

    assert backend.embeddings_read == 1
    expected = FAISSLocalVectorStore(backend).search(
        [0.0, 1.0], 5, model_name="model", dim=2
    )
    assert [hit.storage_key for hit in hits] == [
        hit.storage_key for hit in expected
    ]
