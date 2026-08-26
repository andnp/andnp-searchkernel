"""Incremental FAISS state updates must match a full rebuild exactly."""

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from searchkernel.domain import Record, RecordStatus
from searchkernel.indices import FAISSLocalVectorStore, LocalRecordBackend

DIM = 16
CORPUS = 40
# efSearch above the corpus size makes HNSW exhaustive, so an approximate
# index is directly comparable with a rebuild instead of merely similar.
EF_SEARCH = 128


def _vectors(seed: int, count: int) -> list[list[float]]:
    raw = np.random.default_rng(seed).standard_normal((count, DIM))
    unit = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    return [[float(value) for value in row] for row in unit]


def _record(
    source_id: str,
    body: str,
    embedding: list[float],
    *,
    status: RecordStatus = RecordStatus.ACTIVE,
) -> Record:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    return Record(
        workspace_id="workspace",
        source_kind="note",
        source_id=source_id,
        title=source_id,
        body=body,
        created_at=timestamp,
        updated_at=timestamp,
        metadata={"name": source_id},
        status=status,
        embedding=embedding,
    )


def _seeded_store(
    tmp_path: Path, strategy: str
) -> tuple[LocalRecordBackend, FAISSLocalVectorStore, list[Record]]:
    """Return a backend and a store whose state is already built and persisted."""
    backend = LocalRecordBackend(tmp_path / "records.db")
    records = [
        _record(f"seed-{index}", f"seed body {index}", embedding)
        for index, embedding in enumerate(_vectors(1, CORPUS))
    ]
    backend.upsert(records, "model", DIM)
    store = FAISSLocalVectorStore(
        backend,
        index_path=tmp_path / "index.faiss",
        search_strategy=strategy,
        hnsw_ef_search=EF_SEARCH,
    )
    store.search(_vectors(2, 1)[0], 5, model_name="model", dim=DIM)
    return backend, store, records


def _rebuilt_hits(
    backend: LocalRecordBackend, strategy: str, query: list[float], k: int
) -> list[tuple[str, float]]:
    """Search a store with no prior state, forcing a from-scratch build."""
    fresh = FAISSLocalVectorStore(
        backend, search_strategy=strategy, hnsw_ef_search=EF_SEARCH
    )
    hits = fresh.search(query, k, model_name="model", dim=DIM)
    assert fresh.last_search_diagnostics["persistence"] == "rebuilt"
    return [(hit.storage_key, hit.score) for hit in hits]


@pytest.mark.parametrize("strategy", ["exact", "approximate"])
def test_added_vectors_match_a_full_rebuild(tmp_path: Path, strategy: str) -> None:
    """New vectors reach search results without re-adding the whole corpus."""
    backend, store, _ = _seeded_store(tmp_path, strategy)
    backend.upsert(
        [
            _record(f"added-{index}", f"added body {index}", embedding)
            for index, embedding in enumerate(_vectors(3, 5))
        ],
        "model",
        DIM,
    )
    query = _vectors(2, 1)[0]

    hits = store.search(query, 8, model_name="model", dim=DIM)

    assert store.last_search_diagnostics["persistence"] == "updated"
    assert store.last_search_diagnostics["incremental_added"] == 5
    assert store.last_search_diagnostics["incremental_replaced"] == 0
    assert [(hit.storage_key, hit.score) for hit in hits] == _rebuilt_hits(
        backend, strategy, query, 8
    )


@pytest.mark.parametrize("strategy", ["exact", "approximate"])
def test_changed_vectors_match_a_full_rebuild(tmp_path: Path, strategy: str) -> None:
    """A re-embedded record scores against its new vector, never its old one."""
    backend, store, records = _seeded_store(tmp_path, strategy)
    replacements = _vectors(4, 3)
    backend.upsert(
        [
            _record(records[index].source_id, f"changed body {index}", embedding)
            for index, embedding in enumerate(replacements)
        ],
        "model",
        DIM,
    )
    query = replacements[0]

    hits = store.search(query, 8, model_name="model", dim=DIM)

    assert [(hit.storage_key, hit.score) for hit in hits] == _rebuilt_hits(
        backend, strategy, query, 8
    )
    assert len({hit.storage_key for hit in hits}) == len(hits)


@pytest.mark.parametrize("strategy", ["exact", "approximate"])
def test_removed_vectors_match_a_full_rebuild(tmp_path: Path, strategy: str) -> None:
    """Deleted records disappear from results even though HNSW cannot remove."""
    backend, store, records = _seeded_store(tmp_path, strategy)
    removed = [record.storage_key for record in records[:4]]
    backend.delete(removed)
    query = records[0].embedding
    assert query is not None

    hits = store.search(list(query), 8, model_name="model", dim=DIM)

    assert store.last_search_diagnostics["persistence"] == "updated"
    assert store.last_search_diagnostics["incremental_tombstoned"] == 4
    assert not {hit.storage_key for hit in hits} & set(removed)
    assert [(hit.storage_key, hit.score) for hit in hits] == _rebuilt_hits(
        backend, strategy, list(query), 8
    )


@pytest.mark.parametrize("strategy", ["exact", "approximate"])
def test_combined_changes_match_a_full_rebuild(tmp_path: Path, strategy: str) -> None:
    """Additions, replacements and deletions in one epoch stay equivalent."""
    backend, store, records = _seeded_store(tmp_path, strategy)
    replacements = _vectors(5, 3)
    backend.upsert(
        [
            _record(f"added-{index}", f"added body {index}", embedding)
            for index, embedding in enumerate(_vectors(6, 4))
        ],
        "model",
        DIM,
    )
    backend.upsert(
        [
            _record(records[index].source_id, f"changed body {index}", embedding)
            for index, embedding in enumerate(replacements)
        ],
        "model",
        DIM,
    )
    backend.delete([record.storage_key for record in records[30:33]])
    query = _vectors(2, 1)[0]

    hits = store.search(query, 10, model_name="model", dim=DIM)

    assert [(hit.storage_key, hit.score) for hit in hits] == _rebuilt_hits(
        backend, strategy, query, 10
    )


@pytest.mark.parametrize("strategy", ["exact", "approximate"])
def test_deleted_key_reindexed_with_new_text_is_not_duplicated(
    tmp_path: Path, strategy: str
) -> None:
    """A tombstoned identifier must never carry two vectors in one index.

    The stable identifier survives deletion inside the index, so re-adding the
    same storage key has to replace the tombstoned vector rather than append
    a second copy under the same identifier.
    """
    backend, store, records = _seeded_store(tmp_path, strategy)
    revived = records[0]
    backend.delete([revived.storage_key])
    store.search(_vectors(2, 1)[0], 5, model_name="model", dim=DIM)
    new_embedding = _vectors(7, 1)[0]
    backend.upsert(
        [_record(revived.source_id, "revived body", new_embedding)], "model", DIM
    )

    hits = store.search(new_embedding, 8, model_name="model", dim=DIM)

    assert hits[0].storage_key == revived.storage_key
    assert len({hit.storage_key for hit in hits}) == len(hits)
    assert [(hit.storage_key, hit.score) for hit in hits] == _rebuilt_hits(
        backend, strategy, new_embedding, 8
    )


def test_changed_vector_forces_a_rebuild_for_the_approximate_index(
    tmp_path: Path,
) -> None:
    """HNSW cannot drop the superseded vector, so a replacement rebuilds.

    `IndexIDMap2.add_with_ids` appends a duplicate for an identifier that is
    already present and `IndexHNSWFlat.remove_ids` is unimplemented, so the
    only correct in-place option is a full rebuild.
    """
    backend, store, records = _seeded_store(tmp_path, "approximate")
    embedding = _vectors(8, 1)[0]
    backend.upsert(
        [_record(records[0].source_id, "changed body", embedding)], "model", DIM
    )

    store.search(embedding, 5, model_name="model", dim=DIM)

    assert store.last_search_diagnostics["persistence"] == "rebuilt"


def test_accumulated_tombstones_trigger_a_compacting_rebuild(
    tmp_path: Path,
) -> None:
    """Tombstones past the compaction ratio rebuild instead of accumulating."""
    backend, store, records = _seeded_store(tmp_path, "approximate")
    query = _vectors(2, 1)[0]

    backend.delete([record.storage_key for record in records[:4]])
    store.search(query, 5, model_name="model", dim=DIM)
    assert store.last_search_diagnostics["persistence"] == "updated"

    backend.delete([record.storage_key for record in records[4:16]])
    hits = store.search(query, 5, model_name="model", dim=DIM)

    assert store.last_search_diagnostics["persistence"] == "rebuilt"
    assert [(hit.storage_key, hit.score) for hit in hits] == _rebuilt_hits(
        backend, "approximate", query, 5
    )


def test_persisted_state_supports_a_later_incremental_diff(tmp_path: Path) -> None:
    """A restart diffs against the persisted generation, not a rebuild.

    Persisted revisions are what make the diff survive process boundaries, so
    a store constructed from scratch must still update rather than rebuild.
    """
    backend, _, _ = _seeded_store(tmp_path, "exact")
    backend.upsert(
        [_record("added", "added body", _vectors(9, 1)[0])], "model", DIM
    )
    query = _vectors(2, 1)[0]

    restarted = FAISSLocalVectorStore(
        backend, index_path=tmp_path / "index.faiss", search_strategy="exact"
    )
    hits = restarted.search(query, 8, model_name="model", dim=DIM)

    assert restarted.last_search_diagnostics["persistence"] == "updated"
    assert restarted.last_search_diagnostics["incremental_added"] == 1
    assert [(hit.storage_key, hit.score) for hit in hits] == _rebuilt_hits(
        backend, "exact", query, 8
    )


def test_manifest_without_a_state_version_is_rejected_into_a_rebuild(
    tmp_path: Path,
) -> None:
    """An older persisted generation rebuilds rather than being misread.

    Pre-incremental manifests carry no revisions, so loading one as if it did
    would silently skip changed vectors.
    """
    backend, _, _ = _seeded_store(tmp_path, "exact")
    manifest_path = (tmp_path / "index.faiss").with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["faiss_state_version"]
    del manifest["revisions"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    query = _vectors(2, 1)[0]

    restarted = FAISSLocalVectorStore(
        backend, index_path=tmp_path / "index.faiss", search_strategy="exact"
    )
    hits = restarted.search(query, 5, model_name="model", dim=DIM)

    assert restarted.last_search_diagnostics["persistence"] == "rebuilt"
    assert [(hit.storage_key, hit.score) for hit in hits] == _rebuilt_hits(
        backend, "exact", query, 5
    )


def test_tombstones_ahead_of_the_top_hit_do_not_shrink_recall(
    tmp_path: Path,
) -> None:
    """Deleted vectors crowding the top of the ranking must not hide live ones.

    Tombstoned vectors stay resident in the index and are discarded only after
    the search returns them, so a scan budget sized from the live key count
    can exhaust its rounds on tombstones and under-return.
    """
    backend = LocalRecordBackend(tmp_path / "records.db")
    angles = np.arange(CORPUS) * 0.02
    records = [
        _record(f"seed-{index}", f"seed body {index}", [float(np.cos(a)), float(np.sin(a))])
        for index, a in enumerate(angles)
    ]
    backend.upsert(records, "model", 2)
    store = FAISSLocalVectorStore(backend, index_path=tmp_path / "index.faiss")
    store.search([1.0, 0.0], 1, model_name="model", dim=2)
    backend.delete([record.storage_key for record in records[:9]])

    hits = store.search([1.0, 0.0], 1, model_name="model", dim=2)

    assert store.last_search_diagnostics["persistence"] == "updated"
    assert [hit.storage_key for hit in hits] == [records[9].storage_key]
