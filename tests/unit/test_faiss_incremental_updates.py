"""Incremental FAISS state updates must match a full rebuild exactly."""

import dataclasses
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from searchkernel.domain import Record, RecordHit, RecordStatus
from searchkernel.indices import (
    FAISSLocalVectorStore,
    LocalRecordBackend,
    faiss_local,
)

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
    metadata: dict[str, object] | None = None,
    uri: str | None = None,
) -> Record:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    return Record(
        workspace_id="workspace",
        uri=uri,
        source_kind="note",
        source_id=source_id,
        title=source_id,
        body=body,
        created_at=timestamp,
        updated_at=timestamp,
        metadata={"name": source_id} if metadata is None else metadata,
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


def _assert_matches_rebuild(
    hits: Sequence[RecordHit], expected: list[tuple[str, float]]
) -> None:
    """Assert a refreshed search agrees with a from-scratch build.

    Keys and their order must match exactly. Scores are compared to float32
    precision rather than bit for bit: a refreshed index adds vectors in a
    different order than a rebuild, and floating point accumulation is order
    dependent, so an inner product can differ in its final unit in the last
    place.
    """
    assert [hit.storage_key for hit in hits] == [key for key, _ in expected]
    assert [hit.score for hit in hits] == pytest.approx(
        [score for _, score in expected], rel=1e-6
    )


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
    _assert_matches_rebuild(hits, _rebuilt_hits(backend, strategy, query, 8))


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

    _assert_matches_rebuild(hits, _rebuilt_hits(backend, strategy, query, 8))
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
    _assert_matches_rebuild(hits, _rebuilt_hits(backend, strategy, list(query), 8))


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

    _assert_matches_rebuild(hits, _rebuilt_hits(backend, strategy, query, 10))


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
    _assert_matches_rebuild(hits, _rebuilt_hits(backend, strategy, new_embedding, 8))


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
    _assert_matches_rebuild(hits, _rebuilt_hits(backend, "approximate", query, 5))


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
    _assert_matches_rebuild(hits, _rebuilt_hits(backend, "exact", query, 8))


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
    _assert_matches_rebuild(hits, _rebuilt_hits(backend, "exact", query, 5))


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


def _edit_metadata_then_force_a_refresh(
    backend: LocalRecordBackend, edited: Record
) -> None:
    """Apply a metadata-only edit plus the vector change that refreshes state.

    A stored revision covers only the embedding inputs, so editing status,
    workspace or metadata alone leaves the vector epoch untouched and no
    refresh runs. A continuously indexed corpus interleaves the two, and the
    added vector is what gives the refresh a chance to miss the edit.
    """
    backend.upsert([edited], "model", DIM)
    backend.upsert(
        [_record("refresh-trigger", "refresh body", _vectors(11, 1)[0])],
        "model",
        DIM,
    )


@pytest.mark.parametrize("strategy", ["exact", "approximate"])
def test_metadata_only_edit_reaches_a_filtered_search(
    tmp_path: Path, strategy: str
) -> None:
    """A refresh must publish metadata that changed without its embedding."""
    backend, store, records = _seeded_store(tmp_path, strategy)
    target = records[0]
    assert target.embedding is not None
    _edit_metadata_then_force_a_refresh(
        backend,
        _record(
            target.source_id,
            target.body,
            list(target.embedding),
            metadata={"name": target.source_id, "tier": "gold"},
        ),
    )

    hits = store.search(
        list(target.embedding),
        CORPUS,
        model_name="model",
        dim=DIM,
        filters={"metadata_equals": {"tier": "gold"}},
    )

    assert store.last_search_diagnostics["persistence"] == "updated"
    assert [hit.storage_key for hit in hits] == [target.storage_key]


@pytest.mark.parametrize("strategy", ["exact", "approximate"])
def test_status_only_edit_withdraws_a_record_from_active_results(
    tmp_path: Path, strategy: str
) -> None:
    """Archiving without re-embedding must stop returning the record."""
    backend, store, records = _seeded_store(tmp_path, strategy)
    target = records[0]
    assert target.embedding is not None
    _edit_metadata_then_force_a_refresh(
        backend,
        _record(
            target.source_id,
            target.body,
            list(target.embedding),
            status=RecordStatus.ARCHIVED,
        ),
    )
    query = list(target.embedding)

    hits = store.search(query, 5, model_name="model", dim=DIM)

    assert store.last_search_diagnostics["persistence"] == "updated"
    assert target.storage_key not in {hit.storage_key for hit in hits}
    _assert_matches_rebuild(hits, _rebuilt_hits(backend, strategy, query, 5))


@pytest.mark.parametrize("strategy", ["exact", "approximate"])
def test_uri_only_edit_reaches_a_path_filtered_search(
    tmp_path: Path, strategy: str
) -> None:
    """A relocated record must be findable at its new path after a refresh."""
    backend, store, records = _seeded_store(tmp_path, strategy)
    target = records[0]
    assert target.embedding is not None
    _edit_metadata_then_force_a_refresh(
        backend,
        _record(
            target.source_id,
            target.body,
            list(target.embedding),
            uri="docs/moved.md",
        ),
    )

    hits = store.search(
        list(target.embedding),
        CORPUS,
        model_name="model",
        dim=DIM,
        filters={"paths": ["docs/moved.md"]},
    )

    assert store.last_search_diagnostics["persistence"] == "updated"
    assert [hit.storage_key for hit in hits] == [target.storage_key]


def test_the_refresh_fingerprint_covers_every_candidate_field() -> None:
    """A candidate field outside the fingerprint would silently go stale.

    A refresh carries an unchanged candidate forward on the strength of its
    fingerprint alone, so a field the digest omits could change without the
    candidate being rebuilt, leaving filtered search on stale values. The two
    are only ever equal by being kept equal.
    """
    assert set(faiss_local._CANDIDATE_METADATA_COLUMNS) == {
        field.name
        for field in dataclasses.fields(faiss_local._CandidateMetadata)
    }
