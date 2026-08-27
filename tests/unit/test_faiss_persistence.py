import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from searchkernel.domain import Record, RecordHit, RecordStatus
from searchkernel.indices import (
    FAISSLocalVectorStore,
    LocalRecordBackend,
    faiss_local,
)


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


def _publish(
    backend: LocalRecordBackend,
    index_path: Path,
    query: list[float],
    k: int,
) -> None:
    """Publish one generation from a fresh store and wait for it to be on disk.

    Publication runs on a background writer, so a test that reads the artifact
    has to wait for that writer rather than assume the search wrote it.
    """
    store = FAISSLocalVectorStore(backend, index_path=index_path)
    store.search(query, k, model_name="model", dim=2)
    assert store.flush_persistence() is True


def test_duplicate_persisted_storage_keys_force_a_rebuild(tmp_path: Path) -> None:
    """Reject duplicate generation keys instead of collapsing search hits.

    A malformed atomically published manifest must rebuild from the canonical
    backend rather than return ambiguous identities.
    """
    backend = LocalRecordBackend(tmp_path / "records.db")
    records = [_record("one", [1.0, 0.0]), _record("two", [0.0, 1.0])]
    backend.upsert(records, "model", 2)
    index_path = tmp_path / "index.faiss"

    store = FAISSLocalVectorStore(backend, index_path=index_path)
    assert [hit.source_id for hit in store.search([1.0, 0.0], 2, model_name="model", dim=2)] == [
        "one",
        "two",
    ]
    assert store.flush_persistence() is True

    manifest_path = index_path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["storage_keys"][1] = manifest["storage_keys"][0]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    restored = FAISSLocalVectorStore(backend, index_path=index_path)
    hits = restored.search([1.0, 0.0], 2, model_name="model", dim=2)

    assert [hit.source_id for hit in hits] == ["one", "two"]
    assert restored.last_search_diagnostics["persistence"] == "rebuilt"


def _write_legacy_artifact(index_path: Path) -> None:
    """Write the pre-split fixed files beside the published generation."""
    manifest_path = index_path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    generation_index = index_path.with_name(manifest["index_file"])
    generation_metadata = index_path.with_name(manifest["metadata_file"])
    legacy = dict(manifest)
    legacy.pop("persistence_format")
    legacy.pop("generation")
    legacy.pop("index_file")
    legacy.pop("metadata_file")
    legacy.pop("index_size")
    legacy.pop("metadata_size")
    legacy.pop("metadata_offsets")
    legacy["candidate_metadata"] = [
        {
            key: value
            for key, value in json.loads(line).items()
            if key != "storage_key"
        }
        for line in generation_metadata.read_text(encoding="utf-8").splitlines()
    ]
    index_path.write_bytes(generation_index.read_bytes())
    index_path.with_suffix(".json").write_text(json.dumps(legacy), encoding="utf-8")


def _promote_split_artifact_to_legacy(index_path: Path) -> None:
    _write_legacy_artifact(index_path)
    index_path.with_suffix(".manifest.json").unlink()


def test_legacy_inline_artifact_remains_readable(tmp_path: Path) -> None:
    """Load the pre-split inline artifact without requiring a migration.

    Existing deployments may still have only the fixed FAISS and JSON files,
    so the loader must preserve that contract after split persistence ships.
    """
    backend = LocalRecordBackend(tmp_path / "records.db")
    backend.upsert(
        [_record("one", [1.0, 0.0]), _record("two", [0.0, 1.0])],
        "model",
        2,
    )
    index_path = tmp_path / "index.faiss"
    _publish(backend, index_path, [1.0, 0.0], 2)
    _promote_split_artifact_to_legacy(index_path)

    restored = FAISSLocalVectorStore(backend, index_path=index_path)
    hits = restored.search([1.0, 0.0], 2, model_name="model", dim=2)

    assert [hit.source_id for hit in hits] == ["one", "two"]
    assert restored.last_search_diagnostics["persistence"] == "loaded"


def test_explicit_legacy_migration_publishes_verified_split(tmp_path: Path) -> None:
    """Migrate a validated legacy artifact without changing its rollback copy.

    The explicit operation publishes and verifies split state, then a fresh
    store reloads that generation while the original inline files remain.
    """
    backend = LocalRecordBackend(tmp_path / "records.db")
    backend.upsert([_record("one", [1.0, 0.0])], "model", 2)
    index_path = tmp_path / "index.faiss"
    _publish(backend, index_path, [1.0, 0.0], 1)
    _promote_split_artifact_to_legacy(index_path)
    legacy_index = index_path.read_bytes()
    legacy_metadata = index_path.with_suffix(".json").read_bytes()

    store = FAISSLocalVectorStore(backend, index_path=index_path)
    assert store.migrate_legacy_persistence("model", 2) is True
    assert store.last_search_diagnostics["migration"] == "migrated"
    assert index_path.read_bytes() == legacy_index
    assert index_path.with_suffix(".json").read_bytes() == legacy_metadata
    assert index_path.with_suffix(".manifest.json").is_file()

    reloaded = FAISSLocalVectorStore(backend, index_path=index_path)
    hits = reloaded.search([1.0, 0.0], 1, model_name="model", dim=2)
    assert [hit.source_id for hit in hits] == ["one"]
    assert reloaded.last_search_diagnostics["persistence"] == "loaded"


def test_explicit_migration_reports_already_split_without_republishing(
    tmp_path: Path, monkeypatch
) -> None:
    """Treat an already valid split generation as an explicit no-op.

    The migration call verifies and caches the current generation without
    invoking publication a second time.
    """
    backend = LocalRecordBackend(tmp_path / "records.db")
    backend.upsert([_record("one", [1.0, 0.0])], "model", 2)
    index_path = tmp_path / "index.faiss"
    store = FAISSLocalVectorStore(backend, index_path=index_path)
    store.search([1.0, 0.0], 1, model_name="model", dim=2)
    assert store.flush_persistence() is True

    def fail_publication(state, **kwargs) -> bool:
        raise AssertionError("already split state must not republish")

    monkeypatch.setattr(store, "_persist_state", fail_publication)
    assert store.migrate_legacy_persistence("model", 2) is True
    assert store.last_search_diagnostics["migration"] == "already_split"


def test_failed_migration_preserves_legacy_fallback(
    tmp_path: Path, monkeypatch
) -> None:
    """Keep legacy loading available when split publication fails.

    A failed manifest publication may leave generation files behind, but the
    fixed legacy files remain unchanged and a fresh store still loads them.
    """
    backend = LocalRecordBackend(tmp_path / "records.db")
    backend.upsert([_record("one", [1.0, 0.0])], "model", 2)
    index_path = tmp_path / "index.faiss"
    _publish(backend, index_path, [1.0, 0.0], 1)
    _promote_split_artifact_to_legacy(index_path)
    legacy_index = index_path.read_bytes()
    legacy_metadata = index_path.with_suffix(".json").read_bytes()

    def fail_manifest(*args, **kwargs) -> None:
        raise OSError("manifest publication blocked")

    monkeypatch.setattr(
        "searchkernel.indices.faiss_local.atomic_write_json", fail_manifest
    )
    store = FAISSLocalVectorStore(backend, index_path=index_path)
    assert store.migrate_legacy_persistence("model", 2) is False
    assert store.last_search_diagnostics["migration"] == "failed"
    assert index_path.read_bytes() == legacy_index
    assert index_path.with_suffix(".json").read_bytes() == legacy_metadata
    assert not index_path.with_suffix(".manifest.json").exists()

    fallback = FAISSLocalVectorStore(backend, index_path=index_path)
    hits = fallback.search([1.0, 0.0], 1, model_name="model", dim=2)
    assert [hit.source_id for hit in hits] == ["one"]
    assert fallback.last_search_diagnostics["persistence"] == "loaded"


def test_split_sidecar_uses_one_handle_for_filtered_search(
    tmp_path: Path, monkeypatch
) -> None:
    """Resolve one filtered search through one transient sidecar handle.

    Candidate metadata is read lazily from a single operation-scoped handle;
    the implementation must not reopen the JSONL file per candidate.
    """
    backend = LocalRecordBackend(tmp_path / "records.db")
    backend.upsert(
        [
            _record("first", [1.0, 0.0], metadata={"project": "keep"}),
            _record("second", [0.9, 0.1], metadata={"project": "keep"}),
            _record("blocked", [0.8, 0.2], metadata={"project": "skip"}),
        ],
        "model",
        2,
    )
    index_path = tmp_path / "index.faiss"
    _publish(backend, index_path, [1.0, 0.0], 3)
    opened: list[Path] = []
    original_open = Path.open

    def count_sidecar_opens(path: Path, *args, **kwargs):
        if path.suffix == ".jsonl":
            opened.append(path)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", count_sidecar_opens)
    restored = FAISSLocalVectorStore(backend, index_path=index_path)
    hits = restored.search(
        [1.0, 0.0],
        3,
        model_name="model",
        dim=2,
        filters={"metadata_equals": {"project": "keep"}},
    )

    assert [hit.source_id for hit in hits] == ["first", "second"]
    assert len(opened) == 1


def test_default_active_search_avoids_split_sidecar_io(
    tmp_path: Path, monkeypatch
) -> None:
    """Use the manifest active-ID subset for the default active-only query.

    The ordinary unfiltered search must preserve inactive-record exclusion
    without opening or reading the large candidate metadata sidecar.
    """
    backend = LocalRecordBackend(tmp_path / "records.db")
    backend.upsert(
        [
            _record("inactive", [1.0, 0.0], status=RecordStatus.ARCHIVED),
            _record("active", [0.9, 0.1]),
        ],
        "model",
        2,
    )
    index_path = tmp_path / "index.faiss"
    first = FAISSLocalVectorStore(backend, index_path=index_path)
    first.search([1.0, 0.0], 2, model_name="model", dim=2)
    opened: list[Path] = []
    original_open = Path.open

    def count_sidecar_opens(path: Path, *args, **kwargs):
        if path.suffix == ".jsonl":
            opened.append(path)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", count_sidecar_opens)
    restored = FAISSLocalVectorStore(backend, index_path=index_path)
    hits = restored.search([1.0, 0.0], 2, model_name="model", dim=2)

    assert [hit.source_id for hit in hits] == ["active"]
    assert opened == []


def test_split_active_ids_must_be_persisted_subset(tmp_path: Path) -> None:
    """Rebuild when the manifest active-ID subset is inconsistent.

    Empty and all-active lists are valid explicit representations, but an ID
    absent from the persisted FAISS mapping invalidates the generation.
    """
    backend = LocalRecordBackend(tmp_path / "records.db")
    backend.upsert([_record("one", [1.0, 0.0])], "model", 2)
    index_path = tmp_path / "index.faiss"
    _publish(backend, index_path, [1.0, 0.0], 1)
    manifest_path = index_path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["active_ids"].append(999)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    restored = FAISSLocalVectorStore(backend, index_path=index_path)
    hits = restored.search([1.0, 0.0], 1, model_name="model", dim=2)

    assert [hit.source_id for hit in hits] == ["one"]
    assert restored.last_search_diagnostics["persistence"] == "rebuilt"


def test_streamed_sidecar_offsets_round_trip_exact_records(tmp_path: Path) -> None:
    """Publish one JSONL record per offset with exact manifest sizes.

    The streamed writer must preserve deterministic key-to-record locations so
    a fresh store can resolve filtered metadata without rebuilding the index.
    """
    backend = LocalRecordBackend(tmp_path / "records.db")
    backend.upsert(
        [
            _record("one", [1.0, 0.0], metadata={"project": "a"}),
            _record("two", [0.0, 1.0], metadata={"project": "b"}),
        ],
        "model",
        2,
    )
    index_path = tmp_path / "index.faiss"
    store = FAISSLocalVectorStore(backend, index_path=index_path)
    expected = store.search(
        [1.0, 0.0], 2, model_name="model", dim=2,
        filters={"metadata_equals": {"project": "a"}},
    )
    assert store.flush_persistence() is True
    manifest = json.loads(
        index_path.with_suffix(".manifest.json").read_text(encoding="utf-8")
    )
    sidecar = index_path.with_name(manifest["metadata_file"]).read_bytes()
    locations = manifest["metadata_offsets"]

    assert manifest["metadata_size"] == len(sidecar)
    previous_end = 0
    for storage_key in manifest["storage_keys"]:
        offset = locations[storage_key]["offset"]
        length = locations[storage_key]["length"]
        assert offset == previous_end
        record = json.loads(sidecar[offset : offset + length])
        assert record["storage_key"] == storage_key
        previous_end = offset + length

    restored = FAISSLocalVectorStore(backend, index_path=index_path)
    actual = restored.search(
        [1.0, 0.0], 2, model_name="model", dim=2,
        filters={"metadata_equals": {"project": "a"}},
    )
    assert [(hit.storage_key, hit.score) for hit in actual] == [
        (hit.storage_key, hit.score) for hit in expected
    ]
    assert restored.last_search_diagnostics["persistence"] == "loaded"


def test_truncated_split_sidecar_rebuilds_deterministically(tmp_path: Path) -> None:
    """Rebuild when a published generation sidecar is incomplete.

    A manifest may outlive a partial sidecar write, but its recorded size
    makes the loader reject that generation and use canonical backend data.
    """
    backend = LocalRecordBackend(tmp_path / "records.db")
    backend.upsert([_record("one", [1.0, 0.0])], "model", 2)
    index_path = tmp_path / "index.faiss"
    _publish(backend, index_path, [1.0, 0.0], 1)
    manifest = json.loads(
        index_path.with_suffix(".manifest.json").read_text(encoding="utf-8")
    )
    sidecar_path = index_path.with_name(manifest["metadata_file"])
    sidecar_path.write_bytes(sidecar_path.read_bytes()[:-1])

    restored = FAISSLocalVectorStore(backend, index_path=index_path)
    hits = restored.search([1.0, 0.0], 1, model_name="model", dim=2)

    assert [hit.source_id for hit in hits] == ["one"]
    assert restored.last_search_diagnostics["persistence"] == "rebuilt"


def test_split_filtered_selectivity_preserves_backend_parity(tmp_path: Path) -> None:
    """Preserve filtered parity for both selective and broad predicates.

    High-selectivity authorization and low-selectivity metadata filters both
    resolve through the same bounded per-operation sidecar path.
    """
    backend = LocalRecordBackend(tmp_path / "records.db")
    records = [
        _record(
            "allowed",
            [1.0, 0.0],
            metadata={"project": "a", "acl": ["allowed"]},
        ),
        _record(
            "also-allowed",
            [0.9, 0.1],
            metadata={"project": "a", "acl": ["denied"]},
        ),
        _record(
            "blocked",
            [0.8, 0.2],
            metadata={"project": "b", "acl": ["denied"]},
        ),
    ]
    backend.upsert(records, "model", 2)
    index_path = tmp_path / "index.faiss"
    _publish(backend, index_path, [1.0, 0.0], 3)
    restored = FAISSLocalVectorStore(backend, index_path=index_path)
    filters = (
        {
            "source_scoped_filters": {
                "note": {"metadata_contains_any": {"acl": ["allowed"]}}
            }
        },
        {"metadata_equals": {"project": "a"}},
    )

    for current_filters in filters:
        expected = backend.search_vector(
            [1.0, 0.0], 3, model_name="model", dim=2, filters=current_filters
        )
        actual = restored.search(
            [1.0, 0.0], 3, model_name="model", dim=2, filters=current_filters
        )
        assert [(hit.storage_key, hit.score) for hit in actual] == [
            (hit.storage_key, hit.score) for hit in expected
        ]


def test_split_manifest_fingerprint_and_offset_mismatch_rebuilds(
    tmp_path: Path,
) -> None:
    """Reject generations whose fingerprint or offsets no longer validate.

    Manifest publication binds the FAISS index and JSONL sidecar to one build;
    altered generation metadata must fall back to a fresh canonical rebuild.
    """
    backend = LocalRecordBackend(tmp_path / "records.db")
    backend.upsert([_record("one", [1.0, 0.0])], "model", 2)
    index_path = tmp_path / "index.faiss"
    _publish(backend, index_path, [1.0, 0.0], 1)
    manifest_path = index_path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["build_fingerprint"] = "wrong"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    fingerprint_fallback = FAISSLocalVectorStore(backend, index_path=index_path)
    fingerprint_fallback.search([1.0, 0.0], 1, model_name="model", dim=2)
    assert fingerprint_fallback.last_search_diagnostics["persistence"] == "rebuilt"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    key = manifest["storage_keys"][0]
    manifest["metadata_offsets"][key]["offset"] = -1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    offset_fallback = FAISSLocalVectorStore(backend, index_path=index_path)
    offset_fallback.search([1.0, 0.0], 1, model_name="model", dim=2)
    assert offset_fallback.last_search_diagnostics["persistence"] == "rebuilt"


class _ManualClock:
    """A monotonic clock the test advances explicitly."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _persisted_generations(index_path: Path) -> int:
    """Count published generations; every persisted write publishes one."""
    return len(list(index_path.parent.glob(f"{index_path.stem}.*.faiss")))


def _seeded_debounce_store(
    tmp_path: Path, clock: _ManualClock
) -> tuple[LocalRecordBackend, FAISSLocalVectorStore]:
    """Return a backend and a store whose first generation is on disk."""
    backend = LocalRecordBackend(tmp_path / "records.db")
    backend.upsert(
        [
            _record(f"seed-{index}", [1.0 - index / 100.0, index / 100.0])
            for index in range(8)
        ],
        "model",
        2,
    )
    store = FAISSLocalVectorStore(
        backend, index_path=tmp_path / "index.faiss", clock=clock
    )
    store.search([1.0, 0.0], 3, model_name="model", dim=2)
    assert store.flush_persistence() is True
    return backend, store


def _add_vector(backend: LocalRecordBackend, index: int) -> None:
    backend.upsert(
        [_record(f"added-{index}", [index / 100.0, 1.0 - index / 100.0])],
        "model",
        2,
    )


def _assert_matches_rebuild(
    hits: list[RecordHit], backend: LocalRecordBackend, query: list[float], k: int
) -> None:
    """Assert results agree with a store that has no persisted state at all.

    Scores are compared to float32 precision rather than bit for bit because
    a refreshed index accumulates inner products in a different order than a
    from-scratch build.
    """
    expected = FAISSLocalVectorStore(backend).search(
        query, k, model_name="model", dim=2
    )
    assert [hit.storage_key for hit in hits] == [
        hit.storage_key for hit in expected
    ]
    assert [hit.score for hit in hits] == pytest.approx(
        [hit.score for hit in expected], rel=1e-6
    )


def test_repeated_refreshes_inside_the_window_persist_once(tmp_path: Path) -> None:
    """Frequent incremental refreshes must not each rewrite the artifact.

    Persisting per refresh dominates disk cost on a continuously indexed
    corpus and stalls queries for the length of the write.
    """
    clock = _ManualClock()
    index_path = tmp_path / "index.faiss"
    backend, store = _seeded_debounce_store(tmp_path, clock)
    assert _persisted_generations(index_path) == 1

    for index in range(3):
        _add_vector(backend, index)
        clock.advance(1.0)
        store.search([0.0, 1.0], 3, model_name="model", dim=2)
        assert store.last_search_diagnostics["persistence"] == "updated"
        assert store.last_search_diagnostics["persistence_written"] is False

    assert _persisted_generations(index_path) == 1


def test_an_elapsed_window_persists_the_next_refresh(tmp_path: Path) -> None:
    """A trickle of changes still reaches disk once the window expires."""
    clock = _ManualClock()
    index_path = tmp_path / "index.faiss"
    backend, store = _seeded_debounce_store(tmp_path, clock)

    _add_vector(backend, 0)
    clock.advance(faiss_local._PERSIST_INTERVAL_SECONDS)
    store.search([0.0, 1.0], 3, model_name="model", dim=2)

    assert store.last_search_diagnostics["persistence"] == "updated"
    assert store.last_search_diagnostics["persistence_written"] is True
    assert store.flush_persistence() is True
    assert _persisted_generations(index_path) == 2


def test_accumulated_changes_persist_before_the_window_elapses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A burst of new vectors is written without waiting out the timer."""
    monkeypatch.setattr(faiss_local, "_PERSIST_PENDING_VECTORS", 2)
    clock = _ManualClock()
    index_path = tmp_path / "index.faiss"
    backend, store = _seeded_debounce_store(tmp_path, clock)

    _add_vector(backend, 0)
    clock.advance(1.0)
    store.search([0.0, 1.0], 3, model_name="model", dim=2)
    assert _persisted_generations(index_path) == 1

    _add_vector(backend, 1)
    clock.advance(1.0)
    store.search([0.0, 1.0], 3, model_name="model", dim=2)

    assert store.last_search_diagnostics["persistence_written"] is True
    assert store.flush_persistence() is True
    assert _persisted_generations(index_path) == 2


def test_flush_persistence_writes_a_debounced_state(tmp_path: Path) -> None:
    """An explicit flush publishes a generation a later store can load."""
    clock = _ManualClock()
    index_path = tmp_path / "index.faiss"
    backend, store = _seeded_debounce_store(tmp_path, clock)
    _add_vector(backend, 0)
    clock.advance(1.0)
    store.search([0.0, 1.0], 3, model_name="model", dim=2)
    assert _persisted_generations(index_path) == 1

    assert store.flush_persistence() is True

    assert _persisted_generations(index_path) == 2
    restarted = FAISSLocalVectorStore(backend, index_path=index_path)
    restarted.search([0.0, 1.0], 3, model_name="model", dim=2)
    assert restarted.last_search_diagnostics["persistence"] == "loaded"


def test_a_debounced_state_serves_rebuild_equal_results(tmp_path: Path) -> None:
    """Deferring the write must not change what a search returns."""
    clock = _ManualClock()
    backend, store = _seeded_debounce_store(tmp_path, clock)
    for index in range(3):
        _add_vector(backend, index)
    clock.advance(1.0)

    hits = store.search([0.0, 1.0], 5, model_name="model", dim=2)

    assert store.last_search_diagnostics["persistence_written"] is False
    _assert_matches_rebuild(hits, backend, [0.0, 1.0], 5)


def test_a_fresh_store_recovers_an_unwritten_window(tmp_path: Path) -> None:
    """Losing the deferred write costs a refresh, never a correct result.

    The persisted artifact is a rebuildable cache of vectors the backend
    already holds, so a store that starts against a stale generation must
    never serve it as current.
    """
    clock = _ManualClock()
    index_path = tmp_path / "index.faiss"
    backend, store = _seeded_debounce_store(tmp_path, clock)
    for index in range(3):
        _add_vector(backend, index)
        clock.advance(1.0)
        store.search([0.0, 1.0], 3, model_name="model", dim=2)

    restarted = FAISSLocalVectorStore(backend, index_path=index_path, clock=clock)
    hits = restarted.search([0.0, 1.0], 5, model_name="model", dim=2)

    assert restarted.last_search_diagnostics["persistence"] != "loaded"
    _assert_matches_rebuild(hits, backend, [0.0, 1.0], 5)


def test_a_store_with_no_persisted_artifact_rebuilds(tmp_path: Path) -> None:
    """A corpus that was never persisted still searches correctly."""
    clock = _ManualClock()
    index_path = tmp_path / "index.faiss"
    backend, _ = _seeded_debounce_store(tmp_path, clock)
    for path in tmp_path.glob("index.*"):
        path.unlink()

    restarted = FAISSLocalVectorStore(backend, index_path=index_path, clock=clock)
    hits = restarted.search([1.0, 0.0], 5, model_name="model", dim=2)

    assert restarted.last_search_diagnostics["persistence"] == "rebuilt"
    _assert_matches_rebuild(hits, backend, [1.0, 0.0], 5)


def _publish_generations(
    backend: LocalRecordBackend,
    store: FAISSLocalVectorStore,
    clock: _ManualClock,
    count: int,
) -> None:
    """Publish ``count`` further generations, one per elapsed window."""
    for index in range(count):
        _add_vector(backend, index)
        clock.advance(faiss_local._PERSIST_INTERVAL_SECONDS)
        store.search([0.0, 1.0], 3, model_name="model", dim=2)
        assert store.last_search_diagnostics["persistence_written"] is True
        assert store.flush_persistence() is True


def test_repeated_publications_bound_the_generations_on_disk(
    tmp_path: Path,
) -> None:
    """Publishing must not accumulate a superseded artifact per write.

    Every write serialises the whole index and its sidecar, so a store that
    never removes superseded generations fills the state directory in
    proportion to how often it indexes rather than how much it holds.
    """
    clock = _ManualClock()
    index_path = tmp_path / "index.faiss"
    backend, store = _seeded_debounce_store(tmp_path, clock)

    _publish_generations(backend, store, clock, 6)

    assert _persisted_generations(index_path) == faiss_local._RETAINED_GENERATIONS
    assert len(list(tmp_path.glob("index.*.jsonl"))) == (
        faiss_local._RETAINED_GENERATIONS
    )


def test_a_pruned_directory_still_loads_its_published_generation(
    tmp_path: Path,
) -> None:
    """The generation the manifest names must survive every prune.

    Deleting it would silently cost a full rebuild on the next start, which
    is the expense the persisted artifact exists to avoid.
    """
    clock = _ManualClock()
    index_path = tmp_path / "index.faiss"
    backend, store = _seeded_debounce_store(tmp_path, clock)
    _publish_generations(backend, store, clock, 6)

    manifest = json.loads(
        index_path.with_suffix(".manifest.json").read_text(encoding="utf-8")
    )
    restarted = FAISSLocalVectorStore(backend, index_path=index_path, clock=clock)
    hits = restarted.search([0.0, 1.0], 5, model_name="model", dim=2)

    assert index_path.with_name(manifest["index_file"]).is_file()
    assert index_path.with_name(manifest["metadata_file"]).is_file()
    assert restarted.last_search_diagnostics["persistence"] == "loaded"
    _assert_matches_rebuild(hits, backend, [0.0, 1.0], 5)


def test_a_generation_a_cached_state_reads_is_never_pruned(
    tmp_path: Path,
) -> None:
    """Keep the sidecar a cached state resolves metadata through.

    A loaded state reads its metadata sidecar by path, lazily, per filtered
    search. Pruning the generation it names would strand that state on a
    missing file and drop every filtered search to the exact fallback.
    """
    backend = LocalRecordBackend(tmp_path / "records.db")
    backend.upsert(
        [
            _record("kept", [1.0, 0.0], metadata={"project": "keep"}),
            _record("other", [0.0, 1.0], metadata={"project": "skip"}),
        ],
        "cached",
        2,
    )
    index_path = tmp_path / "index.faiss"
    seed = FAISSLocalVectorStore(backend, index_path=index_path)
    seed.search([1.0, 0.0], 2, model_name="cached", dim=2)
    assert seed.flush_persistence() is True

    store = FAISSLocalVectorStore(backend, index_path=index_path)
    store.search([1.0, 0.0], 2, model_name="cached", dim=2)
    assert store.last_search_diagnostics["persistence"] == "loaded"
    for index in range(faiss_local._RETAINED_GENERATIONS + 1):
        backend.upsert(
            [_record(f"published-{index}", [1.0, 0.0])], "publisher", 2
        )
        store.search([1.0, 0.0], 1, model_name="publisher", dim=2)
        assert store.flush_persistence() is True

    hits = store.search(
        [1.0, 0.0],
        2,
        model_name="cached",
        dim=2,
        filters={"metadata_equals": {"project": "keep"}},
    )

    assert [hit.source_id for hit in hits] == ["kept"]
    assert store.last_search_diagnostics["fallback"] is False


def test_a_failed_prune_leaves_persistence_and_search_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A directory this process cannot fully control must stay searchable.

    Two processes share the state directory, so an unlink can fail on a
    permission or a race. Reclaiming disk is never worth failing a write or
    a query for.
    """
    def refuse_unlink(self: Path, missing_ok: bool = False) -> None:
        raise PermissionError(f"cannot remove {self}")

    monkeypatch.setattr(Path, "unlink", refuse_unlink)
    clock = _ManualClock()
    index_path = tmp_path / "index.faiss"
    backend, store = _seeded_debounce_store(tmp_path, clock)

    _publish_generations(backend, store, clock, 4)
    prune_error = store.last_search_diagnostics["prune_error"]
    hits = store.search([0.0, 1.0], 5, model_name="model", dim=2)

    assert "cannot remove" in prune_error
    assert _persisted_generations(index_path) == 5
    _assert_matches_rebuild(hits, backend, [0.0, 1.0], 5)


def test_flushing_a_state_loaded_from_disk_reports_success(tmp_path: Path) -> None:
    """A restarted store can flush without republishing what it just loaded.

    A loaded state keeps its metadata in the published sidecar rather than in
    memory, so serialising it again would fail on the first absent entry and
    report the flush as unsuccessful even though the artifact on disk already
    matches.
    """
    clock = _ManualClock()
    backend, _ = _seeded_debounce_store(tmp_path, clock)
    index_path = tmp_path / "index.faiss"
    published = _persisted_generations(index_path)

    restarted = FAISSLocalVectorStore(
        backend, index_path=index_path, clock=_ManualClock()
    )
    hits = restarted.search([1.0, 0.0], 3, model_name="model", dim=2)
    assert restarted.last_search_diagnostics["persistence"] == "loaded"

    assert restarted.flush_persistence() is True
    assert _persisted_generations(index_path) == published
    _assert_matches_rebuild(hits, backend, [1.0, 0.0], 3)


def test_a_stale_split_generation_does_not_read_the_legacy_artifact(
    tmp_path: Path, monkeypatch
) -> None:
    """An epoch the split generation cannot serve must not read legacy files.

    Only split generations are ever published and migration reads legacy to
    publish split, so a legacy artifact beside a manifest is always the older
    of the two and cannot satisfy an epoch the manifest failed. Reading it
    anyway costs a full artifact read on every epoch advance.
    """
    backend = LocalRecordBackend(tmp_path / "records.db")
    backend.upsert([_record("one", [1.0, 0.0])], "model", 2)
    index_path = tmp_path / "index.faiss"
    store = FAISSLocalVectorStore(backend, index_path=index_path)
    store.search([1.0, 0.0], 1, model_name="model", dim=2)
    assert store.flush_persistence() is True
    _write_legacy_artifact(index_path)
    backend.upsert([_record("two", [0.0, 1.0])], "model", 2)
    read_paths: list[Path] = []
    read_text = Path.read_text

    def counting_read_text(self: Path, *args: object, **kwargs: object) -> str:
        read_paths.append(Path(self))
        return read_text(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "read_text", counting_read_text)

    hits = store.search([0.0, 1.0], 2, model_name="model", dim=2)

    assert index_path.with_suffix(".manifest.json") in read_paths
    assert index_path.with_suffix(".json") not in read_paths
    assert store.last_search_diagnostics["persistence"] == "updated"
    assert [hit.source_id for hit in hits] == ["two", "one"]
