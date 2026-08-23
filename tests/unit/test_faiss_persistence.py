import json
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

    manifest_path = index_path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["storage_keys"][1] = manifest["storage_keys"][0]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    restored = FAISSLocalVectorStore(backend, index_path=index_path)
    hits = restored.search([1.0, 0.0], 2, model_name="model", dim=2)

    assert [hit.source_id for hit in hits] == ["one", "two"]
    assert restored.last_search_diagnostics["persistence"] == "rebuilt"


def _promote_split_artifact_to_legacy(index_path: Path) -> None:
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
    manifest_path.unlink()


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
    FAISSLocalVectorStore(backend, index_path=index_path).search(
        [1.0, 0.0], 2, model_name="model", dim=2
    )
    _promote_split_artifact_to_legacy(index_path)

    restored = FAISSLocalVectorStore(backend, index_path=index_path)
    hits = restored.search([1.0, 0.0], 2, model_name="model", dim=2)

    assert [hit.source_id for hit in hits] == ["one", "two"]
    assert restored.last_search_diagnostics["persistence"] == "loaded"


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
    FAISSLocalVectorStore(backend, index_path=index_path).search(
        [1.0, 0.0], 3, model_name="model", dim=2
    )
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
    FAISSLocalVectorStore(backend, index_path=index_path).search(
        [1.0, 0.0], 1, model_name="model", dim=2
    )
    manifest_path = index_path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["active_ids"].append(999)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    restored = FAISSLocalVectorStore(backend, index_path=index_path)
    hits = restored.search([1.0, 0.0], 1, model_name="model", dim=2)

    assert [hit.source_id for hit in hits] == ["one"]
    assert restored.last_search_diagnostics["persistence"] == "rebuilt"


def test_truncated_split_sidecar_rebuilds_deterministically(tmp_path: Path) -> None:
    """Rebuild when a published generation sidecar is incomplete.

    A manifest may outlive a partial sidecar write, but its recorded size
    makes the loader reject that generation and use canonical backend data.
    """
    backend = LocalRecordBackend(tmp_path / "records.db")
    backend.upsert([_record("one", [1.0, 0.0])], "model", 2)
    index_path = tmp_path / "index.faiss"
    FAISSLocalVectorStore(backend, index_path=index_path).search(
        [1.0, 0.0], 1, model_name="model", dim=2
    )
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
    FAISSLocalVectorStore(backend, index_path=index_path).search(
        [1.0, 0.0], 3, model_name="model", dim=2
    )
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
    FAISSLocalVectorStore(backend, index_path=index_path).search(
        [1.0, 0.0], 1, model_name="model", dim=2
    )
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
