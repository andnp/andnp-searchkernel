import json
from datetime import UTC, datetime
from pathlib import Path

from searchkernel.domain import Record
from searchkernel.indices import FAISSLocalVectorStore, LocalRecordBackend


def _record(source_id: str, embedding: list[float]) -> Record:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    return Record(
        source_kind="note",
        source_id=source_id,
        title=source_id,
        body=source_id,
        created_at=timestamp,
        updated_at=timestamp,
        embedding=embedding,
    )


def test_duplicate_persisted_storage_keys_force_a_rebuild(tmp_path: Path) -> None:
    """Duplicate persisted keys are rejected instead of collapsing search hits."""
    backend = LocalRecordBackend(tmp_path / "records.db")
    records = [_record("one", [1.0, 0.0]), _record("two", [0.0, 1.0])]
    backend.upsert(records, "model", 2)
    index_path = tmp_path / "index.faiss"

    store = FAISSLocalVectorStore(backend, index_path=index_path)
    assert [hit.source_id for hit in store.search([1.0, 0.0], 2, model_name="model", dim=2)] == [
        "one",
        "two",
    ]

    metadata_path = index_path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["storage_keys"][1] = metadata["storage_keys"][0]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    restored = FAISSLocalVectorStore(backend, index_path=index_path)
    hits = restored.search([1.0, 0.0], 2, model_name="model", dim=2)

    assert [hit.source_id for hit in hits] == ["one", "two"]
    assert restored.last_search_diagnostics["persistence"] == "rebuilt"
