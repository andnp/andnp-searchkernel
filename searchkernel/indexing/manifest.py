import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from searchkernel.domain.reindex import ActiveModelMetadata, MigrationState
from searchkernel.utils.atomic_io import atomic_write_json

CURRENT_MANIFEST_SPEC_VERSION = "2.0.0"


@dataclass
class IndexManifest:
    spec_version: str
    embedding_model: str
    chunking_config: dict[str, Any]
    indexed_files: dict[str, str] | None = None  # doc_id -> relative_file_path
    active_model: ActiveModelMetadata | None = None
    migration: MigrationState | None = None


def save_manifest(path: Path, manifest: IndexManifest) -> None:
    manifest_path = path / "index.manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "spec_version": manifest.spec_version,
        "embedding_model": manifest.embedding_model,
        "chunking_config": manifest.chunking_config,
        "indexed_files": manifest.indexed_files or {},
    }
    if manifest.active_model is not None:
        data["active_model"] = manifest.active_model.to_dict()
    if manifest.migration is not None:
        data["migration"] = manifest.migration.to_dict()

    atomic_write_json(manifest_path, data)


def load_manifest(path: Path):
    manifest_path = path / "index.manifest.json"

    if not manifest_path.exists():
        return None

    try:
        with manifest_path.open("r") as f:
            data = json.load(f)

        return IndexManifest(
            spec_version=data["spec_version"],
            embedding_model=data["embedding_model"],
            chunking_config=data.get("chunking_config", {}),
            indexed_files=data.get("indexed_files"),
            active_model=(
                ActiveModelMetadata.from_dict(data["active_model"])
                if data.get("active_model") is not None
                else None
            ),
            migration=(
                MigrationState.from_dict(data["migration"])
                if data.get("migration") is not None
                else None
            ),
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def should_rebuild(current: IndexManifest, saved: IndexManifest | None):
    if saved is None:
        return True

    # Missing indexed_files triggers a one-time rebuild to populate it
    if saved.indexed_files is None:
        return True

    return (
        current.spec_version != saved.spec_version
        or current.embedding_model != saved.embedding_model
        or current.chunking_config != saved.chunking_config
    )
