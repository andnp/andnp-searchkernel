from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from searchkernel.indexing.bootstrap_checkpoint import (
    BootstrapCheckpoint,
    BootstrapFileStamp,
)
from searchkernel.indexing.manifest import IndexManifest
from searchkernel.indexing.runtime_readiness import SearchAvailability

PublicIndexStatus = Literal["indexing", "partial", "ready"]


@dataclass(frozen=True)
class PublicIndexStateSnapshot:
    status: PublicIndexStatus
    indexed_count: int
    total_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "indexed_count": self.indexed_count,
            "total_count": self.total_count,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> PublicIndexStateSnapshot:
        return cls(
            status=_public_index_status(data["status"]),
            indexed_count=_non_negative_count(data["indexed_count"]),
            total_count=_non_negative_count(data["total_count"]),
        )


@dataclass(frozen=True)
class BootstrapReadinessSnapshot:
    total_targets: int
    durably_completed_targets: int
    loaded_indexed_count: int
    queryable: bool
    public_state: PublicIndexStateSnapshot
    availability: SearchAvailability | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "total_targets": self.total_targets,
            "durably_completed_targets": self.durably_completed_targets,
            "loaded_indexed_count": self.loaded_indexed_count,
            "queryable": self.queryable,
            "public_state": self.public_state.to_dict(),
            "availability": (
                self.availability.to_dict()
                if self.availability is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> BootstrapReadinessSnapshot:
        availability_data = data.get("availability")
        return cls(
            total_targets=_non_negative_count(data["total_targets"]),
            durably_completed_targets=_non_negative_count(
                data["durably_completed_targets"]
            ),
            loaded_indexed_count=_non_negative_count(data["loaded_indexed_count"]),
            queryable=_bool_value(data["queryable"]),
            public_state=PublicIndexStateSnapshot.from_dict(
                _mapping_value(data["public_state"], "public_state")
            ),
            availability=(
                SearchAvailability.from_dict(availability_data)
                if isinstance(availability_data, Mapping)
                else None
            ),
        )


def compute_bootstrap_completed_paths(
    checkpoint: BootstrapCheckpoint | None,
    saved_manifest: IndexManifest | None,
    target_stamps: dict[str, BootstrapFileStamp],
) -> set[str]:
    if checkpoint is None or saved_manifest is None:
        return set()

    indexed_files = saved_manifest.indexed_files or {}
    completed_paths: set[str] = set()
    for relative_path, stamp in checkpoint.completed.items():
        if relative_path not in target_stamps:
            continue
        if not stamp.matches(target_stamps[relative_path]):
            continue

        doc_id = str(Path(relative_path).with_suffix(""))
        if indexed_files.get(doc_id) != relative_path:
            continue

        completed_paths.add(relative_path)

    return completed_paths


def derive_loaded_index_state_snapshot(
    total_targets: int,
    loaded_indexed_count: int,
) -> PublicIndexStateSnapshot:
    status: PublicIndexStatus = "ready"
    if total_targets > 0 and loaded_indexed_count < total_targets:
        status = "partial"

    return PublicIndexStateSnapshot(
        status=status,
        indexed_count=loaded_indexed_count,
        total_count=total_targets,
    )


def derive_bootstrap_readiness_snapshot(
    checkpoint: BootstrapCheckpoint | None,
    saved_manifest: IndexManifest | None,
    target_stamps: dict[str, BootstrapFileStamp],
    *,
    loaded_indexed_count: int,
    queryable: bool,
    rebuild_pending: bool,
    availability: SearchAvailability | None = None,
) -> BootstrapReadinessSnapshot | None:
    if availability is None and checkpoint is not None:
        availability = checkpoint.availability

    completed_paths = compute_bootstrap_completed_paths(
        checkpoint,
        saved_manifest,
        target_stamps,
    )
    durably_completed_targets = len(completed_paths)
    total_targets = len(target_stamps)
    indexed_count = max(loaded_indexed_count, durably_completed_targets)

    if total_targets > 0:
        indexed_count = min(indexed_count, total_targets)

    if indexed_count == 0 and total_targets > 0:
        return None

    if total_targets == 0:
        status: PublicIndexStatus = "ready"
    elif indexed_count < total_targets:
        status = "partial"
    elif rebuild_pending:
        status = "indexing"
    else:
        status = "ready"

    return BootstrapReadinessSnapshot(
        total_targets=total_targets,
        durably_completed_targets=durably_completed_targets,
        loaded_indexed_count=loaded_indexed_count,
        queryable=queryable,
        public_state=PublicIndexStateSnapshot(
            status=status,
            indexed_count=indexed_count,
            total_count=total_targets,
        ),
        availability=availability,
    )


def _public_index_status(value: object) -> PublicIndexStatus:
    if value not in {"indexing", "partial", "ready"}:
        raise ValueError(f"invalid public index status: {value!r}")
    return cast(PublicIndexStatus, value)


def _non_negative_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"invalid non-negative count: {value!r}")
    return value


def _bool_value(value: object) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"invalid boolean value: {value!r}")
    return value


def _mapping_value(value: object, key: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{key} must be an object")
    return value


__all__ = [
    "BootstrapReadinessSnapshot",
    "PublicIndexStateSnapshot",
    "compute_bootstrap_completed_paths",
    "derive_bootstrap_readiness_snapshot",
    "derive_loaded_index_state_snapshot",
]