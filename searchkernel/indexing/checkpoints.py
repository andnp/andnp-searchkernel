"""Durable source cursor stores for asynchronous ingestion."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from searchkernel.domain import Cursor
from searchkernel.utils.atomic_io import atomic_write_json


class JsonCheckpointStore:
    """Persist composite source cursors in one atomically replaced JSON file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    async def load(
        self, source_kind: str, workspace_id: str | None = None
    ) -> Cursor:
        return await asyncio.to_thread(self._load, source_kind, workspace_id)

    async def save(
        self,
        source_kind: str,
        workspace_id: str | None,
        checkpoint: Cursor,
    ) -> None:
        await asyncio.to_thread(
            self._save, source_kind, workspace_id, checkpoint
        )

    def _load(self, source_kind: str, workspace_id: str | None) -> Cursor:
        if not self.path.exists():
            return None
        data = json.loads(self.path.read_text(encoding="utf-8"))
        key = _checkpoint_key(source_kind, workspace_id)
        value = data.get(key)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"checkpoint for {key!r} must be a string or null")
        return value

    def _save(
        self, source_kind: str, workspace_id: str | None, checkpoint: Cursor
    ) -> None:
        data: dict[str, str | None] = {}
        if self.path.exists():
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError("checkpoint store must contain a JSON object")
            data = loaded
        data[_checkpoint_key(source_kind, workspace_id)] = checkpoint
        atomic_write_json(self.path, data)


class MemoryCheckpointStore:
    """Explicit in-memory checkpoint store for tests and process-local callers."""

    def __init__(self) -> None:
        self.values: dict[tuple[str, str | None], Cursor] = {}

    async def load(
        self, source_kind: str, workspace_id: str | None = None
    ) -> Cursor:
        return self.values.get((source_kind, workspace_id))

    async def save(
        self,
        source_kind: str,
        workspace_id: str | None,
        checkpoint: Cursor,
    ) -> None:
        self.values[(source_kind, workspace_id)] = checkpoint


def _checkpoint_key(source_kind: str, workspace_id: str | None) -> str:
    return json.dumps(
        [workspace_id, source_kind],
        ensure_ascii=False,
        separators=(",", ":"),
    )


__all__ = ["JsonCheckpointStore", "MemoryCheckpointStore"]
