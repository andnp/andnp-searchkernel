"""Ports for optional record chunking during ingestion."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from searchkernel.domain import Chunk, Record


@runtime_checkable
class RecordChunker(Protocol):
    """Split a record into deterministic, retrievable chunks."""

    def chunk_record(self, record: Record) -> Sequence[Chunk]:
        ...


__all__ = ["RecordChunker"]
