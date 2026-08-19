"""Prepared record batches for indexing coordination."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from searchkernel.domain import Chunk, Record


@dataclass
class PreparedIndexRecord:
    file_path: str
    parser: object
    record: Record
    chunks: list[Chunk]
    graph_metadata: dict


def iter_prepared_index_batches(
    records: Iterable[PreparedIndexRecord],
    *,
    max_records: int,
    max_chunks: int,
) -> Iterator[list[PreparedIndexRecord]]:
    """Yield non-empty record batches bounded by record and chunk counts."""
    if max_records <= 0 or max_chunks <= 0:
        raise ValueError("batch bounds must be positive")

    current: list[PreparedIndexRecord] = []
    chunk_count = 0
    for record in records:
        record_chunks = len(record.chunks)
        if current and (
            len(current) >= max_records or chunk_count + record_chunks > max_chunks
        ):
            yield current
            current = []
            chunk_count = 0
        current.append(record)
        chunk_count += record_chunks
    if current:
        yield current
