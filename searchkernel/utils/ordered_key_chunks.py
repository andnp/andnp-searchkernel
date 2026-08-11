"""Utilities for bounded, ordered key batches."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

DEFAULT_KEY_CHUNK_LIMIT = 900


def iter_ordered_key_chunks(
    keys: Iterable[str], *, limit: int = DEFAULT_KEY_CHUNK_LIMIT
) -> Iterator[tuple[str, ...]]:
    """Yield unique keys in input order using bounded chunks."""
    if limit < 1:
        raise ValueError("chunk limit must be positive")

    seen: set[str] = set()
    chunk: list[str] = []
    for key in keys:
        if key in seen:
            continue
        seen.add(key)
        chunk.append(key)
        if len(chunk) == limit:
            yield tuple(chunk)
            chunk = []
    if chunk:
        yield tuple(chunk)
