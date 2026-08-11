"""Focused tests for bounded key batching and embedding cache lookups."""

from pathlib import Path

import pytest

from searchkernel.indexing.embedding_cache import SQLiteEmbeddingCache
from searchkernel.utils.ordered_key_chunks import iter_ordered_key_chunks


def test_ordered_key_chunks_deduplicate_without_reordering() -> None:
    """
    Preserve first-seen order while removing repeated keys.
    """
    assert list(iter_ordered_key_chunks(["b", "a", "b", "c"], limit=2)) == [
        ("b", "a"),
        ("c",),
    ]


def test_ordered_key_chunks_return_no_batches_for_empty_input() -> None:
    """
    Avoid producing a placeholder batch for empty input.
    """
    assert list(iter_ordered_key_chunks([])) == []


def test_ordered_key_chunks_reject_nonpositive_limit() -> None:
    """
    Reject limits that cannot bound a non-empty batch.
    """
    with pytest.raises(ValueError, match="positive"):
        list(iter_ordered_key_chunks(["key"], limit=0))


def test_embedding_cache_handles_duplicate_keys_across_bounded_batches(
    tmp_path: Path,
) -> None:
    """
    Retrieve all unique vectors when the lookup spans SQLite batches.
    """
    cache = SQLiteEmbeddingCache(tmp_path / "embeddings.db", "encoder", dimension=1)
    vectors = {f"key-{index}": [float(index)] for index in range(901)}
    cache.put_many(vectors)

    requested = ["key-900", "key-0", "key-900", *vectors]

    assert cache.get_many(requested) == vectors
    assert cache.metrics.hits == len(vectors)
    assert cache.metrics.misses == 0


def test_embedding_cache_empty_lookup_is_a_fast_path(tmp_path: Path) -> None:
    """
    Return an empty result without changing cache metrics.
    """
    cache = SQLiteEmbeddingCache(tmp_path / "embeddings.db", "encoder")

    assert cache.get_many([]) == {}
    assert cache.metrics.hits == 0
    assert cache.metrics.misses == 0
