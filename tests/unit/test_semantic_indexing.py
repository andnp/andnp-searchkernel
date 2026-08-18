"""Unit tests for progressive semantic indexing subsystem."""

import json
import sqlite3
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from searchkernel.domain import Chunk, Record
from searchkernel.indexing.embedding_cache import SQLiteEmbeddingCache
from searchkernel.indexing.semantic import (
    EncoderFingerprint,
    LlamaIndexEmbeddingCacheAdapter,
    SemanticInput,
    SemanticWorkPlanner,
    embedding_identity,
    semantic_input_for_chunk,
    semantic_input_for_record,
)


def _with_hash(chunk):
    """Finalize a freshly-built domain.Chunk (test helper).

    domain.Chunk, unlike the legacy models.Chunk, does not auto-compute
    content_hash in __post_init__, and its metadata dict must stay JSON
    serializable (it flows into index/docstore persistence), so a raw
    datetime `modified_time` is normalized to ISO text.
    """
    if not chunk.content_hash:
        chunk.content_hash = chunk.compute_content_hash()
    modified_time = chunk.metadata.get("modified_time")
    if hasattr(modified_time, "isoformat"):
        chunk.metadata["modified_time"] = modified_time.isoformat()
    return chunk



def make_test_chunk(
    chunk_id: str = "chunk-1",
    doc_id: str = "doc-1",
    content: str = "test content",
    header_path: str = "Header",
) -> Chunk:
    """Create a test chunk with minimal required fields."""
    return _with_hash(Chunk(chunk_id=chunk_id, record_id=doc_id, content=content, metadata={ "header_path": header_path, "start_pos": 0, "end_pos": len(content), "file_path": "/test.md", "modified_time": datetime.now(UTC)}, chunk_index=0))


class FakeEncoder:
    """Mock encoder for testing."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.call_count = 0

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        self.call_count += 1
        # Simple mock: hash text to deterministic vector
        return [
            [float((hash(text) % 1000) / 1000) for _ in range(self.dimension)]
            for text in texts
        ]


class FakeCache:
    """Mock cache for testing."""

    def __init__(self) -> None:
        self.vectors: dict[str, Sequence[float]] = {}

    def get_many(self, content_hashes: Sequence[str]) -> Mapping[str, Sequence[float]]:
        return {
            h: self.vectors[h] for h in content_hashes if h in self.vectors
        }

    def put_many(self, vectors: Mapping[str, Sequence[float]]) -> None:
        self.vectors.update(vectors)


class FakeMaterializer:
    """Mock materializer for testing."""

    def __init__(self) -> None:
        self.materialized: dict[str, tuple[str, Sequence[float]]] = {}

    def materialize(
        self, source_id: str, vector: Sequence[float], semantic_input: SemanticInput
    ) -> None:
        self.materialized[source_id] = (semantic_input.content_hash, vector)


class TestEncoderFingerprint:
    """Test encoder fingerprint and namespace generation."""

    def test_fingerprint_namespace_is_deterministic(self) -> None:
        fp1 = EncoderFingerprint(
            model="test-model",
            version="1.0",
            dimension=384,
        )
        fp2 = EncoderFingerprint(
            model="test-model",
            version="1.0",
            dimension=384,
        )
        assert fp1.namespace == fp2.namespace

    def test_fingerprint_namespace_differs_by_model(self) -> None:
        fp1 = EncoderFingerprint(model="model-a", dimension=384)
        fp2 = EncoderFingerprint(model="model-b", dimension=384)
        assert fp1.namespace != fp2.namespace

    def test_fingerprint_namespace_differs_by_dimension(self) -> None:
        fp1 = EncoderFingerprint(model="test", dimension=384)
        fp2 = EncoderFingerprint(model="test", dimension=1024)
        assert fp1.namespace != fp2.namespace


class TestEmbeddingIdentity:
    """Test content hash computation."""

    def test_embedding_identity_is_deterministic(self) -> None:
        namespace = "test-namespace"
        text = "test content"
        h1 = embedding_identity(text, namespace)
        h2 = embedding_identity(text, namespace)
        assert h1 == h2

    def test_embedding_identity_differs_by_text(self) -> None:
        namespace = "test-namespace"
        h1 = embedding_identity("text-1", namespace)
        h2 = embedding_identity("text-2", namespace)
        assert h1 != h2

    def test_embedding_identity_differs_by_namespace(self) -> None:
        text = "test content"
        h1 = embedding_identity(text, "namespace-1")
        h2 = embedding_identity(text, "namespace-2")
        assert h1 != h2


class TestSemanticInputForChunk:
    """Test semantic input construction from chunks."""

    def test_semantic_input_from_chunk(self) -> None:
        chunk = make_test_chunk(header_path="Header 1")
        namespace = "test-ns"
        input_item = semantic_input_for_chunk(chunk, namespace)

        assert input_item.source_id == "chunk-1"
        assert input_item.text == "Header 1\n\ntest content"
        assert input_item.tier == "fine"
        assert input_item.priority == 0
        assert input_item.content_hash == embedding_identity(
            input_item.text, namespace
        )

    def test_semantic_input_with_empty_header(self) -> None:
        chunk = make_test_chunk(header_path="")
        input_item = semantic_input_for_chunk(chunk)
        assert input_item.text == "test content"

    def test_semantic_input_with_tier_and_priority(self) -> None:
        chunk = make_test_chunk()
        input_item = semantic_input_for_chunk(
            chunk, tier="coarse", priority=10
        )
        assert input_item.tier == "coarse"
        assert input_item.priority == 10


class TestSemanticInputForRecord:
    def test_semantic_input_uses_indexed_text_and_storage_key(self) -> None:
        timestamp = datetime(2026, 1, 1, tzinfo=UTC)
        record = Record(
            workspace_id="workspace",
            source_kind="note",
            source_id="note-1",
            title="Title",
            body="Raw body",
            indexed_text="Search text",
            created_at=timestamp,
            updated_at=timestamp,
        )

        input_item = semantic_input_for_record(record, "test-ns")

        assert input_item.source_id == record.storage_key
        assert input_item.text == "Title: Title\n\nSearch text"
        assert input_item.content_hash == embedding_identity(
            "Title: Title\n\nSearch text", "test-ns"
        )

    def test_semantic_input_falls_back_to_body(self) -> None:
        timestamp = datetime(2026, 1, 1, tzinfo=UTC)
        record = Record(
            "note",
            "note-1",
            "",
            "Raw body",
            timestamp,
            timestamp,
        )

        assert semantic_input_for_record(record).text == "Raw body"

    def test_title_changes_embedding_identity(self) -> None:
        timestamp = datetime(2026, 1, 1, tzinfo=UTC)
        first = Record(
            "note", "note-1", "First title", "Same body", timestamp, timestamp
        )
        second = Record(
            "note", "note-2", "Second title", "Same body", timestamp, timestamp
        )

        first_input = semantic_input_for_record(first, "test-ns")
        second_input = semantic_input_for_record(second, "test-ns")

        assert first_input.text != second_input.text
        assert first_input.content_hash != second_input.content_hash

    def test_record_inputs_share_work_for_duplicate_indexed_text(self) -> None:
        timestamp = datetime(2026, 1, 1, tzinfo=UTC)
        records = [
            Record(
                "note",
                f"note-{index}",
                "Title",
                f"Raw body {index}",
                timestamp,
                timestamp,
                indexed_text="same search text",
            )
            for index in range(2)
        ]
        inputs = [semantic_input_for_record(record) for record in records]

        plan = SemanticWorkPlanner("test-ns", dimension=3).plan(inputs, FakeCache())

        assert len(plan.groups) == 1
        group = next(iter(plan.groups.values()))
        assert [item.source_id for item in group] == [
            record.storage_key for record in records
        ]

    def test_record_inputs_reuse_cached_embedding(self) -> None:
        timestamp = datetime(2026, 1, 1, tzinfo=UTC)
        record = Record(
            "note",
            "note-1",
            "Title",
            "Raw body",
            timestamp,
            timestamp,
            indexed_text="Search text",
        )
        input_item = semantic_input_for_record(record)
        planner = SemanticWorkPlanner("test-ns", dimension=384)
        cache = FakeCache()
        encoder = FakeEncoder()

        planner.execute([input_item], cache, encoder, FakeMaterializer())
        progress = planner.execute([input_item], cache, encoder, FakeMaterializer())

        assert encoder.call_count == 1
        assert progress.cache_hits == 1
        assert progress.cache_misses == 0


class TestSemanticWorkPlanner:
    """Test semantic work planning and deduplication."""

    def test_planner_deduplicate_identical_inputs(self) -> None:
        planner = SemanticWorkPlanner("test-ns", dimension=384)
        inputs = [
            SemanticInput("chunk-1", "same content", "old-hash-a", "fine", 0),
            SemanticInput("chunk-2", "same content", "old-hash-b", "fine", 0),
        ]
        plan = planner.plan(inputs, FakeCache())

        # Both should map to the same recalculated content hash
        assert len(plan.groups) == 1
        # Get the actual hash that was calculated
        actual_hash = next(iter(plan.groups.keys()))
        assert len(plan.groups[actual_hash]) == 2

    def test_planner_handles_cache_hits(self) -> None:
        cache = FakeCache()
        # Create a proper content hash for the text
        text = "cached content"
        namespace = "test-ns"
        content_hash = embedding_identity(text, namespace)
        cache.vectors[content_hash] = [0.1, 0.2, 0.3]

        planner = SemanticWorkPlanner("test-ns", dimension=3)
        inputs = [
            SemanticInput("chunk-1", text, content_hash, "fine", 0),
        ]
        plan = planner.plan(inputs, cache)

        assert len(plan.hits) == 1
        assert content_hash in plan.hits
        assert len(plan.misses) == 0

    def test_planner_detects_cache_misses(self) -> None:
        cache = FakeCache()
        planner = SemanticWorkPlanner("test-ns", dimension=3)
        text = "uncached content"
        namespace = "test-ns"
        content_hash = embedding_identity(text, namespace)
        inputs = [
            SemanticInput("chunk-1", text, content_hash, "fine", 0),
        ]
        plan = planner.plan(inputs, cache)

        assert len(plan.hits) == 0
        assert len(plan.misses) == 1
        assert plan.misses[0].content_hash == content_hash

    def test_planner_prioritizes_coarse_tier(self) -> None:
        planner = SemanticWorkPlanner("test-ns", dimension=3)
        inputs = [
            SemanticInput("chunk-1", "same content", "hash-a", "fine", 0),
            SemanticInput("chunk-2", "same content", "hash-a", "coarse", 0),
        ]
        plan = planner.plan(inputs, FakeCache())

        # Coarse tier should be selected
        assert len(plan.misses) == 1
        assert plan.misses[0].tier == "coarse"

    def test_planner_respects_priority(self) -> None:
        planner = SemanticWorkPlanner("test-ns", dimension=3)
        inputs = [
            SemanticInput("chunk-1", "same content", "hash-a", "fine", 10),
            SemanticInput("chunk-2", "same content", "hash-a", "fine", 5),
        ]
        plan = planner.plan(inputs, FakeCache())

        # Lower priority should be used
        assert plan.misses[0].priority == 5

    def test_planner_executes_with_cache_hits(self) -> None:
        cache = FakeCache()
        text = "cached"
        namespace = "test-ns"
        content_hash = embedding_identity(text, namespace)
        cache.vectors[content_hash] = [0.1, 0.2, 0.3]
        encoder = FakeEncoder(dimension=3)
        materializer = FakeMaterializer()

        planner = SemanticWorkPlanner("test-ns", dimension=3)
        inputs = [
            SemanticInput("chunk-1", text, content_hash, "fine", 0),
        ]
        progress = planner.execute(inputs, cache, encoder, materializer)

        assert progress.total == 1
        assert progress.completed == 1
        assert progress.cache_hits == 1
        assert progress.cache_misses == 0
        assert encoder.call_count == 0  # No encoding needed for cache hit

    def test_planner_executes_with_cache_misses(self) -> None:
        cache = FakeCache()
        encoder = FakeEncoder(dimension=3)
        materializer = FakeMaterializer()

        planner = SemanticWorkPlanner("test-ns", dimension=3)
        text = "uncached"
        namespace = "test-ns"
        content_hash = embedding_identity(text, namespace)
        inputs = [
            SemanticInput("chunk-1", text, content_hash, "fine", 0),
        ]
        progress = planner.execute(inputs, cache, encoder, materializer)

        assert progress.total == 1
        assert progress.completed == 1
        assert progress.cache_hits == 0
        assert progress.cache_misses == 1
        assert encoder.call_count == 1

        # Vector should be cached and materialized
        assert content_hash in cache.vectors
        assert "chunk-1" in materializer.materialized

    def test_planner_batches_encoding(self) -> None:
        cache = FakeCache()
        encoder = FakeEncoder(dimension=3)
        materializer = FakeMaterializer()

        planner = SemanticWorkPlanner("test-ns", dimension=3)
        inputs = [
            SemanticInput("chunk-1", "text-1", "hash-1", "fine", 0),
            SemanticInput("chunk-2", "text-2", "hash-2", "fine", 0),
        ]
        planner.execute(inputs, cache, encoder, materializer)

        # Should encode both texts in one call
        assert encoder.call_count == 1


class TestSQLiteEmbeddingCache:
    """Test SQLite embedding cache implementation."""

    def test_get_many_batches_large_hash_lists(self) -> None:
        """Large cache lookups stay below SQLite's bound-variable limit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = SQLiteEmbeddingCache(
                Path(tmpdir) / "cache.db", "namespace-1", dimension=3
            )
            vectors = {
                f"hash-{index}": [0.1, 0.2, 0.3]
                for index in range(1_200)
            }

            cache.put_many(vectors)

            assert set(cache.get_many(vectors)) == set(vectors)

    def test_cache_put_and_get(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cache.db"
            cache = SQLiteEmbeddingCache(path, "namespace-1", dimension=3)

            vectors = {
                "hash-1": [0.1, 0.2, 0.3],
                "hash-2": [0.4, 0.5, 0.6],
            }
            cache.put_many(vectors)

            results = cache.get_many(["hash-1", "hash-2"])
            assert len(results) == 2
            assert list(results["hash-1"]) == np.asarray([0.1, 0.2, 0.3], dtype="<f4").tolist()
            assert list(results["hash-2"]) == np.asarray([0.4, 0.5, 0.6], dtype="<f4").tolist()

    def test_cache_namespace_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cache.db"
            cache1 = SQLiteEmbeddingCache(path, "namespace-1", dimension=3)
            cache2 = SQLiteEmbeddingCache(path, "namespace-2", dimension=3)

            cache1.put_many({"hash-1": [0.1, 0.2, 0.3]})

            # Cache2 should not see cache1's vectors
            results = cache2.get_many(["hash-1"])
            assert len(results) == 0

    def test_cache_prune_to_keeps_retained_namespace_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cache.db"
            cache1 = SQLiteEmbeddingCache(path, "namespace-1", dimension=3)
            cache2 = SQLiteEmbeddingCache(path, "namespace-2", dimension=3)
            cache1.put_many(
                {
                    "keep": [0.1, 0.2, 0.3],
                    "stale": [0.4, 0.5, 0.6],
                },
            )
            cache2.put_many({"stale": [0.7, 0.8, 0.9]})

            assert cache1.prune_to({"keep"}) == 1
            assert set(cache1.get_many(["keep", "stale"])) == {"keep"}
            assert set(cache2.get_many(["stale"])) == {"stale"}

    def test_cache_miss(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cache.db"
            cache = SQLiteEmbeddingCache(path, "namespace-1", dimension=3)

            results = cache.get_many(["hash-nonexistent"])
            assert len(results) == 0

    def test_cache_dimension_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cache.db"
            cache = SQLiteEmbeddingCache(path, "namespace-1", dimension=3)

            # Should accept 3-d vector
            cache.put_many({"hash-1": [0.1, 0.2, 0.3]})
            results = cache.get_many(["hash-1"])
            assert "hash-1" in results

            # Should reject non-matching dimension
            with pytest.raises(ValueError):
                cache.put_many({"hash-2": [0.1, 0.2]})

    def test_cache_metrics_tracking(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cache.db"
            cache = SQLiteEmbeddingCache(path, "namespace-1", dimension=3)

            cache.put_many({"hash-1": [0.1, 0.2, 0.3]})
            cache.get_many(["hash-1", "hash-2"])

            metrics = cache.metrics
            assert metrics.writes == 1
            assert metrics.hits == 1
            assert metrics.misses == 1

    def test_cache_persistence_across_instances(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cache.db"

            # Store in cache1
            cache1 = SQLiteEmbeddingCache(path, "namespace-1", dimension=3)
            cache1.put_many({"hash-1": [0.1, 0.2, 0.3]})
            cache1.close()

            # Retrieve with cache2
            cache2 = SQLiteEmbeddingCache(path, "namespace-1", dimension=3)
            results = cache2.get_many(["hash-1"])
            assert "hash-1" in results

    def test_cache_integrity_check_is_opt_in(self, monkeypatch) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cache.db"
            SQLiteEmbeddingCache(path, "namespace-1", dimension=3).close()
            calls = 0

            def count_checks(connection) -> None:
                nonlocal calls
                calls += 1

            monkeypatch.setattr(
                SQLiteEmbeddingCache,
                "_validate_integrity",
                staticmethod(count_checks),
            )
            SQLiteEmbeddingCache(path, "namespace-1", dimension=3).close()
            SQLiteEmbeddingCache(
                path,
                "namespace-1",
                dimension=3,
                validate=True,
            ).close()

            assert calls == 1

    def test_put_many_round_trips_at_float32_precision(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cache.db"
            cache = SQLiteEmbeddingCache(path, "namespace-1", dimension=3)
            vector = [0.123456789, -0.987654321, 1.0 / 3.0]

            cache.put_many({"hash-1": vector})

            results = cache.get_many(["hash-1"])
            expected = np.asarray(vector, dtype="<f4").tolist()
            assert list(results["hash-1"]) == expected

    def test_legacy_json_text_row_is_treated_as_miss_and_removed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cache.db"
            cache = SQLiteEmbeddingCache(path, "namespace-1", dimension=3)
            connection = sqlite3.connect(str(path))
            connection.execute(
                "INSERT OR REPLACE INTO embeddings "
                "(namespace, content_hash, dimension, vector) VALUES (?, ?, ?, ?)",
                ("namespace-1", "legacy-hash", 3, json.dumps([0.1, 0.2, 0.3])),
            )
            connection.commit()
            connection.close()

            results = cache.get_many(["legacy-hash"])

            assert results == {}
            remaining = cache._connection.execute(
                "SELECT content_hash FROM embeddings WHERE content_hash = ?",
                ("legacy-hash",),
            ).fetchall()
            assert remaining == []

    def test_blob_with_wrong_length_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cache.db"
            cache = SQLiteEmbeddingCache(path, "namespace-1", dimension=3)
            connection = sqlite3.connect(str(path))
            connection.execute(
                "INSERT OR REPLACE INTO embeddings "
                "(namespace, content_hash, dimension, vector) VALUES (?, ?, ?, ?)",
                ("namespace-1", "short-hash", 3, b"\x00\x00\x00"),
            )
            connection.commit()
            connection.close()

            results = cache.get_many(["short-hash"])

            assert results == {}
            remaining = cache._connection.execute(
                "SELECT content_hash FROM embeddings WHERE content_hash = ?",
                ("short-hash",),
            ).fetchall()
            assert remaining == []

    def test_old_format_cache_file_still_opens(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cache.db"
            connection = sqlite3.connect(str(path))
            connection.execute(
                "CREATE TABLE embeddings ("
                "namespace TEXT NOT NULL, content_hash TEXT NOT NULL, "
                "dimension INTEGER NOT NULL, vector TEXT NOT NULL, "
                "PRIMARY KEY (namespace, content_hash))"
            )
            connection.execute(
                "INSERT INTO embeddings "
                "(namespace, content_hash, dimension, vector) VALUES (?, ?, ?, ?)",
                ("namespace-1", "legacy-hash", 3, json.dumps([0.1, 0.2, 0.3])),
            )
            connection.commit()
            connection.close()

            cache = SQLiteEmbeddingCache(path, "namespace-1", dimension=3)

            assert cache.get_many(["legacy-hash"]) == {}

    def test_put_many_rejects_values_too_large_for_float32(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cache.db"
            cache = SQLiteEmbeddingCache(path, "namespace-1", dimension=3)

            with pytest.raises(ValueError, match="finite float32"):
                cache.put_many({"hash-1": [1e300, 0.2, 0.3]})


class TestLlamaIndexEmbeddingCacheAdapter:
    """Test the llama_index BaseKVStore-shaped cache adapter."""

    def test_get_returns_none_on_miss(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = SQLiteEmbeddingCache(Path(tmpdir) / "cache.db", "ns")
            adapter = LlamaIndexEmbeddingCacheAdapter(cache=cache, encoder_namespace="ns")
            assert adapter.get("some text") is None

    def test_put_then_get_round_trips_by_text_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = SQLiteEmbeddingCache(Path(tmpdir) / "cache.db", "ns")
            adapter = LlamaIndexEmbeddingCacheAdapter(cache=cache, encoder_namespace="ns")

            adapter.put("hello world", {"some-uuid": [0.1, 0.2, 0.3]})
            result = adapter.get("hello world")

            assert result is not None
            (vector,) = result.values()
            assert list(vector) == np.asarray([0.1, 0.2, 0.3], dtype="<f4").tolist()

    def test_different_text_keys_do_not_collide(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = SQLiteEmbeddingCache(Path(tmpdir) / "cache.db", "ns")
            adapter = LlamaIndexEmbeddingCacheAdapter(cache=cache, encoder_namespace="ns")

            adapter.put("text a", {"id-a": [0.1, 0.2, 0.3]})
            adapter.put("text b", {"id-b": [0.4, 0.5, 0.6]})

            result_a = adapter.get("text a")
            result_b = adapter.get("text b")
            assert result_a is not None
            assert result_b is not None
            (vec_a,) = result_a.values()
            (vec_b,) = result_b.values()
            assert list(vec_a) == np.asarray([0.1, 0.2, 0.3], dtype="<f4").tolist()
            assert list(vec_b) == np.asarray([0.4, 0.5, 0.6], dtype="<f4").tolist()
