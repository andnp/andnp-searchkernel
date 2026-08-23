"""Integration tests for pgvector store adapter.

Tests VectorStore, KeywordStore, GraphStore, and CacheStore implementations
against a live Postgres database with pgvector extension.
"""

from dataclasses import replace
from datetime import UTC, datetime
from typing import cast

import numpy as np
import pytest

from searchkernel.adapters.stores.pgvector import (
    PGCacheStore,
    PGGraphStore,
    PGKeywordStore,
    PGVectorStore,
    PostgresConnection,
    _vector_table_name,
    create_schema,
)
from searchkernel.domain import (
    GraphEdge,
    GraphNeighbor,
    Record,
    RecordIdentity,
    RecordStatus,
    Vector,
)
from searchkernel.indices import LocalRecordBackend
from tests.integration.conftest import pg_dsn_for_schema, pg_worker_schema


@pytest.fixture(scope="function")
def pg_conn(pg_dsn, request, pg_cleanup_executor):
    """Create a test connection pool scoped to this xdist worker's own schema.

    Each xdist worker gets a private Postgres schema (pinned via search_path
    on the connection DSN), so this file's DELETE-everything cleanup below
    only ever touches this worker's own tables -- concurrent workers running
    the same file's tests can never collide, regardless of --dist mode.
    """
    schema = pg_worker_schema(request.config)
    scoped_dsn = pg_dsn_for_schema(pg_dsn, schema)

    bootstrap_pool = PostgresConnection(pg_dsn, min_connections=1, max_connections=1)
    bootstrap_conn = bootstrap_pool.get_connection()
    bootstrap_cursor = bootstrap_conn.cursor()
    bootstrap_cursor.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}";')
    bootstrap_conn.commit()
    bootstrap_cursor.close()
    bootstrap_pool.put_connection(bootstrap_conn)
    bootstrap_pool.close()

    conn_pool = PostgresConnection(scoped_dsn)
    create_schema(conn_pool)

    # Clean slate for this worker's schema before every test.
    conn = conn_pool.get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT table_name FROM vector_tables;")
    for (table_name,) in cursor.fetchall():
        cursor.execute(f'DROP TABLE IF EXISTS "{table_name}";')

    cursor.execute("DELETE FROM vector_tables;")
    cursor.execute("DELETE FROM records;")
    cursor.execute("DELETE FROM graph_edges;")
    cursor.execute("DELETE FROM cache_store;")
    cursor.execute(
        "UPDATE index_epoch SET epoch = 0, keyword_epoch = 0, "
        "vector_epoch = 0, graph_epoch = 0;"
    )
    conn.commit()
    cursor.close()
    conn_pool.put_connection(conn)

    yield conn_pool

    # Cleanup
    pg_cleanup_executor.submit(conn_pool.close)


@pytest.fixture
def fixture_records():
    """Create fixture records for testing."""
    now = datetime.now(UTC)
    return [
        Record(
            source_kind="test",
            source_id="test:1",
            title="Machine Learning Basics",
            body="Machine learning is a subset of AI. It enables systems to learn from data.",
            created_at=now,
            updated_at=now,
            metadata={"category": "ai"},
            uri="http://example.com/ml",
            status=RecordStatus.ACTIVE,
            embedding=[1.0, 0.0, 0.0, 0.0],
        ),
        Record(
            source_kind="test",
            source_id="test:2",
            title="Deep Learning Neural Networks",
            body="Neural networks are inspired by biological neurons. Deep learning uses many layers.",
            created_at=now,
            updated_at=now,
            metadata={"category": "ai"},
            uri="http://example.com/dl",
            status=RecordStatus.ACTIVE,
            embedding=[0.9, 0.1, 0.0, 0.0],
        ),
        Record(
            source_kind="test",
            source_id="test:3",
            title="Database Systems",
            body="Relational databases use SQL. PostgreSQL is a popular open-source database.",
            created_at=now,
            updated_at=now,
            metadata={"category": "database"},
            uri="http://example.com/db",
            status=RecordStatus.ACTIVE,
            embedding=[0.0, 0.0, 1.0, 0.0],
        ),
    ]


class TestVectorStore:
    """Tests for VectorStore port implementation."""

    def test_repeated_upsert_skips_vector_write_and_keyword_epoch(
        self, pg_conn, fixture_records
    ):
        """An identical retry preserves both index lanes and the vector row."""
        store = PGVectorStore(pg_conn)
        record = fixture_records[0]
        table_name = _vector_table_name("repeat-model", 4)

        store.upsert([record], model_name="repeat-model", dim=4)
        before = store.epochs()
        before_vector_timestamp = pg_conn.execute_one(
            f"""
            SELECT updated_at
            FROM "{table_name}"
            WHERE record_id = %s;
            """,
            (record.storage_key,),
        )

        store.upsert([record], model_name="repeat-model", dim=4)

        assert store.epochs() == {
            "keyword": before["keyword"],
            "vector": before["vector"],
            "graph": before["graph"],
        }
        assert pg_conn.execute_one(
            f"""
            SELECT updated_at
            FROM "{table_name}"
            WHERE record_id = %s;
            """,
            (record.storage_key,),
        ) == before_vector_timestamp

    def test_mixed_upsert_changes_only_changed_and_missing_vectors(
        self, pg_conn, fixture_records
    ):
        """Mixed batches keep unchanged rows and write changed or missing ones."""
        store = PGVectorStore(pg_conn)
        initial = fixture_records[:2]
        store.upsert(initial, model_name="mixed-model", dim=4)
        before = store.epochs()
        unchanged_vector_timestamp = pg_conn.execute_one(
            f'SELECT updated_at FROM "{_vector_table_name("mixed-model", 4)}" '
            "WHERE record_id = %s;",
            (initial[0].storage_key,),
        )
        original_revision = pg_conn.execute_one(
            f'SELECT revision FROM "{_vector_table_name("mixed-model", 4)}" '
            "WHERE record_id = %s;",
            (initial[1].storage_key,),
        )[0]

        changed = replace(initial[1], embedding=[0.0, 0.0, 0.0, 1.0])
        missing = replace(
            initial[0], source_id="test:missing", embedding=[0.0, 0.0, 1.0, 0.0]
        )
        store.upsert(
            [initial[0], changed, missing],
            model_name="mixed-model",
            dim=4,
        )

        assert store.epochs() == {
            "keyword": before["keyword"],
            "vector": before["vector"] + 1,
            "graph": before["graph"],
        }
        stored = pg_conn.execute_one(
            f'SELECT embedding::text, revision FROM "{_vector_table_name("mixed-model", 4)}" '
            "WHERE record_id = %s;",
            (initial[1].storage_key,),
        )
        assert stored[0] == "[0,0,0,1]"
        assert stored[1] == original_revision
        assert pg_conn.execute_one(
            f'SELECT updated_at FROM "{_vector_table_name("mixed-model", 4)}" '
            "WHERE record_id = %s;",
            (initial[0].storage_key,),
        ) == unchanged_vector_timestamp
        assert pg_conn.execute_one(
            f'SELECT COUNT(*) FROM "{_vector_table_name("mixed-model", 4)}";'
        ) == (3,)

    def test_duplicate_storage_keys_keep_last_record_and_vector(
        self, pg_conn, fixture_records
    ):
        """Duplicate keys use the final vector without a cardinality error."""
        store = PGVectorStore(pg_conn)
        first = fixture_records[0]
        last = replace(
            first,
            title="Final title",
            body="Final body",
            embedding=[0.0, 0.0, 0.0, 1.0],
        )

        store.upsert([first, last], model_name="duplicate-model", dim=4)

        assert pg_conn.execute_one(
            "SELECT title, body FROM records WHERE record_id = %s;",
            (first.storage_key,),
        ) == ("Final title", "Final body")
        assert pg_conn.execute_one(
            f'SELECT embedding::text FROM "{_vector_table_name("duplicate-model", 4)}" '
            "WHERE record_id = %s;",
            (first.storage_key,),
        ) == ("[0,0,0,1]",)
        assert pg_conn.execute_one("SELECT COUNT(*) FROM records;") == (1,)

    @pytest.mark.parametrize(
        ("embedding", "message"),
        [
            (cast(Vector, [1.0, float("nan"), 0.0, 0.0]), "finite numbers"),
            (cast(Vector, [1.0, float("inf"), 0.0, 0.0]), "finite numbers"),
            (cast(Vector, [1.0, object(), 0.0, 0.0]), "finite numbers"),
        ],
    )
    def test_invalid_vector_elements_roll_back_storage(
        self, pg_conn, fixture_records, embedding, message
    ):
        """Invalid vector elements fail before canonical rows are committed."""
        store = PGVectorStore(pg_conn)
        record = replace(
            fixture_records[0], source_id="invalid-vector", embedding=embedding
        )

        with pytest.raises(ValueError, match=message):
            store.upsert([record], model_name="invalid-vector-model", dim=4)

        assert store.epochs() == {"keyword": 0, "vector": 0, "graph": 0}
        assert pg_conn.execute_one("SELECT COUNT(*) FROM records;") == (0,)
        assert pg_conn.execute_one("SELECT COUNT(*) FROM vector_tables;") == (0,)

    def test_upsert_repairs_damaged_payload_when_revision_matches(
        self, pg_conn, fixture_records
    ):
        """A matching revision does not hide a damaged stored vector."""
        store = PGVectorStore(pg_conn)
        record = fixture_records[0]
        model_name = "repair-model"
        table_name = _vector_table_name(model_name, 4)
        store.upsert([record], model_name=model_name, dim=4)
        revision = pg_conn.execute_one(
            f'SELECT revision FROM "{table_name}" WHERE record_id = %s;',
            (record.storage_key,),
        )[0]
        connection = pg_conn.get_connection()
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    f'UPDATE "{table_name}" SET embedding = %s::vector, '
                    "revision = %s WHERE record_id = %s;",
                    ("[0,0,0,1]", revision, record.storage_key),
                )
            connection.commit()
        finally:
            pg_conn.put_connection(connection)
        before = store.epochs()

        store.upsert([record], model_name=model_name, dim=4)

        assert store.epochs() == {
            "keyword": before["keyword"],
            "vector": before["vector"] + 1,
            "graph": before["graph"],
        }
        assert pg_conn.execute_one(
            f'SELECT embedding::text, revision FROM "{table_name}" '
            "WHERE record_id = %s;",
            (record.storage_key,),
        ) == ("[1,0,0,0]", revision)

    def test_failed_upsert_rolls_back_canonical_rows_and_epochs(self, pg_conn):
        """A vector serialization failure rolls back earlier canonical writes."""
        now = datetime.now(UTC)
        good = replace(
            Record(
                source_kind="rollback",
                source_id="good",
                title="Good",
                body="Good body",
                created_at=now,
                updated_at=now,
                embedding=[1.0, 0.0, 0.0, 0.0],
            ),
            embedding=cast(Vector, ["invalid", 0.0, 0.0, 0.0]),
        )
        store = PGVectorStore(pg_conn)

        with pytest.raises(ValueError):
            store.upsert([good], model_name="rollback-model", dim=4)

        assert store.epochs() == {"keyword": 0, "vector": 0, "graph": 0}
        assert pg_conn.execute_one("SELECT COUNT(*) FROM records;") == (0,)
        assert pg_conn.execute_one("SELECT COUNT(*) FROM vector_tables;") == (0,)

    def test_upsert_and_search_parity(self, pg_conn, fixture_records):
        """Test that pgvector ANN search returns same top-k as brute-force cosine.

        This is the key acceptance test: verify that pgvector HNSW gives
        same (or very similar) results as a reference numpy implementation.
        """
        store = PGVectorStore(pg_conn)

        # Upsert fixture records
        store.upsert(fixture_records, model_name="test-model", dim=4)

        # Query vector similar to record 1 and 2
        query_vec: Vector = [0.95, 0.05, 0.0, 0.0]

        # Get results from pgvector
        pgvector_results = store.search(query_vec, k=3, model_name="test-model", dim=4)

        # Compute brute-force reference using numpy
        embeddings = np.array([r.embedding for r in fixture_records])
        query_array = np.array(query_vec)

        # Cosine similarity: (A · B) / (||A|| ||B||)
        similarities = []
        for emb in embeddings:
            dot = np.dot(query_array, emb)
            norm_q = np.linalg.norm(query_array)
            norm_e = np.linalg.norm(emb)
            cosine_sim = dot / (norm_q * norm_e)
            similarities.append(cosine_sim)

        # Sort by similarity descending
        sorted_indices = np.argsort(similarities)[::-1]
        reference_ids = [fixture_records[i].source_id for i in sorted_indices[:3]]

        # Verify pgvector results match reference (order may differ slightly for ties)
        pgvector_ids = [r[0] for r in pgvector_results]
        assert len(pgvector_ids) == 3
        assert set(pgvector_ids) == set(reference_ids[:3])
        # First result should be closest
        assert pgvector_ids[0] in reference_ids[:2]

    def test_mixed_dimension_rejection(self, pg_conn, fixture_records):
        """Test that mixed-dimension writes are rejected."""
        store = PGVectorStore(pg_conn)

        # Upsert with dim=4
        store.upsert(fixture_records, model_name="model-v1", dim=4)

        # Try to upsert with dim=5 (should fail)
        bad_record = Record(
            source_kind="test",
            source_id="test:bad",
            title="Bad Embedding",
            body="This has wrong dimension",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            embedding=[1.0, 0.0, 0.0, 0.0, 0.0],  # 5 dims
        )

        with pytest.raises(ValueError, match="Dimension mismatch"):
            store.upsert([bad_record], model_name="model-v1", dim=5)

    def test_missing_embedding_rejection_precedes_mutation(
        self, pg_conn, fixture_records
    ):
        """Reject a mixed batch before writing records or vectors."""
        store = PGVectorStore(pg_conn)
        missing_embedding = Record(
            source_kind="test",
            source_id="test:missing-embedding",
            title="Missing embedding",
            body="This record has no vector",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        with pytest.raises(ValueError, match="must have an embedding"):
            store.upsert(
                [fixture_records[0], missing_embedding],
                model_name="missing-embedding-model",
                dim=4,
            )

        assert pg_conn.execute_one("SELECT COUNT(*) FROM records;") == (0,)
        assert pg_conn.execute_one("SELECT COUNT(*) FROM vector_tables;") == (0,)

    def test_delete_records(self, pg_conn, fixture_records):
        """Test delete operation."""
        store = PGVectorStore(pg_conn)
        store.upsert(fixture_records, model_name="test-model", dim=4)

        # Delete first two records
        store.delete([fixture_records[0].storage_key, fixture_records[1].storage_key])

        # Search should only return third record
        query_vec = [0.0, 0.0, 1.0, 0.0]
        results = store.search(query_vec, k=10, model_name="test-model", dim=4)

        result_ids = [r[0] for r in results]
        assert fixture_records[0].source_id not in result_ids
        assert fixture_records[1].source_id not in result_ids
        assert fixture_records[2].source_id in result_ids

    def test_delete_removes_incident_graph_edges(self, pg_conn, fixture_records):
        vector_store = PGVectorStore(pg_conn)
        graph_store = PGGraphStore(pg_conn)
        vector_store.upsert(fixture_records, model_name="test-model", dim=4)
        source = RecordIdentity(None, "test", fixture_records[0].source_id)
        target = RecordIdentity(None, "test", fixture_records[1].source_id)
        graph_store.upsert_edges(
            [
                GraphEdge(source, target, "links", 0.9),
                GraphEdge(target, source, "links", 0.8),
            ]
        )
        graph_epoch = graph_store.graph_epoch()

        vector_store.delete([source.storage_key])

        assert graph_store.neighbors(target) == []
        assert graph_store.graph_epoch() > graph_epoch

    def test_model_delete_preserves_graph_edges_until_last_model(self, pg_conn):
        vector_store = PGVectorStore(pg_conn)
        graph_store = PGGraphStore(pg_conn)
        now = datetime.now(UTC)
        source = Record(
            source_kind="note",
            source_id="shared-source",
            title="Source",
            body="Source",
            created_at=now,
            updated_at=now,
            embedding=[1.0, 0.0, 0.0, 0.0],
        )
        target = RecordIdentity(None, "note", "shared-target")
        vector_store.upsert([source], model_name="model-one", dim=4)
        vector_store.upsert([source], model_name="model-two", dim=4)
        graph_store.upsert_edges(
            [GraphEdge(RecordIdentity(None, "note", source.source_id), target, "links", 1.0)]
        )

        vector_store.delete_for_model([source.storage_key], "model-one", 4)
        assert graph_store.neighbors(RecordIdentity(None, "note", source.source_id))

        vector_store.delete_for_model([source.storage_key], "model-two", 4)
        assert graph_store.neighbors(RecordIdentity(None, "note", source.source_id)) == []

    def test_delete_rejects_bare_source_id(self, pg_conn, fixture_records):
        store = PGVectorStore(pg_conn)
        store.upsert(fixture_records[:1], model_name="test-model", dim=4)

        with pytest.raises(ValueError, match="canonical storage key"):
            store.delete([fixture_records[0].source_id])

    def test_schema_declares_canonical_record_and_graph_fields(self, pg_conn):
        assert pg_conn.execute_one(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'records' AND column_name = 'indexed_text';
            """
        ) == ("indexed_text",)
        assert pg_conn.execute_one(
            """
            SELECT column_default
            FROM information_schema.columns
            WHERE table_name = 'graph_edges' AND column_name = 'source_kind';
            """
        ) == (None,)
        assert pg_conn.execute_one(
            """
            SELECT column_default
            FROM information_schema.columns
            WHERE table_name = 'graph_edges' AND column_name = 'target_kind';
            """
        ) == (None,)

    def test_schema_declares_graph_identity_indexes(self, pg_conn):
        """Graph edges have identity indexes for both traversal directions."""
        assert pg_conn.execute_one(
            """
            SELECT array_agg(DISTINCT indexname ORDER BY indexname)
            FROM pg_indexes
            WHERE tablename = 'graph_edges'
              AND indexname IN (
                  'idx_graph_edges_source_identity',
                  'idx_graph_edges_target_identity'
              );
            """
        ) == ([
            "idx_graph_edges_source_identity",
            "idx_graph_edges_target_identity",
        ],)

    def test_vector_revision_persists_and_legacy_tables_are_initialized(
        self, pg_conn, fixture_records
    ):
        """Vector bootstrap persists revisions and repairs old tables."""
        store = PGVectorStore(pg_conn)
        record = fixture_records[0]
        model_name = "revision-model"
        dim = 4
        table_name = _vector_table_name(model_name, dim)

        store.upsert([record], model_name=model_name, dim=dim)
        revision = pg_conn.execute_one(
            f'SELECT revision FROM "{table_name}" WHERE record_id = %s;',
            (record.storage_key,),
        )[0]
        assert revision

        connection = pg_conn.get_connection()
        try:
            with connection.cursor() as cursor:
                cursor.execute(f'ALTER TABLE "{table_name}" DROP COLUMN revision;')
            connection.commit()
        finally:
            pg_conn.put_connection(connection)
        store.upsert([record], model_name=model_name, dim=dim)

        restored_revision = pg_conn.execute_one(
            f'SELECT revision FROM "{table_name}" WHERE record_id = %s;',
            (record.storage_key,),
        )[0]
        assert restored_revision == revision

    def test_epoch_tracking(self, pg_conn, fixture_records):
        """Test each index lane increments independently and totals advance."""
        vector_store = PGVectorStore(pg_conn)
        keyword_store = PGKeywordStore(pg_conn)
        graph_store = PGGraphStore(pg_conn)
        cache_store = PGCacheStore(pg_conn)
        initial = vector_store.epochs()
        total = vector_store.epoch()
        assert initial == {"keyword": 0, "vector": 0, "graph": 0}
        assert total == 0

        vector_store.upsert(fixture_records, model_name="test-model", dim=4)
        after_vector = vector_store.epochs()
        assert after_vector == {"keyword": 0, "vector": 1, "graph": 0}
        assert vector_store.epoch() == total + 1

        keyword_store.index(fixture_records)
        after_keyword = keyword_store.epochs()
        assert after_keyword == {"keyword": 1, "vector": 1, "graph": 0}
        assert keyword_store.epoch() == total + 2

        edge = GraphEdge(
            RecordIdentity(None, "test", "source"),
            RecordIdentity(None, "test", "target"),
            "related",
            1.0,
        )
        graph_store.upsert_edges([edge])
        assert graph_store.epochs() == {"keyword": 1, "vector": 1, "graph": 1}
        assert graph_store.epochs() == keyword_store.epochs()
        assert graph_store.graph_epoch() == 1
        assert vector_store.epoch() == total + 3

        cache_store.set("epoch-check", {"ok": True}, epoch=3)
        cache_store.invalidate_epoch(0)
        assert vector_store.epochs() == {"keyword": 1, "vector": 1, "graph": 1}
        assert vector_store.epoch() == total + 3

    def test_per_model_isolation(self, pg_conn, fixture_records):
        """Test that different embedding models are isolated."""
        store = PGVectorStore(pg_conn)

        # Upsert same records with different models
        store.upsert(fixture_records, model_name="model-v1", dim=4)
        store.upsert(fixture_records, model_name="model-v2", dim=4)

        # Search in model-v1; should work
        results = store.search(
            [1.0, 0.0, 0.0, 0.0], k=3, model_name="model-v1", dim=4
        )
        assert len(results) == 3

    def test_search_filters_source_kinds(self, pg_conn, fixture_records):
        """Test vector search filters without leaking other record kinds."""
        other = Record(
            source_kind="other",
            source_id="other:1",
            title="Other record",
            body="Machine learning from another source",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            embedding=[1.0, 0.0, 0.0, 0.0],
        )
        store = PGVectorStore(pg_conn)
        store.upsert([*fixture_records, other], model_name="filtered-model", dim=4)

        filtered = store.search(
            [1.0, 0.0, 0.0, 0.0],
            k=10,
            model_name="filtered-model",
            dim=4,
            filters={"source_kinds": ["test"]},
        )
        assert "other:1" not in {record_id for record_id, _score in filtered}
        assert store.search(
            [1.0, 0.0, 0.0, 0.0],
            k=10,
            model_name="filtered-model",
            dim=4,
            filters={"source_kinds": ["missing"]},
        ) == []

    def test_filtered_eligible_identities_match_local_backend(
        self,
        pg_conn,
        tmp_path,
    ):
        now = datetime.now(UTC)
        records = [
            Record(
                workspace_id="workspace",
                source_kind="note",
                source_id="keep",
                title="Keep",
                body="Keep",
                created_at=now,
                updated_at=now,
                metadata={
                    "project_id": "project-a",
                    "doc_id": "docs/keep",
                    "file_path": "docs/keep.md",
                },
                uri="docs/keep.md",
                embedding=[1.0, 0.0, 0.0, 0.0],
            ),
            Record(
                workspace_id="workspace",
                source_kind="note",
                source_id="excluded-project",
                title="Excluded project",
                body="Excluded project",
                created_at=now,
                updated_at=now,
                metadata={
                    "project_id": "project-b",
                    "doc_id": "docs/project-b",
                    "file_path": "docs/project-b.md",
                },
                uri="docs/project-b.md",
                embedding=[1.0, 0.0, 0.0, 0.0],
            ),
            Record(
                workspace_id="other-workspace",
                source_kind="note",
                source_id="other-workspace",
                title="Other workspace",
                body="Other workspace",
                created_at=now,
                updated_at=now,
                metadata={"project_id": "project-a", "doc_id": "docs/other"},
                embedding=[1.0, 0.0, 0.0, 0.0],
            ),
            Record(
                workspace_id="workspace",
                source_kind="note",
                source_id="archived",
                title="Archived",
                body="Archived",
                created_at=now,
                updated_at=now,
                status=RecordStatus.ARCHIVED,
                metadata={"project_id": "project-a", "doc_id": "docs/archived"},
                embedding=[1.0, 0.0, 0.0, 0.0],
            ),
        ]
        local = LocalRecordBackend(tmp_path / "local.db")
        local.upsert(records, "contract-model", 4)
        pg = PGVectorStore(pg_conn)
        pg.upsert(records, "contract-model", 4)
        filters = {
            "workspace_id": "workspace",
            "source_kinds": ["note"],
            "statuses": ["active"],
            "project_id": "project-a",
            "candidate_ids": [records[0].storage_key],
            "excluded_files": {"docs/archived"},
            "excluded_documents": {"docs/blocked"},
        }

        local_hits = local.search_vector(
            [1.0, 0.0, 0.0, 0.0],
            10,
            model_name="contract-model",
            dim=4,
            filters=filters,
        )
        pg_hits = pg.search(
            [1.0, 0.0, 0.0, 0.0],
            10,
            model_name="contract-model",
            dim=4,
            filters=filters,
        )

        assert [hit.storage_key for hit in pg_hits] == [
            hit.storage_key for hit in local_hits
        ] == [records[0].storage_key]

    def test_equal_vector_ties_preserve_workspace_identity_order(
        self,
        pg_conn,
        tmp_path,
    ):
        now = datetime.now(UTC)
        records = [
            Record(
                workspace_id=workspace_id,
                source_kind="note",
                source_id="shared",
                title=workspace_id,
                body=workspace_id,
                created_at=now,
                updated_at=now,
                embedding=[1.0, 0.0, 0.0, 0.0],
            )
            for workspace_id in ("workspace-c", "workspace-a", "workspace-b")
        ]
        local = LocalRecordBackend(tmp_path / "local.db")
        local.upsert(records, "tie-model", 4)
        pg = PGVectorStore(pg_conn)
        pg.upsert(records, "tie-model", 4)

        local_hits = local.search_vector(
            [1.0, 0.0, 0.0, 0.0],
            10,
            model_name="tie-model",
            dim=4,
        )
        pg_hits = pg.search(
            [1.0, 0.0, 0.0, 0.0],
            10,
            model_name="tie-model",
            dim=4,
        )

        expected = sorted(record.storage_key for record in records)
        assert [hit.storage_key for hit in local_hits] == expected
        assert [hit.storage_key for hit in pg_hits] == expected
        assert [
            hit.storage_key
            for hit in pg.search(
                [1.0, 0.0, 0.0, 0.0],
                10,
                model_name="tie-model",
                dim=4,
                filters={"workspace_id": "workspace-a"},
            )
        ] == [records[1].storage_key]

    def test_dimension_and_model_tables_are_isolated(self, pg_conn, fixture_records):
        """Test that model/dimension pairs select only their own table."""
        store = PGVectorStore(pg_conn)
        store.upsert(fixture_records, model_name="model-four", dim=4)
        three_dimensional = Record(
            source_kind="three",
            source_id="three:1",
            title="Three dimensional",
            body="Three dimensional vector",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            embedding=[1.0, 0.0, 0.0],
        )
        store.upsert([three_dimensional], model_name="model-three", dim=3)

        assert store.search(
            [1.0, 0.0, 0.0],
            k=10,
            model_name="model-four",
            dim=3,
        ) == []
        result = store.search(
            [1.0, 0.0, 0.0],
            k=10,
            model_name="model-three",
            dim=3,
        )[0]
        assert result.source_id == "three:1"
        assert result.score == pytest.approx(1.0)

    def test_delete_removes_records_from_all_models(self, pg_conn, fixture_records):
        """Test global deletion removes vectors from every model table."""
        store = PGVectorStore(pg_conn)
        store.upsert(fixture_records, model_name="model-v1", dim=4)
        store.upsert(fixture_records, model_name="model-v2", dim=4)

        store.delete([fixture_records[0].storage_key])

        for model_name in ("model-v1", "model-v2"):
            results = store.search(
                [1.0, 0.0, 0.0, 0.0],
                k=10,
                model_name=model_name,
                dim=4,
            )
            assert "test:1" not in {record_id for record_id, _score in results}

    def test_failed_upsert_does_not_poison_connection_or_leave_rows(self, pg_conn):
        """Test malformed vectors fail before partially writing a transaction."""
        now = datetime.now(UTC)
        good = Record(
            source_kind="failure",
            source_id="failure:good",
            title="Good",
            body="Good record",
            created_at=now,
            updated_at=now,
            embedding=[1.0, 0.0, 0.0, 0.0],
        )
        bad = Record(
            source_kind="failure",
            source_id="failure:bad",
            title="Bad",
            body="Bad record",
            created_at=now,
            updated_at=now,
            embedding=[1.0, 0.0, 0.0],
        )
        store = PGVectorStore(pg_conn)
        good_embedding = good.embedding
        assert good_embedding is not None

        with pytest.raises(ValueError, match="Embedding dimension mismatch"):
            store.upsert([good, bad], model_name="failure-model", dim=4)

        assert store.search(
            good_embedding, k=10, model_name="failure-model", dim=4
        ) == []
        store.upsert([good], model_name="failure-model", dim=4)
        assert store.search(
            good_embedding, k=10, model_name="failure-model", dim=4
        )[0][0] == good.source_id

    def test_hnsw_index_exists_per_model_table(self, pg_conn, fixture_records):
        """Test that an HNSW index is created on the per-model vector table."""
        store = PGVectorStore(pg_conn)
        store.upsert(fixture_records, model_name="test-model", dim=4)

        table_name = _vector_table_name("test-model", 4)

        conn = pg_conn.get_connection()
        cursor = None
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT indexdef FROM pg_indexes WHERE tablename = %s;",
                (table_name,),
            )
            indexdefs = [row[0] for row in cursor.fetchall()]
        finally:
            if cursor is not None:
                cursor.close()
            pg_conn.put_connection(conn)

        assert any("hnsw" in indexdef.lower() for indexdef in indexdefs), (
            f"Expected an HNSW index on {table_name}, found: {indexdefs}"
        )

    def test_filtered_query_plan_is_measurable(self, pg_conn, fixture_records):
        store = PGVectorStore(pg_conn)
        store.upsert(fixture_records, model_name="plan-model", dim=4)

        conn = pg_conn.get_connection()
        cursor = None
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT indexname
                FROM pg_indexes
                WHERE tablename = 'records'
                  AND indexname IN (
                      'idx_records_vector_filters',
                      'idx_records_project_filter',
                      'idx_records_document_filter',
                      'idx_records_path_filter'
                  );
                """
            )
            assert {
                row[0] for row in cursor.fetchall()
            } == {
                "idx_records_vector_filters",
                "idx_records_project_filter",
                "idx_records_document_filter",
                "idx_records_path_filter",
            }
            table_name = _vector_table_name("plan-model", 4)
            cursor.execute(
                f"""
                EXPLAIN (ANALYZE, BUFFERS)
                SELECT r.record_id
                FROM "{table_name}" v
                JOIN records r ON r.record_id = v.record_id
                WHERE r.workspace_id IS NULL
                  AND r.source_kind = %s
                  AND r.status = %s
                ORDER BY v.embedding <=> %s::vector, v.record_id
                LIMIT %s;
                """,
                ("test", "active", "[1,0,0,0]", 3),
            )
            plan = [row[0] for row in cursor.fetchall()]
        finally:
            if cursor is not None:
                cursor.close()
            pg_conn.put_connection(conn)

        assert plan
        assert any("Limit" in line for line in plan)

    def test_ann_recall_at_10(self, pg_conn):
        """Test that HNSW ANN search achieves recall@10 >= 0.9 vs brute-force cosine.

        HNSW is approximate, so exact top-k equality is not guaranteed;
        this verifies the index returns a highly-overlapping result set
        with a numpy brute-force reference over a larger, randomized corpus.
        """
        rng = np.random.default_rng(42)
        n_records = 500
        dim = 1024

        raw_vectors = rng.normal(size=(n_records, dim))
        norms = np.linalg.norm(raw_vectors, axis=1, keepdims=True)
        vectors = raw_vectors / norms

        now = datetime.now(UTC)
        records = [
            Record(
                source_kind="test",
                source_id=f"recall:{i}",
                title=f"Recall fixture {i}",
                body="Randomized recall corpus entry.",
                created_at=now,
                updated_at=now,
                embedding=vectors[i].tolist(),
            )
            for i in range(n_records)
        ]

        store = PGVectorStore(pg_conn)
        store.upsert(records, model_name="recall-model", dim=dim)

        query_vec = rng.normal(size=dim)
        query_vec = query_vec / np.linalg.norm(query_vec)

        k = 10
        pgvector_results = store.search(
            query_vec.tolist(), k=k, model_name="recall-model", dim=dim
        )
        pgvector_ids = {r[0] for r in pgvector_results}

        # Brute-force cosine similarity reference (vectors are unit-norm).
        similarities = vectors @ query_vec
        top_k_indices = np.argsort(similarities)[::-1][:k]
        reference_ids = {f"recall:{i}" for i in top_k_indices}

        recall = len(pgvector_ids & reference_ids) / k
        assert recall >= 0.9, (
            f"recall@{k} = {recall} below threshold; "
            f"pgvector={pgvector_ids} reference={reference_ids}"
        )

    def test_get_many_returns_stored_vectors_for_unchanged_records(
        self, pg_conn, fixture_records
    ):
        store = PGVectorStore(pg_conn)
        store.upsert(fixture_records, model_name="test-model", dim=4)

        stored = store.get_many(fixture_records, model_name="test-model", dim=4)

        assert set(stored) == {record.storage_key for record in fixture_records}
        for record in fixture_records:
            assert stored[record.storage_key] == record.embedding

    def test_get_many_omits_records_whose_content_changed_since_upsert(
        self, pg_conn, fixture_records
    ):
        store = PGVectorStore(pg_conn)
        store.upsert(fixture_records, model_name="test-model", dim=4)

        changed = replace(fixture_records[0], body="Completely different content now.")
        current_records = [changed, *fixture_records[1:]]

        stored = store.get_many(current_records, model_name="test-model", dim=4)

        assert changed.storage_key not in stored
        for record in fixture_records[1:]:
            assert record.storage_key in stored

    def test_get_many_omits_records_with_no_stored_vector(self, pg_conn, fixture_records):
        store = PGVectorStore(pg_conn)
        store.upsert(fixture_records[:1], model_name="test-model", dim=4)

        stored = store.get_many(fixture_records, model_name="test-model", dim=4)

        assert set(stored) == {fixture_records[0].storage_key}

    def test_get_many_returns_empty_for_unknown_model(self, pg_conn, fixture_records):
        store = PGVectorStore(pg_conn)

        assert store.get_many(fixture_records, model_name="never-indexed", dim=4) == {}

    def test_get_many_backfills_a_missing_revision_column(self, pg_conn, fixture_records):
        """A table created before revision tracking existed has no such
        column; get_many must add it rather than assume every existing
        deployment has already upserted since that column was added."""
        store = PGVectorStore(pg_conn)
        store.upsert(fixture_records, model_name="test-model", dim=4)

        conn = pg_conn.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            f'ALTER TABLE "{_vector_table_name("test-model", 4)}" DROP COLUMN revision;'
        )
        conn.commit()
        cursor.close()
        pg_conn.put_connection(conn)

        stored = store.get_many(fixture_records, model_name="test-model", dim=4)

        assert stored == {}

    def test_get_many_empty_records_returns_empty_without_querying(self, pg_conn):
        store = PGVectorStore(pg_conn)

        assert store.get_many([], model_name="test-model", dim=4) == {}


class TestKeywordStore:
    """Tests for KeywordStore port implementation."""

    def test_lexical_queries_match_local_backend(self, pg_conn, tmp_path):
        """Keep PostgreSQL lexical retrieval aligned with the local contract.

        Phrase, prefix, artifact, filter, empty-query, and tie ordering cases
        exercise the query shapes that must remain portable across backends.
        """
        now = datetime(2026, 1, 1, tzinfo=UTC)
        records = [
            Record(
                workspace_id="workspace-a",
                source_kind="note",
                source_id="phrase",
                title="Alpha beta guide",
                body="alpha beta phrase",
                uri="src/searchkernel/search.py",
                created_at=now,
                updated_at=now,
                embedding=[1.0, 0.0, 0.0, 0.0],
            ),
            Record(
                workspace_id="workspace-a",
                source_kind="note",
                source_id="prefix",
                title="Alphabet",
                body="alphabet soup",
                created_at=now,
                updated_at=now,
                embedding=[1.0, 0.0, 0.0, 0.0],
            ),
            Record(
                workspace_id="workspace-a",
                source_kind="note",
                source_id="symbol",
                title="Parser",
                body="parse_record implementation",
                created_at=now,
                updated_at=now,
                embedding=[1.0, 0.0, 0.0, 0.0],
            ),
            Record(
                workspace_id="workspace-a",
                source_kind="note",
                source_id="active",
                title="Common active",
                body="common token",
                created_at=now,
                updated_at=now,
                embedding=[1.0, 0.0, 0.0, 0.0],
            ),
            Record(
                workspace_id="workspace-b",
                source_kind="note",
                source_id="other-workspace",
                title="Common other workspace",
                body="common token",
                created_at=now,
                updated_at=now,
                embedding=[1.0, 0.0, 0.0, 0.0],
            ),
            Record(
                workspace_id="workspace-a",
                source_kind="note",
                source_id="archived",
                title="Common archived",
                body="common token",
                status=RecordStatus.ARCHIVED,
                created_at=now,
                updated_at=now,
                embedding=[1.0, 0.0, 0.0, 0.0],
            ),
        ]
        local = LocalRecordBackend(tmp_path / "local.db")
        local.index(records)
        keyword_store = PGKeywordStore(pg_conn)
        PGVectorStore(pg_conn).upsert(records, "lexical-parity", 4)
        keyword_store.index(records)

        cases = [
            ('"alpha beta"', None),
            ("alph*", None),
            ("src/searchkernel/search.py", None),
            ("parse_record", None),
            ("common", {"workspace_id": "workspace-a", "statuses": ["active"]}),
            ("", None),
        ]
        for query, filters in cases:
            local_keys = [hit.storage_key for hit in local.search_keyword(query, 10, filters)]
            pg_keys = [hit.storage_key for hit in keyword_store.search(query, 10, filters)]
            assert set(pg_keys) == set(local_keys)

        tie_records = [record for record in records if record.source_id in {"active", "other-workspace"}]
        assert [hit.storage_key for hit in keyword_store.search("common", 10)] == sorted(
            record.storage_key for record in tie_records
        )

    def test_keyword_search(self, pg_conn, fixture_records):
        """Test full-text search returns expected results."""
        vector_store = PGVectorStore(pg_conn)
        keyword_store = PGKeywordStore(pg_conn)

        # First upsert records (populates records table)
        vector_store.upsert(fixture_records, model_name="test-model", dim=4)

        # Index for keyword search
        keyword_store.index(fixture_records)

        # Search for "machine learning" should return top result
        results = keyword_store.search("machine learning", k=3)
        assert len(results) > 0

        # First result should be about machine learning
        top_id = results[0][0]
        top_record = next(r for r in fixture_records if r.source_id == top_id)
        assert "machine" in top_record.body.lower()

    def test_repeated_keyword_index_skips_unchanged_projection(
        self, pg_conn, fixture_records
    ):
        """Repeated keyword indexing preserves the projection and epoch.

        The stored record timestamp is also unchanged by the clean skip.
        """
        vector_store = PGVectorStore(pg_conn)
        keyword_store = PGKeywordStore(pg_conn)
        record = fixture_records[0]

        vector_store.upsert([record], model_name="keyword-idempotency", dim=4)
        keyword_store.index([record])
        before = (
            keyword_store.keyword_epoch(),
            pg_conn.execute_one(
                "SELECT tsvector_body::text, updated_at FROM records "
                "WHERE record_id = %s;",
                (record.storage_key,),
            ),
        )

        keyword_store.index([record])

        assert keyword_store.keyword_epoch() == before[0]
        assert pg_conn.execute_one(
            "SELECT tsvector_body::text, updated_at FROM records "
            "WHERE record_id = %s;",
            (record.storage_key,),
        ) == before[1]

    def test_keyword_index_updates_changed_and_missing_projections(
        self, pg_conn, fixture_records
    ):
        """Mixed keyword batches update changed fields and missing rows.

        Title, body, URI, metadata, and an unchanged projection are covered.
        """
        vector_store = PGVectorStore(pg_conn)
        keyword_store = PGKeywordStore(pg_conn)
        unchanged = fixture_records[0]
        original = fixture_records[1]
        vector_store.upsert([unchanged, original], model_name="keyword-mixed", dim=4)
        keyword_store.index([unchanged, original])
        unchanged_projection = pg_conn.execute_one(
            "SELECT tsvector_body::text FROM records WHERE record_id = %s;",
            (unchanged.storage_key,),
        )

        changed = replace(
            original,
            title="changedtitleterm",
            body="changedbodyterm",
            uri="docs/changeduripath",
            metadata={"marker": "changedmetadataterm"},
        )
        missing = replace(
            unchanged,
            source_id="keyword-missing",
            title="missingtitleterm",
            body="missingbodyterm",
            uri="docs/missinguripath",
            metadata={"marker": "missingmetadataterm"},
        )
        vector_store.upsert(
            [unchanged, changed, missing], model_name="keyword-mixed", dim=4
        )
        before = keyword_store.keyword_epoch()

        keyword_store.index([unchanged, changed, missing])

        assert keyword_store.keyword_epoch() == before + 1
        assert pg_conn.execute_one(
            "SELECT tsvector_body::text FROM records WHERE record_id = %s;",
            (unchanged.storage_key,),
        ) == unchanged_projection
        for term, expected_id in (
            ("changedtitleterm", "test:2"),
            ("changedbodyterm", "test:2"),
            ("docs/changeduripath", "test:2"),
            ("changedmetadataterm", "test:2"),
            ("missingtitleterm", "keyword-missing"),
            ("missingbodyterm", "keyword-missing"),
            ("docs/missinguripath", "keyword-missing"),
            ("missingmetadataterm", "keyword-missing"),
        ):
            assert [hit.source_id for hit in keyword_store.search(term, 10)] == [
                expected_id
            ]

    def test_keyword_index_duplicate_storage_keys_keep_last_projection(
        self, pg_conn, fixture_records
    ):
        """Duplicate keyword identities use the final record in the batch."""
        first = fixture_records[0]
        last = replace(first, title="lastkeywordterm", body="lastbodyterm")
        vector_store = PGVectorStore(pg_conn)
        keyword_store = PGKeywordStore(pg_conn)

        vector_store.upsert([first, last], model_name="keyword-duplicates", dim=4)
        before = keyword_store.keyword_epoch()
        keyword_store.index([first, last])

        assert keyword_store.keyword_epoch() == before + 1
        assert [hit.source_id for hit in keyword_store.search("lastkeywordterm", 10)] == [
            first.source_id
        ]
        assert keyword_store.search("machine", 10) == []

    @pytest.mark.parametrize("keyword_first", [True, False])
    def test_vector_upsert_preserves_weighted_keyword_projection(
        self, pg_conn, keyword_first
    ):
        """Vector writes preserve the keyword projection in either order.

        The initial vector write creates the records before either index lane
        updates the shared row.
        """
        timestamp = datetime.now(UTC)
        records = [
            Record(
                source_kind="test",
                source_id="title-match",
                title="projection-token",
                body="ordinary body",
                uri="docs/uniquepath.md",
                metadata={"marker": "uniquemetadata"},
                created_at=timestamp,
                updated_at=timestamp,
                embedding=[1.0, 0.0, 0.0, 0.0],
            ),
            Record(
                source_kind="test",
                source_id="body-match",
                title="ordinary title",
                body="projection-token",
                created_at=timestamp,
                updated_at=timestamp,
                embedding=[0.0, 1.0, 0.0, 0.0],
            ),
        ]
        vector_store = PGVectorStore(pg_conn)
        keyword_store = PGKeywordStore(pg_conn)

        vector_store.upsert(records, model_name="test-model", dim=4)
        if keyword_first:
            keyword_store.index(records)
            vector_store.upsert(records, model_name="test-model", dim=4)
        else:
            vector_store.upsert(records, model_name="test-model", dim=4)
            keyword_store.index(records)

        projection_hits = keyword_store.search("projection-token", 10)
        assert [hit.source_id for hit in projection_hits] == [
            "title-match",
            "body-match",
        ]
        assert [
            hit.source_id
            for hit in keyword_store.search("docs/uniquepath.md", 10)
        ] == ["title-match"]
        assert [hit.source_id for hit in keyword_store.search("marker", 10)] == [
            "title-match"
        ]

    def test_keyword_search_uses_indexed_text_with_raw_body_persisted(self, pg_conn):
        """Search uses the override while records retain citation text."""
        timestamp = datetime.now(UTC)
        record = Record(
            source_kind="test",
            source_id="indexed",
            title="Indexed text",
            body="Raw citation body",
            indexed_text="search-only vocabulary",
            created_at=timestamp,
            updated_at=timestamp,
            embedding=[1.0, 0.0, 0.0, 0.0],
        )
        vector_store = PGVectorStore(pg_conn)
        keyword_store = PGKeywordStore(pg_conn)

        vector_store.upsert([record], model_name="test-model", dim=4)
        keyword_store.index([record])

        assert [hit.source_id for hit in keyword_store.search("vocabulary", 10)] == [
            "indexed"
        ]
        assert keyword_store.search("citation", 10) == []
        conn = pg_conn.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT body, indexed_text FROM records WHERE record_id = %s;",
            (record.storage_key,),
        )
        assert cursor.fetchone() == ("Raw citation body", "search-only vocabulary")
        cursor.close()
        pg_conn.put_connection(conn)

    def test_upsert_preserves_body_fallback_for_empty_indexed_text(self, pg_conn):
        """Keyword indexing owns body fallback across vector upserts."""
        timestamp = datetime.now(UTC)

        def make_record(
            source_id: str, body: str, indexed_text: str
        ) -> Record:
            return Record(
                source_kind="test",
                source_id=source_id,
                title=f"Record {source_id}",
                body=body,
                indexed_text=indexed_text,
                created_at=timestamp,
                updated_at=timestamp,
                embedding=[1.0, 0.0, 0.0, 0.0],
            )

        initial = [
            make_record("empty", "fallback alpha", ""),
            make_record("override", "raw beta", "indexed gamma"),
        ]
        updated = [
            make_record("empty", "fallback delta", ""),
            make_record("override", "raw epsilon", "indexed zeta"),
        ]
        vector_store = PGVectorStore(pg_conn)
        keyword_store = PGKeywordStore(pg_conn)

        vector_store.upsert(initial, model_name="test-model", dim=4)
        keyword_store.index(initial)
        matching = lambda query: {
            hit.source_id for hit in keyword_store.search(query, 10)
        }
        assert matching("alpha") == {"empty"}
        assert matching("beta") == set()
        assert matching("gamma") == {"override"}

        vector_store.upsert(updated, model_name="test-model", dim=4)
        assert matching("alpha") == {"empty"}
        assert matching("delta") == set()
        assert matching("gamma") == {"override"}
        assert matching("zeta") == set()

        keyword_store.index(updated)
        assert matching("delta") == {"empty"}
        assert matching("epsilon") == set()
        assert matching("zeta") == {"override"}

    def test_keyword_search_with_filters(self, pg_conn, fixture_records):
        """Test keyword search with source_kind filter."""
        vector_store = PGVectorStore(pg_conn)
        keyword_store = PGKeywordStore(pg_conn)

        vector_store.upsert(fixture_records, model_name="test-model", dim=4)
        keyword_store.index(fixture_records)

        # Search with filter for "database" category but only in ai source_kind
        results = keyword_store.search(
            "database",
            k=10,
            filters={"source_kinds": ["test"]},
        )

        # Should find the database record
        assert len(results) > 0
        assert results[0][0] == "test:3"

    def test_keyword_search_weights_title_above_indexed_body(self, pg_conn):
        """Title matches rank above equivalent indexed-body matches."""
        now = datetime.now(UTC)
        records = [
            Record(
                source_kind="test",
                source_id="title-match",
                title="PostgreSQL",
                body="A guide to database systems.",
                created_at=now,
                updated_at=now,
                embedding=[1.0, 0.0, 0.0, 0.0],
            ),
            Record(
                source_kind="test",
                source_id="body-match",
                title="Database systems",
                body="PostgreSQL",
                created_at=now,
                updated_at=now,
                embedding=[0.0, 1.0, 0.0, 0.0],
            ),
        ]
        vector_store = PGVectorStore(pg_conn)
        keyword_store = PGKeywordStore(pg_conn)

        vector_store.upsert(records, model_name="test-model", dim=4)
        keyword_store.index(records)

        results = keyword_store.search("PostgreSQL", k=2)

        assert [hit.source_id for hit in results] == ["title-match", "body-match"]

    def test_keyword_index_ignores_missing_records(self, pg_conn):
        """Unknown records do not create searchable data or epoch changes."""
        record = Record(
            source_kind="missing",
            source_id="missing:1",
            title="Missing",
            body="This record is not in the records table",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        keyword_store = PGKeywordStore(pg_conn)
        before = keyword_store.keyword_epoch()

        keyword_store.index([record])

        assert keyword_store.keyword_epoch() == before
        assert keyword_store.search("missing", k=10) == []


class TestGraphStore:
    """Tests for GraphStore port implementation."""

    def test_upsert_and_retrieve_edges(self, pg_conn):
        """Test edge upsert and neighbor retrieval."""
        store = PGGraphStore(pg_conn)
        identities = {
            source_id: RecordIdentity(None, "test", source_id)
            for source_id in ("test:1", "test:2", "test:3")
        }

        edges = [
            GraphEdge(identities["test:1"], identities["test:2"], "related", 0.9),
            GraphEdge(identities["test:1"], identities["test:3"], "related", 0.5),
            GraphEdge(
                identities["test:2"], identities["test:3"], "derived_from", 0.7
            ),
        ]

        store.upsert_edges(edges)

        # Get neighbors of test:1
        neighbors = store.neighbors(identities["test:1"])
        neighbor_ids = [neighbor.source_id for neighbor in neighbors]

        assert "test:2" in neighbor_ids
        assert "test:3" in neighbor_ids

    def test_incoming_neighbors_return_sources_of_target_edges(self, pg_conn):
        store = PGGraphStore(pg_conn)
        target = RecordIdentity("workspace-a", "note", "target")
        inbound = RecordIdentity("workspace-a", "note", "inbound")
        outbound = RecordIdentity("workspace-a", "note", "outbound")
        store.upsert_edges(
            [
                GraphEdge(inbound, target, "links_to", 0.9),
                GraphEdge(target, outbound, "links_to", 0.8),
            ]
        )

        assert store.incoming_neighbors(target) == [
            GraphNeighbor(inbound, "links_to", 0.9)
        ]
        assert store.incoming_neighbors_many([target], depth=1) == {
            target.storage_key: [
                GraphNeighbor(inbound, "links_to", 0.9)
            ]
        }

    def test_batch_graph_traversal_preserves_direction_identity_and_order(
        self, pg_conn
    ):
        """Batch traversal handles multiple seeds and recursive graph semantics.

        Incoming traversal must preserve canonical seed keys, best-path weights,
        cycle prevention, and deterministic ties while matching single-seed
        outgoing traversal.
        """
        store = PGGraphStore(pg_conn)
        target_a = RecordIdentity("workspace-a", "note", "shared")
        target_b = RecordIdentity("workspace-b", "note", "shared")
        inbound_a = RecordIdentity("workspace-a", "note", "inbound-a")
        inbound_b = RecordIdentity("workspace-a", "note", "inbound-b")
        upstream = RecordIdentity("workspace-a", "note", "upstream")

        store.upsert_edges(
            [
                GraphEdge(inbound_b, target_a, "links", 0.9),
                GraphEdge(inbound_a, target_a, "links", 0.9),
                GraphEdge(upstream, inbound_a, "links", 0.8),
                GraphEdge(target_a, inbound_a, "links", 0.7),
                GraphEdge(target_a, target_b, "links", 0.6),
            ]
        )

        expected_a = [
            GraphNeighbor(inbound_a, "links", 0.9),
            GraphNeighbor(inbound_b, "links", 0.9),
            GraphNeighbor(upstream, "links", 0.72),
        ]
        expected_b = [
            GraphNeighbor(target_a, "links", 0.6),
            GraphNeighbor(inbound_a, "links", 0.54),
            GraphNeighbor(inbound_b, "links", 0.54),
            GraphNeighbor(upstream, "links", 0.432),
        ]

        incoming = store.incoming_neighbors_many(
            [target_a, target_a, target_b], depth=3
        )
        assert list(incoming) == [target_a.storage_key, target_b.storage_key]
        for seed, expected in ((target_a, expected_a), (target_b, expected_b)):
            assert [neighbor.identity for neighbor in incoming[seed.storage_key]] == [
                neighbor.identity for neighbor in expected
            ]
            assert [neighbor.edge_type for neighbor in incoming[seed.storage_key]] == [
                neighbor.edge_type for neighbor in expected
            ]
            assert [neighbor.weight for neighbor in incoming[seed.storage_key]] == pytest.approx(
                [neighbor.weight for neighbor in expected]
            )
        assert store.incoming_neighbors_many([], depth=1) == {}
        assert store.neighbors_many(
            [inbound_a, inbound_b], depth=1
        ) == {
            inbound_a.storage_key: [
                GraphNeighbor(target_a, "links", 0.9)
            ],
            inbound_b.storage_key: [
                GraphNeighbor(target_a, "links", 0.9)
            ],
        }
        assert store.neighbors_many([inbound_a], depth=2) == {
            inbound_a.storage_key: list(store.neighbors(inbound_a, depth=2))
        }

    def test_edge_upsert_updates_weight_and_missing_neighbors_are_empty(self, pg_conn):
        """Test graph edge conflict updates and empty lookups."""
        store = PGGraphStore(pg_conn)
        source = RecordIdentity(None, "test", "source")
        target = RecordIdentity(None, "test", "target")
        store.upsert_edges([GraphEdge(source, target, "related", 0.9)])
        store.upsert_edges([GraphEdge(source, target, "related", 0.2)])

        assert store.neighbors(source, depth=3) == [
            GraphNeighbor(target, "related", 0.2)
        ]
        assert store.neighbors(RecordIdentity(None, "test", "missing")) == []

    def test_neighbors_with_edge_type_filter(self, pg_conn):
        """Test neighbor retrieval with edge type filter."""
        store = PGGraphStore(pg_conn)
        identities = {
            source_id: RecordIdentity(None, "test", source_id)
            for source_id in ("test:1", "test:2", "test:3")
        }

        edges = [
            GraphEdge(identities["test:1"], identities["test:2"], "related", 0.9),
            GraphEdge(
                identities["test:1"], identities["test:3"], "derived_from", 0.5
            ),
        ]

        store.upsert_edges(edges)

        # Get neighbors only of type "related"
        neighbors = store.neighbors(identities["test:1"], edge_types=["related"])
        neighbor_ids = [neighbor.source_id for neighbor in neighbors]

        assert "test:2" in neighbor_ids
        assert "test:3" not in neighbor_ids

    def test_neighbors_preserve_duplicate_ids_across_workspaces(self, pg_conn):
        store = PGGraphStore(pg_conn)
        start = RecordIdentity("workspace-a", "note", "start")
        first_shared = RecordIdentity("workspace-b", "note", "shared")
        second_shared = RecordIdentity("workspace-c", "note", "shared")

        store.upsert_edges(
            [
                GraphEdge(start, first_shared, "links", 0.9),
                GraphEdge(first_shared, second_shared, "links", 0.8),
            ]
        )

        neighbors = store.neighbors(start, depth=2)

        assert [neighbor.identity for neighbor in neighbors] == [
            first_shared,
            second_shared,
        ]

    def test_neighbors_stop_cycles_without_returning_the_seed(self, pg_conn):
        store = PGGraphStore(pg_conn)
        start = RecordIdentity("workspace-a", "note", "start")
        target = RecordIdentity("workspace-b", "note", "target")

        store.upsert_edges(
            [
                GraphEdge(start, target, "links", 0.9),
                GraphEdge(target, start, "links", 0.8),
            ]
        )

        neighbors = store.neighbors(start, depth=3)

        assert [neighbor.identity for neighbor in neighbors] == [target]

    def test_neighbors_order_ties_by_complete_identity(self, pg_conn):
        store = PGGraphStore(pg_conn)
        start = RecordIdentity("workspace-a", "note", "start")
        earlier = RecordIdentity("workspace-b", "note", "shared")
        later = RecordIdentity("workspace-c", "note", "shared")

        store.upsert_edges(
            [
                GraphEdge(start, later, "links", 0.5),
                GraphEdge(start, earlier, "links", 0.5),
            ]
        )

        assert [neighbor.identity for neighbor in store.neighbors(start)] == [
            earlier,
            later,
        ]
        assert store.neighbors(start, max_neighbors=1) == [
            GraphNeighbor(earlier, "links", 0.5)
        ]

class TestCacheStore:
    """Tests for CacheStore port implementation."""

    def test_get_set(self, pg_conn):
        """Test basic cache get/set."""
        store = PGCacheStore(pg_conn)

        store.set("key1", {"data": "value1"}, epoch=0)

        result = store.get("key1")
        assert result == {"data": "value1"}

    def test_cache_miss(self, pg_conn):
        """Test cache miss returns None."""
        store = PGCacheStore(pg_conn)

        result = store.get("nonexistent")
        assert result is None

    def test_epoch_invalidation(self, pg_conn):
        """Test epoch-based invalidation."""
        store = PGCacheStore(pg_conn)

        # Set values with different epochs
        store.set("key1", {"data": "value1"}, epoch=0)
        store.set("key2", {"data": "value2"}, epoch=1)
        store.set("key3", {"data": "value3"}, epoch=2)

        # Both should exist
        assert store.get("key1") is not None
        assert store.get("key2") is not None
        assert store.get("key3") is not None

        # Invalidate epochs 0 and 1
        store.invalidate_epoch(1)

        # key1 and key2 should be gone; key3 should remain
        assert store.get("key1") is None
        assert store.get("key2") is None
        assert store.get("key3") is not None

    def test_cache_update(self, pg_conn):
        """Test that set overwrites existing values."""
        store = PGCacheStore(pg_conn)

        store.set("key1", {"data": "value1"}, epoch=0)
        store.set("key1", {"data": "value2"}, epoch=1)

        result = store.get("key1")
        assert result == {"data": "value2"}


class TestRoundTrip:
    """Integration tests for complete workflows."""

    def test_full_workflow(self, pg_conn, fixture_records):
        """Test a complete upsert -> search -> delete workflow."""
        vector_store = PGVectorStore(pg_conn)
        keyword_store = PGKeywordStore(pg_conn)
        cache_store = PGCacheStore(pg_conn)

        # Upsert vectors
        vector_store.upsert(fixture_records, model_name="test-model", dim=4)
        keyword_store.index(fixture_records)

        epoch_before = vector_store.epoch()

        # Cache a result
        cache_store.set("search:1", {"results": ["test:1"]}, epoch=epoch_before)

        # Search
        results = vector_store.search(
            [1.0, 0.0, 0.0, 0.0], k=2, model_name="test-model", dim=4
        )
        assert len(results) > 0

        # Delete a record
        vector_store.delete([fixture_records[0].storage_key])

        epoch_after = vector_store.epoch()
        assert epoch_after > epoch_before

        # Cache should still be there (epoch hasn't changed for it)
        cached = cache_store.get("search:1")
        assert cached is not None

        # After invalidation, cache should be gone
        cache_store.invalidate_epoch(epoch_before)
        cached = cache_store.get("search:1")
        assert cached is None
