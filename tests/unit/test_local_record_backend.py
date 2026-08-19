import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

import searchkernel.indices.local as local_indices
from searchkernel.domain import GraphNeighbor, Record, RecordIdentity, RecordStatus
from searchkernel.indices import (
    LocalGraphStore,
    LocalKeywordStore,
    LocalRecordBackend,
    LocalVectorStore,
)
from searchkernel.search.orchestrator import SearchOrchestrator
from searchkernel.storage.db import DatabaseManager


class _Embedder:
    model_name = "test"
    dim = 2

    def embed_query(self, query: str) -> list[float]:
        return [1.0, 0.0] if query == "first" else [0.0, 1.0]


def _record(
    source_kind: str,
    source_id: str,
    body: str,
    *,
    workspace_id: str | None = None,
    status: RecordStatus = RecordStatus.ACTIVE,
    title: str | None = None,
    uri: str | None = None,
    metadata: dict | None = None,
    indexed_text: str | None = None,
) -> Record:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    return Record(
        workspace_id=workspace_id,
        source_kind=source_kind,
        source_id=source_id,
        title=title or source_id,
        body=body,
        created_at=timestamp,
        updated_at=timestamp,
        status=status,
        uri=uri,
        metadata=metadata or {},
        indexed_text=indexed_text,
    )


def _backend(tmp_path):
    backend = LocalRecordBackend(tmp_path / "records.db")
    return (
        backend,
        LocalKeywordStore(backend),
        LocalVectorStore(backend),
        LocalGraphStore(backend),
    )


def test_local_store_uses_collision_safe_identity_and_filters(tmp_path) -> None:
    backend, keyword, vector, graph = _backend(tmp_path)
    records = [
        _record("note", "same", "first alpha", workspace_id="one"),
        _record("commit", "same", "first beta", workspace_id="one"),
        _record("note", "same", "first hidden", workspace_id="two"),
        _record("note", "archived", "first archived", status=RecordStatus.ARCHIVED),
    ]
    for record in records:
        record.embedding = [1.0, 0.0]
    vector.upsert(records, "test", 2)
    keyword.index(records)
    graph.upsert_edges([(records[0].storage_key, records[1].storage_key, "related", 0.5)])

    hits = keyword.search("first", 10, {"workspace_id": "one"})
    vector_hits = vector.search(
        [1.0, 0.0], 10, model_name="test", dim=2, filters={"workspace_id": "one"}
    )

    assert keyword.keyword_index_available
    assert keyword.check_keyword_index()
    assert [hit.storage_key for hit in hits] == [
        records[1].storage_key,
        records[0].storage_key,
    ]
    assert [hit.storage_key for hit in vector_hits] == [
        records[1].storage_key,
        records[0].storage_key,
    ]
    hydrated = backend.hydrate_record(records[0].storage_key)
    assert hydrated is not None
    assert hydrated.storage_key == records[0].storage_key
    assert hydrated.body == records[0].body
    assert backend.hydrate_record("same") is None
    assert graph.neighbors(records[0].storage_key) == [
        GraphNeighbor(RecordIdentity("one", "commit", "same"), "related", 0.5)
    ]
    assert graph.neighbors(
        RecordIdentity("one", "note", "same")
    ) == [GraphNeighbor(RecordIdentity("one", "commit", "same"), "related", 0.5)]


def test_direct_backend_close_releases_owned_database(tmp_path) -> None:
    backend = LocalRecordBackend(tmp_path / "records.db")

    backend.close()

    with pytest.raises(RuntimeError, match="database manager is closed"):
        backend.db_manager.get_connection()


def test_injected_database_remains_owned_by_caller(tmp_path) -> None:
    database = DatabaseManager(tmp_path / "records.db")
    backend = LocalRecordBackend(db_manager=database)

    backend.close()

    assert database.get_connection().execute("SELECT 1").fetchone()[0] == 1
    database.close()


def test_local_keyword_searches_indexed_text_hydrates_raw_body(tmp_path) -> None:
    backend = LocalRecordBackend(tmp_path / "records.db")
    record = _record(
        "note",
        "indexed",
        "Raw citation body",
        indexed_text="search-only vocabulary",
    )

    backend.index([record])

    assert [hit.source_id for hit in backend.search_keyword("vocabulary", 10)] == [
        "indexed"
    ]
    assert backend.search_keyword("citation", 10) == []
    hydrated = backend.hydrate_record(record.storage_key)
    assert hydrated is not None
    assert hydrated.body == "Raw citation body"
    assert hydrated.indexed_text == "search-only vocabulary"


def test_repeated_canonical_sync_preserves_results_without_keyword_epoch_changes(
    tmp_path,
) -> None:
    """Identical keyword and vector syncs preserve records.

    Keyword synchronization must not churn canonical or keyword epochs.
    """
    backend, _keyword, vector, _graph = _backend(tmp_path)
    record = _record("note", "repeat", "raw body", indexed_text="findable text")
    record.embedding = [1.0, 0.0]

    vector.upsert([record], "test", 2)
    before = backend.epochs()
    before_record_epoch = backend.epoch()
    queries: list[str] = []
    connection = backend.db_manager.get_connection()
    connection.set_trace_callback(queries.append)
    vector.upsert([record], "test", 2)
    connection.set_trace_callback(None)
    after_vector = backend.epochs()
    backend.index([record])

    assert backend.epoch() == before_record_epoch + 1
    assert after_vector["vector"] == before["vector"]
    assert not any(
        "local_vectors_v2" in query and "INSERT" in query for query in queries
    )
    assert backend.epochs() == after_vector
    assert [hit.source_id for hit in backend.search_keyword("findable", 10)] == [
        "repeat"
    ]
    hydrated = backend.hydrate_record(record.storage_key)
    assert hydrated is not None
    assert hydrated.body == "raw body"


def test_changed_indexed_fields_update_keyword_results_and_hydrated_record(
    tmp_path,
) -> None:
    """Indexed text, metadata keywords, and URI changes replace search data.

    Hydration must expose the new canonical values after replacement.
    """
    backend = LocalRecordBackend(tmp_path / "records.db")
    record = _record(
        "note",
        "changed",
        "raw body",
        indexed_text="old indexed text",
        uri="docs/old-uri.md",
        metadata={"tags": ["old-tag"]},
    )
    backend.index([record])

    record.indexed_text = "new indexed text"
    record.uri = "docs/new-uri.md"
    record.metadata = {"tags": ["new-tag"]}
    backend.index([record])

    assert backend.search_keyword("old", 10) == []
    assert [hit.source_id for hit in backend.search_keyword("new", 10)] == [
        "changed"
    ]
    assert backend.search_keyword("old-tag", 10) == []
    assert [hit.source_id for hit in backend.search_keyword("new-tag", 10)] == [
        "changed"
    ]
    assert backend.search_keyword("old-uri", 10) == []
    assert [hit.source_id for hit in backend.search_keyword("new-uri", 10)] == [
        "changed"
    ]
    hydrated = backend.hydrate_record(record.storage_key)
    assert hydrated is not None
    assert hydrated.indexed_text == "new indexed text"
    assert hydrated.uri == "docs/new-uri.md"


def test_standalone_vector_upsert_creates_searchable_canonical_record(tmp_path) -> None:
    """A vector-only write persists the canonical row.

    The row must be available to keyword retrieval without a separate index call.
    """
    backend = LocalRecordBackend(tmp_path / "records.db")
    record = _record("note", "vector-only", "vector-created body")
    record.embedding = [1.0, 0.0]

    backend.upsert([record], "test", 2)

    assert [hit.source_id for hit in backend.search_keyword("vector-created", 10)] == [
        "vector-only"
    ]
    assert backend.hydrate_record(record.storage_key) is not None


def test_chunk_replacement_removes_stale_records_from_search_and_state(tmp_path) -> None:
    """Replacing chunks removes stale canonical rows and keyword hits.

    Chunk state must contain only the replacement set for the parent.
    """
    backend = LocalRecordBackend(tmp_path / "records.db")
    parent = _record("note", "parent", "parent body")

    def chunk(chunk_id: str, content: str, index: int) -> Record:
        return _record(
            "note",
            f"parent#chunk:{chunk_id}",
            content,
            indexed_text=content,
            metadata={
                "_searchkernel_chunk": True,
                "_chunk_parent_storage_key": parent.storage_key,
                "_chunk_id": chunk_id,
                "_chunk_index": index,
                "_chunk_metadata": {"header_path": chunk_id},
            },
        )

    first = chunk("first", "first needle", 0)
    stale = chunk("stale", "stale needle", 1)
    replacement = chunk("replacement", "replacement needle", 0)
    backend.index([parent, first, stale])
    backend.index([parent, replacement])

    assert backend.search_keyword("stale", 10) == []
    assert [hit.source_id for hit in backend.search_keyword("replacement", 10)] == [
        replacement.source_id
    ]
    assert list(backend.chunk_records(parent.identity)) == [replacement.storage_key]


def test_local_batch_hydration_and_graph_preserve_canonical_keys(tmp_path) -> None:
    backend, _keyword, _vector, graph = _backend(tmp_path)
    source = _record("note", "source", "source", workspace_id="one")
    target = _record("commit", "target", "target", workspace_id="two")
    source_identity = RecordIdentity("one", "note", "source")
    target_identity = RecordIdentity("two", "commit", "target")
    backend.index([source, target])
    graph.upsert_edges([(source.storage_key, target.storage_key, "related", 0.5)])

    hydrated = backend.hydrate_records([source_identity, target_identity])
    neighbors = graph.neighbors_many([source_identity], depth=1)

    hydrated_records = [
        hydrated[identity.storage_key]
        for identity in (source_identity, target_identity)
    ]
    assert all(record is not None for record in hydrated_records)
    assert [
        record.storage_key
        for record in hydrated_records
        if record is not None
    ] == [source.storage_key, target.storage_key]
    assert neighbors == {
        source.storage_key: [GraphNeighbor(target_identity, "related", 0.5)]
    }


def test_local_batch_hydration_chunks_large_identity_lists(tmp_path) -> None:
    backend = LocalRecordBackend(tmp_path / "records.db")
    records = [_record("note", f"record-{index}", "body") for index in range(901)]
    backend.index(records)

    hydrated = backend.hydrate_records([record.identity for record in records])

    assert list(hydrated) == [record.storage_key for record in records]
    assert all(hydrated[record.storage_key] is not None for record in records)


def test_local_record_operations_chunk_large_key_lists(tmp_path) -> None:
    """Large record and graph batches stay below SQLite's variable limit."""
    backend, _keyword, _vector, graph = _backend(tmp_path)
    records = [_record("note", f"record-{index}", "body") for index in range(1001)]
    backend.index(records)

    graph.upsert_edges(
        [
            (records[index].storage_key, records[index + 1].storage_key, "next", 1.0)
            for index in range(len(records) - 1)
        ]
    )

    assert graph.neighbors(records[0].identity) == [
        GraphNeighbor(records[1].identity, "next", 1.0)
    ]


def test_local_graph_top_neighbors_are_bounded_and_deterministic(tmp_path) -> None:
    backend, _keyword, _vector, graph = _backend(tmp_path)
    source = _record("note", "source", "source")
    targets = [
        _record("note", "target-a", "target-a"),
        _record("note", "target-b", "target-b"),
        _record("note", "target-c", "target-c"),
    ]
    backend.index([source, *targets])
    graph.upsert_edges(
        [
            (source.storage_key, targets[2].storage_key, "related", 0.5),
            (source.storage_key, targets[1].storage_key, "related", 0.9),
            (source.storage_key, targets[0].storage_key, "related", 0.9),
        ]
    )

    expected = [
        GraphNeighbor(targets[0].identity, "related", 0.9),
        GraphNeighbor(targets[1].identity, "related", 0.9),
    ]
    assert graph.neighbors(source.identity, max_neighbors=2) == expected
    assert graph.neighbors_many(
        [source.identity], depth=1, max_neighbors=2
    ) == {source.storage_key: expected}


def test_local_graph_supports_incoming_neighbors_with_canonical_order(tmp_path) -> None:
    backend, _keyword, _vector, graph = _backend(tmp_path)
    target = _record("note", "target", "target", workspace_id="project-a")
    inbound_a = _record("note", "inbound-a", "inbound a", workspace_id="project-a")
    inbound_b = _record("note", "inbound-b", "inbound b", workspace_id="project-a")
    outbound = _record("note", "outbound", "outbound", workspace_id="project-a")
    outsider = _record("note", "outsider", "outsider", workspace_id="project-b")
    backend.index([target, inbound_a, inbound_b, outbound, outsider])
    graph.upsert_edges(
        [
            (inbound_b.storage_key, target.storage_key, "links_to", 0.9),
            (inbound_a.storage_key, target.storage_key, "links_to", 0.9),
            (target.storage_key, outbound.storage_key, "links_to", 1.0),
            (outsider.storage_key, target.storage_key, "links_to", 2.0),
        ]
    )

    expected = [
        GraphNeighbor(outsider.identity, "links_to", 2.0),
        GraphNeighbor(inbound_a.identity, "links_to", 0.9),
        GraphNeighbor(inbound_b.identity, "links_to", 0.9),
    ]
    assert graph.incoming_neighbors(target.identity) == expected
    assert graph.incoming_neighbors_many([target.identity], depth=1) == {
        target.storage_key: expected
    }
    assert graph.incoming_neighbors(inbound_a.identity) == []


def test_keyword_search_supports_tokens_phrases_prefixes_artifacts_and_symbols(
    tmp_path,
) -> None:
    backend = LocalRecordBackend(tmp_path / "records.db")
    records = [
        _record(
            "note",
            "guide",
            "alpha beta phrase",
            title="Alpha Guide",
            uri="docs/guide.py",
        ),
        _record("note", "prefix", "alphabet soup", title="Alphabet"),
        _record("note", "symbol", "parse_record implementation", title="Parser"),
        _record("note", "substring", "concatenate only", title="Other"),
    ]
    backend.index(records)

    assert [hit.source_id for hit in backend.search_keyword("alpha", 10)] == ["guide"]
    assert [
        hit.source_id for hit in backend.search_keyword('"alpha beta"', 10)
    ] == ["guide"]
    assert {
        hit.source_id for hit in backend.search_keyword("alph*", 10)
    } == {"guide", "prefix"}
    assert [hit.source_id for hit in backend.search_keyword("guide.py", 10)] == [
        "guide"
    ]
    assert [hit.source_id for hit in backend.search_keyword("parse_record", 10)] == [
        "symbol"
    ]


def test_keyword_search_falls_back_for_natural_language_queries(tmp_path) -> None:
    backend = LocalRecordBackend(tmp_path / "records.db")
    records = [
        _record("note", "alpha", "alpha"),
        _record("note", "beta", "beta"),
    ]
    backend.index(records)

    assert [hit.source_id for hit in backend.search_keyword("alpha beta", 10)] == [
        "alpha",
        "beta",
    ]


def test_keyword_search_keeps_artifact_queries_exact(tmp_path) -> None:
    backend = LocalRecordBackend(tmp_path / "records.db")
    backend.index(
        [
            _record("note", "alpha", "alpha"),
            _record("note", "beta", "beta"),
        ]
    )

    assert backend.search_keyword("alpha.beta", 10) == []


def test_keyword_search_recovers_bounded_technical_typos(tmp_path) -> None:
    backend = LocalRecordBackend(tmp_path / "records.db")
    backend.index(
        [
            _record(
                "note",
                "fusion",
                "The reciprocal rank fusion algorithm",
            ),
            _record("note", "other", "A reciprocal ranking overview"),
        ]
    )

    assert [hit.source_id for hit in backend.search_keyword(
        "reciprical rank fuson", 10
    )] == ["fusion"]
    assert backend.search_keyword('"reciprical rank fuson"', 10) == []
    assert backend.search_keyword("reciprical.py", 10) == []


def test_keyword_fuzzy_search_includes_uri_and_keyword_fields(tmp_path) -> None:
    backend = LocalRecordBackend(tmp_path / "records.db")
    keyword_record = _record(
        "note",
        "keyword-only",
        "Unrelated body",
        metadata={"keywords": ["reciprocal", "rank", "fusion"]},
    )
    uri_record = _record(
        "note",
        "uri-only",
        "Unrelated body",
        uri="docs/reciprocal-rank-fusion.md",
    )
    backend.index([keyword_record, uri_record])

    assert {
        hit.source_id
        for hit in backend.search_keyword("reciprical rank fuson", 10)
    } == {"keyword-only", "uri-only"}


def test_keyword_fuzzy_search_narrows_similarity_candidates(
    tmp_path,
    monkeypatch,
) -> None:
    backend = LocalRecordBackend(tmp_path / "records.db")
    records = [
        _record("note", f"unrelated-{index}", "unrelated content")
        for index in range(1_000)
    ]
    records.append(
        _record(
            "note",
            "target",
            "The reciprocal rank fusion algorithm",
        )
    )
    backend.index(records)

    calls = 0
    original = local_indices._fuzzy_term_score

    def counted_score(query_term, tokens):
        nonlocal calls
        calls += 1
        return original(query_term, tokens)

    monkeypatch.setattr(local_indices, "_fuzzy_term_score", counted_score)

    assert [hit.source_id for hit in backend.search_keyword(
        "reciprical rank fuson", 10
    )] == ["target"]
    assert calls <= 3 * 256


def test_keyword_fuzzy_search_supports_unicode_tokens(tmp_path) -> None:
    """Unicode query tokens should participate in fuzzy fallback matching."""
    backend = LocalRecordBackend(tmp_path / "records.db")
    backend.index(
        [
            _record(
                "note",
                "multilingual",
                "поиск метаданных",
            )
        ]
    )

    assert [hit.source_id for hit in backend.search_keyword(
        "поиск метаданны", 10
    )] == ["multilingual"]


def test_keyword_fuzzy_search_matches_tokens_after_long_document_prefix(
    tmp_path,
) -> None:
    """Fuzzy matching should find tokens beyond the old document cutoff."""
    backend = LocalRecordBackend(tmp_path / "records.db")
    long_body = " ".join(f"filler{index}" for index in range(300))
    backend.index(
        [
            _record(
                "note",
                "long-document",
                f"{long_body} reciprocal rank fusion",
            )
        ]
    )

    assert [hit.source_id for hit in backend.search_keyword(
        "reciprical rank fuson", 10
    )] == ["long-document"]


def test_keyword_fuzzy_search_preserves_filtered_recall_beyond_batch(
    tmp_path,
) -> None:
    backend = LocalRecordBackend(tmp_path / "records.db")
    records = [
        _record(
            "note",
            f"excluded-{index:03d}",
            "The reciprocal rank fusion algorithm",
            metadata={"kind": "excluded"},
        )
        for index in range(300)
    ]
    records.append(
        _record(
            "note",
            "target",
            "The reciprocal rank fusion algorithm",
            metadata={"kind": "target"},
        )
    )
    backend.index(records)

    assert [
        hit.source_id
        for hit in backend.search_keyword(
            "reciprical rank fuson",
            1,
            {"metadata_equals": {"kind": "target"}},
        )
    ] == ["target"]


def test_keyword_search_applies_filters_to_natural_language_fallback(tmp_path) -> None:
    backend = LocalRecordBackend(tmp_path / "records.db")
    records = [
        _record(
            "note",
            "kept",
            "alpha",
            metadata={"project_id": "keep"},
        ),
        _record(
            "note",
            "excluded",
            "beta",
            metadata={"project_id": "drop"},
        ),
    ]
    backend.index(records)

    assert [
        hit.source_id
        for hit in backend.search_keyword(
            "alpha beta",
            10,
            {"project_filter": ["keep"]},
        )
    ] == ["kept"]


def test_keyword_scan_fallback_scans_large_corpora(tmp_path, monkeypatch) -> None:
    """
    The no-FTS fallback finds matches beyond its historical scan threshold.
    """
    backend = LocalRecordBackend(tmp_path / "records.db")
    records = [
        _record("note", f"unrelated-{index}", "unrelated")
        for index in range(10_001)
    ]
    records.append(_record("note", "target", "needle"))
    backend.index(records)
    monkeypatch.setattr(backend, "_fts5_available", False)

    assert [hit.source_id for hit in backend.search_keyword("needle", 1)] == [
        "target"
    ]
    assert backend.last_keyword_search_diagnostics == {
        "scanned": 10_002,
        "requested_k": 1,
        "returned": 1,
        "scan_complete": True,
        "fallback": True,
    }


def test_keyword_scan_fallback_returns_empty_for_large_no_match(tmp_path, monkeypatch) -> None:
    """
    The no-FTS fallback returns no hits when a large corpus has no match.
    """
    backend = LocalRecordBackend(tmp_path / "records.db")
    keyword = LocalKeywordStore(backend)
    backend.index(
        [
            _record("note", f"record-{index}", "unrelated")
            for index in range(10_001)
        ]
    )
    monkeypatch.setattr(backend, "_fts5_available", False)

    assert keyword.search("missing", 10) == []
    diagnostics = keyword.last_keyword_search_diagnostics
    assert diagnostics == {
        "scanned": 10_001,
        "requested_k": 10,
        "returned": 0,
        "scan_complete": True,
        "fallback": True,
    }
    with pytest.raises(TypeError):
        diagnostics["returned"] = 1


def test_keyword_scan_fallback_matches_uri_and_metadata_keywords(tmp_path, monkeypatch) -> None:
    """
    The fallback searches the same URI and metadata keyword fields as FTS.
    """
    backend = LocalRecordBackend(tmp_path / "records.db")
    backend.index(
        [
            _record(
                "note",
                "artifact",
                "unrelated body",
                uri="/repo/needle.py",
                metadata={"tags": ["keyword"]},
            )
        ]
    )
    monkeypatch.setattr(backend, "_fts5_available", False)

    assert [hit.source_id for hit in backend.search_keyword("needle", 10)] == [
        "artifact"
    ]
    assert [hit.source_id for hit in backend.search_keyword("keyword", 10)] == [
        "artifact"
    ]


def test_keyword_search_handles_case_sanitization_and_keyword_metadata(tmp_path) -> None:
    backend = LocalRecordBackend(tmp_path / "records.db")
    record = _record(
        "note",
        "tagged",
        "Alpha body",
        metadata={"tags": ["Important"], "keywords": ["needle"]},
    )
    backend.index([record])

    assert backend.search_keyword("ALPHA", 10)
    assert [hit.source_id for hit in backend.search_keyword("NEEDLE", 10)] == [
        "tagged"
    ]


def test_keyword_search_retrieves_artifact_metadata_without_uri(tmp_path) -> None:
    backend = LocalRecordBackend(tmp_path / "records.db")
    backend.index(
        [
            _record(
                "git_commit",
                "commit",
                "unrelated commit body",
                metadata={
                    "file_tokens": ["src/searchkernel/search.py"],
                    "symbols": ["RecordSearchPipeline"],
                    "commit_file_tokens": ["search.py"],
                },
            ),
            _record("git_commit", "other", "unrelated record"),
        ]
    )

    assert [
        hit.source_id
        for hit in backend.search_keyword("src/searchkernel/search.py", 10)
    ] == ["commit"]
    assert [hit.source_id for hit in backend.search_keyword(
        "RecordSearchPipeline", 10
    )] == ["commit"]


def test_keyword_search_handles_artifact_variants_and_missing_tokens(tmp_path) -> None:
    backend = LocalRecordBackend(tmp_path / "records.db")
    backend.index(
        [
            _record(
                "note",
                "target",
                "unrelated body",
                metadata={
                    "header_path": "Search > RecordSearchPipeline",
                    "file_path": "src/searchkernel/search/record_pipeline.py",
                    "exact_tokens": ["RecordSearchPipeline"],
                },
            ),
            _record("git_commit", "unrelated", "unrelated body"),
        ]
    )

    assert [
        hit.source_id
        for hit in backend.search_keyword(
            r"src\searchkernel\search\record_pipeline.py,", 10
        )
    ] == ["target"]
    assert [
        hit.source_id
        for hit in backend.search_keyword("RecordSearchPipeline!", 10)
    ] == ["target"]
    assert backend.search_keyword("src/searchkernel/missing.py", 10) == []


def test_keyword_search_ranks_embedded_artifact_source_above_mentions(tmp_path) -> None:
    backend = LocalRecordBackend(tmp_path / "records.db")
    backend.index(
        [
            _record(
                "note",
                "exact",
                "unrelated body",
                title="document_tools.py",
                uri="mcp_markdown_ragdocs/mcp/tools/document_tools.py",
            ),
            _record(
                "note",
                "mention",
                "References handle_query_documents.",
                title="Other",
                uri="docs/other.py",
            ),
        ]
    )

    for query in (
        "mcp_markdown_ragdocs/mcp/tools/document_tools.py handle_query_documents",
        "document_tools.py handle_query_documents",
    ):
        assert [
            hit.source_id for hit in backend.search_keyword(query, 10)
        ] == ["exact"]


def test_keyword_search_prefers_exact_heading_across_workspaces(tmp_path) -> None:
    backend = LocalRecordBackend(tmp_path / "records.db")
    backend.index(
        [
            _record(
                "note",
                "exact",
                "Exact section content",
                workspace_id="project-a",
                title="Guide",
                metadata={"header_path": "6.1. Overview"},
            ),
            _record(
                "note",
                "near",
                "Near section content",
                workspace_id="project-b",
                title="6.1. Overview",
                metadata={"header_path": "6.1. Overview (Legacy)"},
            ),
        ]
    )

    assert [
        hit.source_id
        for hit in backend.search_keyword("6.1. Overview", 10)
    ] == ["exact", "near"]


def test_keyword_search_applies_all_sql_filters(tmp_path) -> None:
    backend = LocalRecordBackend(tmp_path / "records.db")
    records = [
        _record("note", "one", "common", workspace_id="one"),
        _record("commit", "two", "common", workspace_id="one"),
        _record(
            "note",
            "three",
            "common",
            workspace_id="two",
            status=RecordStatus.STALE,
        ),
        _record(
            "note",
            "four",
            "common",
            workspace_id="one",
            status=RecordStatus.ARCHIVED,
        ),
    ]
    backend.index(records)

    assert {
        hit.source_id for hit in backend.search_keyword(
            "common", 10, {"workspace_id": "one"}
        )
    } == {"one", "two"}
    assert [hit.source_id for hit in backend.search_keyword(
        "common", 10, {"statuses": ["stale"], "workspace_id": "two"}
    )] == ["three"]
    assert [hit.source_id for hit in backend.search_keyword(
        "common", 10, {"status": RecordStatus.ARCHIVED, "include_inactive": False}
    )] == ["four"]
    assert [hit.source_id for hit in backend.search_keyword(
        "common", 10, {"source_kinds": ["commit"], "include_inactive": True}
    )] == ["two"]
    assert [hit.source_id for hit in backend.search_keyword(
        "common", 10, {"lifecycle_status": RecordStatus.ARCHIVED}
    )] == ["four"]
    assert [hit.source_id for hit in backend.search_keyword(
        "common",
        10,
        {"candidate_storage_keys": [records[2].storage_key], "include_inactive": True},
    )] == ["three"]


def test_keyword_search_applies_metadata_and_exclusion_filters(tmp_path) -> None:
    backend = LocalRecordBackend(tmp_path / "records.db")
    kept = _record(
        "note",
        "kept",
        "common kept",
        workspace_id="workspace",
        uri="projects/keep/guide.md",
        metadata={"project_id": "keep", "doc_id": "doc-kept", "kind": "guide"},
    )
    wrong_project = _record(
        "note",
        "wrong-project",
        "common wrong project",
        workspace_id="workspace",
        uri="projects/drop/guide.md",
        metadata={"project_id": "drop", "doc_id": "doc-drop", "kind": "guide"},
    )
    excluded = _record(
        "note",
        "excluded",
        "common excluded",
        workspace_id="workspace",
        uri="projects/keep/ignored.md",
        metadata={"project_id": "keep", "doc_id": "doc-excluded", "kind": "guide"},
    )
    backend.index([kept, wrong_project, excluded])

    filters = {
        "source_filter": ["note"],
        "project_filter": ["keep"],
        "paths": ["projects/keep/guide.md"],
        "document_id": "doc-kept",
        "metadata_equals": {"kind": "guide"},
        "excluded_files": ["ignored.md"],
        "excluded_projects": ["drop"],
    }

    assert [hit.storage_key for hit in backend.search_keyword("common", 10, filters)] == [
        kept.storage_key
    ]


def test_keyword_project_filters_match_numeric_metadata_ids(tmp_path) -> None:
    backend = LocalRecordBackend(tmp_path / "records.db")
    kept = _record(
        "note",
        "kept",
        "common",
        metadata={"project_id": 123},
    )
    excluded = _record(
        "note",
        "excluded",
        "common",
        metadata={"project_id": 456},
    )
    backend.index([kept, excluded])

    assert [
        hit.source_id
        for hit in backend.search_keyword(
            "common",
            10,
            {"project_filter": ["123"], "excluded_projects": ["456"]},
        )
    ] == ["kept"]


def test_keyword_search_applies_document_path_and_metadata_filters(tmp_path) -> None:
    backend = LocalRecordBackend(tmp_path / "records.db")
    kept = _record(
        "note",
        "kept",
        "common",
        metadata={
            "file_path": "src/guide.md",
            "doc_id": "guide",
            "kind": "reference",
        },
    )
    wrong_document = _record(
        "note",
        "wrong-document",
        "common",
        metadata={
            "file_path": "src/guide.md",
            "doc_id": "other",
            "kind": "reference",
        },
    )
    excluded_path = _record(
        "note",
        "excluded-path",
        "common",
        metadata={
            "file_path": "src/ignored.md",
            "doc_id": "guide",
            "kind": "reference",
        },
    )
    backend.index([kept, wrong_document, excluded_path])

    filters = {
        "paths": ["guide.md"],
        "document_ids": ["guide"],
        "metadata_equals": {"kind": "reference"},
        "excluded_files": ["ignored.md"],
    }

    assert [hit.source_id for hit in backend.search_keyword("common", 10, filters)] == [
        "kept"
    ]


def test_keyword_path_filters_escape_like_wildcards(tmp_path) -> None:
    backend = LocalRecordBackend(tmp_path / "records.db")
    exact = _record(
        "exact",
        "common",
        "common",
        metadata={"file_path": "src/my_file.md"},
    )
    near = _record(
        "near",
        "common-near",
        "common",
        metadata={"file_path": "src/myXfile.md"},
    )
    backend.index([exact, near])

    assert [hit.source_id for hit in backend.search_keyword(
        "common", 10, {"paths": ["my_file.md"]}
    )] == ["common"]


def test_keyword_search_rejects_invalid_metadata_filter_fields(tmp_path) -> None:
    backend = LocalRecordBackend(tmp_path / "records.db")
    backend.index([_record("note", "record", "common")])

    with pytest.raises(ValueError, match="metadata_equals field"):
        backend.search_keyword("common", 10, {"metadata_equals": {"kind-name": "x"}})


def test_keyword_empty_project_scopes_match_nothing(tmp_path) -> None:
    backend = LocalRecordBackend(tmp_path / "records.db")
    backend.index(
        [
            _record("note", "one", "common", metadata={"project_id": "one"}),
            _record("note", "two", "common", metadata={"project_id": "two"}),
        ]
    )

    assert backend.search_keyword("common", 10, {"project_ids": []}) == []
    assert backend.search_keyword("common", 10, {"project_id": []}) == []


def test_source_scoped_filters_apply_before_local_keyword_and_vector_limits(
    tmp_path,
) -> None:
    """Unauthorized high-score local records cannot consume the result limit."""
    backend = LocalRecordBackend(tmp_path / "records.db")
    records = [
        _record(
            "note",
            f"blocked-{index}",
            "common",
            metadata={"acl": ["blocked"]},
        )
        for index in range(3)
    ]
    allowed = _record("note", "allowed", "common", metadata={"acl": ["allowed"]})
    records.append(allowed)
    for record in records:
        record.embedding = [1.0, 0.0]
    backend.index(records)
    backend.upsert(records, "model", 2)
    filters = {
        "source_scoped_filters": {
            "note": {"metadata_contains_any": {"acl": ["allowed"]}}
        }
    }

    assert [hit.source_id for hit in backend.search_keyword("common", 1, filters)] == [
        "allowed"
    ]
    assert [
        hit.source_id
        for hit in backend.search_vector(
            [1.0, 0.0], 1, model_name="model", dim=2, filters=filters
        )
    ] == ["allowed"]


@pytest.mark.asyncio
async def test_scoped_keyword_candidates_match_public_pipeline_boundary(tmp_path) -> None:
    backend = LocalRecordBackend(tmp_path / "records.db")
    kept = _record(
        "memory",
        "backlog-kept",
        "backlog",
        workspace_id="workspace",
        metadata={"project_id": "keep"},
    )
    excluded = [
        _record(
            "memory",
            f"backlog-drop-{index}",
            "backlog",
            workspace_id="workspace",
            metadata={"project_id": "drop"},
        )
        for index in range(4)
    ]
    backend.index([kept, *excluded])
    keyword = LocalKeywordStore(backend)
    filters = {"workspace_id": "workspace", "project_filter": ["keep"]}

    fts_candidates = keyword.search("backlog", 1, filters)
    outcome = await SearchOrchestrator(
        hydrator=backend,
        keyword_store=keyword,
    ).search("backlog", limit=1, filters=filters)

    assert [hit.storage_key for hit in fts_candidates] == [kept.storage_key]
    assert [result.storage_key for result in outcome.results] == [
        kept.storage_key
    ]


def test_keyword_search_rejects_bare_candidate_ids(tmp_path) -> None:
    backend = LocalRecordBackend(tmp_path / "records.db")
    record = _record("note", "same", "common")
    backend.index([record])

    assert backend.search_keyword("common", 10, {"candidate_ids": ["same"]}) == []


def test_keyword_update_delete_and_rebuild_consistency(tmp_path) -> None:
    backend = LocalRecordBackend(tmp_path / "records.db")
    record = _record("note", "one", "before")
    backend.index([record])
    assert backend.check_keyword_index()
    assert backend.search_keyword("before", 10)

    record.body = "after"
    backend.index([record])
    assert not backend.search_keyword("before", 10)
    assert backend.search_keyword("after", 10)

    conn = backend.db_manager.get_connection()
    conn.execute(
        "DELETE FROM local_records_fts WHERE rowid = "
        "(SELECT rowid FROM local_records WHERE storage_key = ?)",
        (record.storage_key,),
    )
    conn.commit()
    assert not backend.check_keyword_index()
    backend.rebuild_keyword_index()
    assert backend.check_keyword_index()

    backend.delete([record.storage_key])
    assert not backend.search_keyword("after", 10)
    assert backend.check_keyword_index()


def test_batch_record_updates_and_deletes_keep_fts_visible(tmp_path) -> None:
    backend = LocalRecordBackend(tmp_path / "records.db")
    first = _record("note", "first", "old first", metadata={"tags": ["old-tag"]})
    second = _record("note", "second", "old second")
    third = _record("note", "third", "keep third")
    backend.index([first, second, third])

    first.body = "new first"
    first.metadata = {"tags": ["new-tag"]}
    second.body = "new second"
    backend.index([first, second])

    assert backend.search_keyword("old", 10) == []
    assert {hit.source_id for hit in backend.search_keyword("new", 10)} == {
        "first",
        "second",
    }
    assert [hit.source_id for hit in backend.search_keyword("new-tag", 10)] == [
        "first"
    ]
    assert backend.check_keyword_index()

    backend.delete([first.storage_key, third.storage_key, "missing"])

    assert [hit.source_id for hit in backend.search_keyword("new", 10)] == ["second"]
    assert backend.search_keyword("keep", 10) == []
    assert backend.check_keyword_index()


def test_batch_delete_counts_all_vectors_and_invalidates_vector_epoch(tmp_path) -> None:
    backend, _keyword, vector, _graph = _backend(tmp_path)
    records = [
        _record("note", "first", "first"),
        _record("note", "second", "second"),
    ]
    for record in records:
        record.embedding = [1.0, 0.0]
    vector.upsert(records, "test", 2)
    before = backend.epochs()

    backend.delete([record.storage_key for record in records])

    conn = backend.db_manager.get_connection()
    assert conn.execute("SELECT COUNT(*) FROM local_vectors_v2").fetchone()[0] == 0
    after = backend.epochs()
    assert after["vector"] == before["vector"] + 1


def test_keyword_ties_are_ordered_by_storage_key() -> None:
    backend = LocalRecordBackend()
    first = _record("note", "a", "same")
    second = _record("note", "b", "same")
    backend.index([second, first])

    hits = backend.search_keyword("same", 10)
    assert [hit.storage_key for hit in hits] == sorted(
        (first.storage_key, second.storage_key)
    )


def test_keyword_migrates_existing_local_records_schema(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE local_records (
            storage_key TEXT PRIMARY KEY,
            workspace_id TEXT,
            source_kind TEXT NOT NULL,
            source_id TEXT NOT NULL,
            title TEXT NOT NULL,
            body TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            metadata TEXT NOT NULL,
            uri TEXT,
            status TEXT NOT NULL
        )
        """
    )
    timestamp = datetime(2026, 1, 1, tzinfo=UTC).isoformat()
    conn.execute(
        "INSERT INTO local_records VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            'record:["workspace","note","legacy"]',
            "workspace",
            "note",
            "legacy",
            "Legacy",
            "migration body",
            timestamp,
            timestamp,
            '{"tags": ["migration"], "file_path": "docs/legacy.md"}',
            None,
            "active",
        ),
    )
    conn.commit()
    conn.close()

    backend = LocalRecordBackend(db_path)
    assert [hit.source_id for hit in backend.search_keyword("migration", 10)] == [
        "legacy"
    ]
    assert [hit.source_id for hit in backend.search_keyword("legacy.md", 10)] == [
        "legacy"
    ]
    assert backend.check_keyword_index()


def test_keyword_skips_metadata_backfill_for_current_schema(
    tmp_path: Path,
    monkeypatch,
) -> None:
    db_path = tmp_path / "current.db"
    LocalRecordBackend(db_path)

    def fail_backfill(self, conn) -> None:
        raise AssertionError("current schemas must skip metadata backfill")

    monkeypatch.setattr(
        local_indices._KeywordEngine,
        "_migrate_keyword_columns",
        fail_backfill,
    )

    LocalRecordBackend(db_path)


def test_local_schema_initializes_current_storage_tables(tmp_path: Path) -> None:
    backend = LocalRecordBackend(tmp_path / "records.db")
    tables = {
        row[0]
        for row in backend.db_manager.get_connection().execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }

    assert {
        "local_records",
        "local_vectors_v2",
        "local_graph_edges",
        "system_state",
    } <= tables
    assert "local_vectors" not in tables
    assert "local_vector_schema" not in tables


def test_scalar_and_batched_keyword_ingestion_are_equivalent() -> None:
    """Scalar and batched keyword writes produce identical search hits."""
    records = [
        _record("note", "one", "alpha"),
        _record("note", "two", "beta alpha"),
        _record("note", "three", "gamma"),
    ]
    scalar = LocalRecordBackend()
    batched = LocalRecordBackend()
    try:
        for record in records:
            scalar.index([record])
        batched.index(records)

        scalar_hits = scalar.search_keyword("alpha", 10)
        batch_hits = batched.search_keyword("alpha", 10)
        assert [(hit.storage_key, hit.score) for hit in scalar_hits] == [
            (hit.storage_key, hit.score) for hit in batch_hits
        ]
    finally:
        scalar.close()
        batched.close()


def test_keyword_schema_rebuild_preserves_rows_beyond_one_init_batch(
    tmp_path: Path,
) -> None:
    """Rebuilding lexical state keeps records from every initialization window."""
    db_path = tmp_path / "records.db"
    backend = LocalRecordBackend(db_path)
    backend.index(
        [
            _record("note", f"record-{index}", f"token-{index}")
            for index in range(1_001)
        ]
    )
    connection = backend.db_manager.get_connection()
    connection.execute("DELETE FROM local_keyword_schema")
    connection.commit()
    backend.close()

    rebuilt = LocalRecordBackend(db_path)
    try:
        assert [hit.source_id for hit in rebuilt.search_keyword("token-1000", 1)] == [
            "record-1000"
        ]
        assert rebuilt.check_keyword_index()
    finally:
        rebuilt.close()


def test_filtered_keyword_search_bounds_post_filter_candidates(tmp_path: Path) -> None:
    """
    Keep post-filtered FTS searches from materializing every matching row.
    """
    backend = LocalRecordBackend(tmp_path / "records.db")
    records = [
        _record(
            "note",
            f"record-{index}",
            "common token",
            metadata={"kind": "note"},
        )
        for index in range(50)
    ]
    backend.index(records)
    statements: list[str] = []
    connection = backend.db_manager.get_connection()
    connection.set_trace_callback(statements.append)
    try:
        hits = backend.search_keyword(
            "common",
            1,
            {"statuses": [RecordStatus.ACTIVE], "metadata_equals": {"kind": "note"}},
        )
    finally:
        connection.set_trace_callback(None)

    assert len(hits) == 1
    assert any("LIMIT 4" in statement for statement in statements)
    assert all("LIMIT -1" not in statement for statement in statements)


@pytest.mark.slow
def test_marked_keyword_scale_smoke() -> None:
    backend = LocalRecordBackend()
    try:
        records = [
            _record("note", f"record-{index}", f"common token {index}")
            for index in range(1_000)
        ]
        backend.index(records)

        hits = backend.search_keyword("common token", 10)
        assert len(hits) == 10
        assert backend.check_keyword_index()
    finally:
        backend.close()


@pytest.mark.asyncio
async def test_local_orchestrator_matches_record_pipeline_contract_deterministically(tmp_path) -> None:
    backend, keyword, vector, _graph = _backend(tmp_path)
    first = _record("note", "one", "first alpha", workspace_id="workspace")
    second = _record("note", "two", "first beta", workspace_id="workspace")
    first.embedding = [1.0, 0.0]
    second.embedding = [0.9, 0.1]
    vector.upsert([first, second], "test", 2)
    keyword.index([first, second])

    orchestrator = SearchOrchestrator(
        hydrator=backend,
        keyword_store=keyword,
        vector_store=vector,
        embedding_provider=_Embedder(),
    )
    outcome = await orchestrator.search(
        "first", limit=2, filters={"workspace_id": "workspace"}
    )
    results = outcome.results

    assert [result.storage_key for result in results] == sorted(
        (first.storage_key, second.storage_key)
    )
    assert all(result.record.body.startswith("first") for result in results)


def test_graph_edges_cascade_when_a_record_is_deleted(tmp_path) -> None:
    backend, _keyword, _vector, graph = _backend(tmp_path)
    source = _record("note", "source", "source")
    target = _record("note", "target", "target")
    backend.index([source, target])
    graph.upsert_edges([(source.storage_key, target.storage_key, "related", 0.5)])
    epoch = graph.graph_epoch()

    backend.delete([target.storage_key])

    assert graph.graph_epoch() == epoch + 1
    assert graph.neighbors(source.storage_key) == []
    assert graph.check_graph_integrity()


def test_graph_rejects_malformed_or_missing_record_endpoints(tmp_path) -> None:
    backend, _keyword, _vector, graph = _backend(tmp_path)
    source = _record("note", "source", "source")
    target = _record("note", "target", "target")
    backend.index([source, target])
    epoch = graph.graph_epoch()

    with pytest.raises(ValueError, match="canonical storage key"):
        graph.upsert_edges([("source", target.storage_key, "related", 0.5)])
    with pytest.raises(ValueError, match="not indexed"):
        graph.upsert_edges(
            [
                (
                    source.storage_key,
                    _record("note", "missing", "missing").storage_key,
                    "related",
                    0.5,
                )
            ]
        )

    assert graph.graph_epoch() == epoch


def test_graph_schema_indexes_both_endpoints(tmp_path) -> None:
    backend, _keyword, _vector, graph = _backend(tmp_path)
    source = _record("note", "source", "source")
    target = _record("note", "target", "target")
    backend.index([source, target])

    conn = backend.db_manager.get_connection()
    foreign_keys = {
        row[3] for row in conn.execute("PRAGMA foreign_key_list(local_graph_edges)")
    }
    index_names = {
        row[1] for row in conn.execute("PRAGMA index_list(local_graph_edges)")
    }

    assert foreign_keys == {"source_id", "target_id"}
    assert {
        "idx_local_graph_source",
        "idx_local_graph_target",
        "idx_local_graph_source_type",
    } <= index_names
    assert graph.check_graph_integrity()


def test_graph_integrity_hides_dangling_rows_from_neighbors(tmp_path) -> None:
    backend, _keyword, _vector, graph = _backend(tmp_path)
    source = _record("note", "source", "source")
    backend.index([source])
    conn = backend.db_manager.get_connection()
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute(
        """
        INSERT INTO local_graph_edges (source_id, target_id, edge_type, weight)
        VALUES (?, ?, ?, ?)
        """,
        (source.storage_key, "record:[null,\"note\",\"missing\"]", "related", 0.5),
    )
    conn.commit()

    assert not graph.check_graph_integrity()
    assert graph.neighbors(source.storage_key) == []


def test_failed_batch_rolls_back_records_and_epochs(tmp_path) -> None:
    backend, _keyword, vector, graph = _backend(tmp_path)
    good = _record("note", "good", "good")
    bad = _record("note", "bad", "bad", metadata={"invalid": object()})
    good.embedding = [1.0, 0.0]
    bad.embedding = [0.0, 1.0]
    before = backend.epochs()

    with pytest.raises(TypeError):
        vector.upsert([good, bad], "test", 2)

    assert backend.hydrate_record(good.storage_key) is None
    assert backend.epochs() == before
    assert graph.graph_epoch() == before["graph"]


def test_vector_upsert_rejects_missing_embedding(tmp_path) -> None:
    """A vector upsert reports which record lacks its required embedding."""
    _backend_instance, _keyword, vector, _graph = _backend(tmp_path)
    record = _record("note", "missing", "missing")

    with pytest.raises(ValueError, match="missing embedding"):
        vector.upsert([record], "test", 2)


def test_vector_upsert_rejects_mixed_batch_before_mutation(tmp_path) -> None:
    """A missing embedding prevents every record in the batch from being stored."""
    backend, _keyword, vector, _graph = _backend(tmp_path)
    good = _record("note", "good-vector", "good")
    missing = _record("note", "missing-vector", "missing")
    good.embedding = [1.0, 0.0]
    before = backend.epochs()

    with pytest.raises(ValueError, match="missing-vector"):
        vector.upsert([good, missing], "test", 2)

    assert backend.hydrate_record(good.storage_key) is None
    assert backend.hydrate_record(missing.storage_key) is None
    assert backend.epochs() == before


def test_repeated_graph_upsert_does_not_advance_epoch(tmp_path) -> None:
    backend, _keyword, _vector, graph = _backend(tmp_path)
    source = _record("note", "source", "source")
    target = _record("note", "target", "target")
    backend.index([source, target])
    edge = (source.storage_key, target.storage_key, "related", 0.5)

    graph.upsert_edges([edge])
    epoch = graph.graph_epoch()
    graph.upsert_edges([edge])

    assert graph.graph_epoch() == epoch


def _set_identity_marker(db_path: Path, value: str | None) -> None:
    connection = sqlite3.connect(db_path)
    if value is None:
        connection.execute(
            "DELETE FROM system_state WHERE key = ?",
            (local_indices._RECORD_IDENTITY_VERSION_KEY,),
        )
    else:
        connection.execute(
            "UPDATE system_state SET value = ? WHERE key = ?",
            (value, local_indices._RECORD_IDENTITY_VERSION_KEY),
        )
    connection.commit()
    connection.close()


def test_new_store_is_not_stale(tmp_path: Path) -> None:
    with LocalRecordBackend(tmp_path / "records.db") as backend:
        assert backend.record_identity_stale is False


def test_store_written_without_a_marker_is_stale(tmp_path: Path) -> None:
    """A populated store predating the marker cannot be assumed current."""
    db_path = tmp_path / "records.db"
    with LocalRecordBackend(db_path) as backend:
        backend.index([_record("notes", "doc", "body")])
    _set_identity_marker(db_path, None)

    with LocalRecordBackend(db_path) as backend:
        assert backend.record_identity_stale is True


@pytest.mark.parametrize("marker", ("1", "banana", ""))
def test_store_written_under_another_identity_scheme_is_stale(
    tmp_path: Path, marker: str
) -> None:
    """An older or unreadable marker cannot prove the store is current."""
    db_path = tmp_path / "records.db"
    with LocalRecordBackend(db_path):
        pass
    _set_identity_marker(db_path, marker)

    with LocalRecordBackend(db_path) as backend:
        assert backend.record_identity_stale is True


def test_a_stale_store_still_serves_searches(tmp_path: Path) -> None:
    """Reporting staleness must not cost the caller access to their data."""
    db_path = tmp_path / "records.db"
    with LocalRecordBackend(db_path) as backend:
        backend.index([_record("notes", "doc", "findable body")])
    _set_identity_marker(db_path, "1")

    with LocalRecordBackend(db_path) as backend:
        assert backend.record_identity_stale is True
        assert backend.search_keyword("findable", 5)


def _cosine_ranked_records() -> list[Record]:
    """Five records with fixed embeddings giving a known cosine order vs [1, 0]."""
    specs = [
        ("a", [1.0, 0.0]),
        ("b", [0.9, 0.1]),
        ("c", [0.5, 0.5]),
        ("d", [0.0, 1.0]),
        ("e", [-1.0, 0.0]),
    ]
    records = []
    for source_id, embedding in specs:
        record = _record("note", source_id, source_id)
        record.embedding = embedding
        records.append(record)
    return records


def test_block_vector_search_agrees_with_snapshot_vector_search(tmp_path) -> None:
    """The disk-streaming block path must rank and score identically to the
    in-memory snapshot path for the same corpus and query."""
    records = _cosine_ranked_records()

    snapshot_backend = LocalRecordBackend(tmp_path / "snapshot.db")
    snapshot_backend.upsert(records, "test", 2)

    block_backend = LocalRecordBackend(
        tmp_path / "block.db",
        vector_snapshot_max_rows=2,
        vector_snapshot_max_bytes=1_000_000,
    )
    block_backend.upsert(records, "test", 2)

    snapshot_hits = snapshot_backend.search_vector(
        [1.0, 0.0], 5, model_name="test", dim=2
    )
    block_hits = block_backend.search_vector(
        [1.0, 0.0], 5, model_name="test", dim=2
    )

    assert [(hit.storage_key, hit.score) for hit in snapshot_hits] == [
        (hit.storage_key, hit.score) for hit in block_hits
    ]
    assert [hit.source_id for hit in block_hits] == ["a", "b", "c", "d", "e"]


def test_block_vector_search_paginates_across_multiple_batches(tmp_path) -> None:
    """Force the block loop to run more than one iteration and still rank
    correctly across the batch boundary."""
    records = _cosine_ranked_records()
    max_rows = 2
    backend = LocalRecordBackend(
        tmp_path / "records.db",
        vector_snapshot_max_rows=max_rows,
        vector_snapshot_max_bytes=1_000_000,
    )
    backend.upsert(records, "test", 2)
    # batch_limit == vector_snapshot_max_rows here (bytes cap is not binding),
    # so 5 records over a limit-2 page size cross at least 2 batch boundaries.
    batch_limit = max_rows
    assert len(records) > 2 * batch_limit

    hits = backend.search_vector([1.0, 0.0], 5, model_name="test", dim=2)

    assert [hit.source_id for hit in hits] == ["a", "b", "c", "d", "e"]


def test_block_vector_search_trims_to_requested_top_k(tmp_path) -> None:
    records = _cosine_ranked_records()
    backend = LocalRecordBackend(
        tmp_path / "records.db",
        vector_snapshot_max_rows=2,
        vector_snapshot_max_bytes=1_000_000,
    )
    backend.upsert(records, "test", 2)

    hits = backend.search_vector([1.0, 0.0], 2, model_name="test", dim=2)

    assert [hit.source_id for hit in hits] == ["a", "b"]


def test_block_vector_search_applies_filters_per_batch(tmp_path) -> None:
    """The block path re-applies filters batch by batch, unlike the snapshot
    path's single mask over the whole matrix."""
    records = _cosine_ranked_records()
    for record in records:
        record.metadata = {"keep": record.source_id in {"a", "c", "e"}}
    backend = LocalRecordBackend(
        tmp_path / "records.db",
        vector_snapshot_max_rows=2,
        vector_snapshot_max_bytes=1_000_000,
    )
    backend.upsert(records, "test", 2)

    hits = backend.search_vector(
        [1.0, 0.0],
        5,
        model_name="test",
        dim=2,
        filters={"metadata_equals": {"keep": True}},
    )

    assert [hit.source_id for hit in hits] == ["a", "c", "e"]


def test_block_vector_search_returns_all_available_when_k_exceeds_corpus(
    tmp_path,
) -> None:
    records = _cosine_ranked_records()
    backend = LocalRecordBackend(
        tmp_path / "records.db",
        vector_snapshot_max_rows=2,
        vector_snapshot_max_bytes=1_000_000,
    )
    backend.upsert(records, "test", 2)

    hits = backend.search_vector([1.0, 0.0], 100, model_name="test", dim=2)

    assert [hit.source_id for hit in hits] == ["a", "b", "c", "d", "e"]


def test_marking_the_identity_current_survives_reopening(tmp_path: Path) -> None:
    db_path = tmp_path / "records.db"
    with LocalRecordBackend(db_path) as backend:
        backend.index([_record("notes", "doc", "body")])
    _set_identity_marker(db_path, "1")

    with LocalRecordBackend(db_path) as backend:
        backend.mark_record_identity_current()
        assert backend.record_identity_stale is False

    with LocalRecordBackend(db_path) as backend:
        assert backend.record_identity_stale is False


def test_rebuilding_the_keyword_index_invalidates_cached_readers(
    tmp_path: Path,
) -> None:
    """Caches key on lane epochs, so a repair that leaves the epoch alone
    would let readers keep serving results from the index it replaced."""
    with LocalRecordBackend(tmp_path / "records.db") as backend:
        backend.index([_record("notes", "doc", "findable body")])
        before = backend.keyword_epoch()

        backend.rebuild_keyword_index()

        assert backend.keyword_epoch() > before


def test_rebuilding_a_desynced_keyword_index_invalidates_cached_readers(
    tmp_path: Path,
) -> None:
    """A rebuild is triggered by detected corruption, so the invalidation
    has to hold on that path too, not only on a healthy rebuild."""
    with LocalRecordBackend(tmp_path / "records.db") as backend:
        record = _record("notes", "doc", "findable body")
        backend.index([record])
        connection = backend.db_manager.get_connection()
        connection.execute(
            "DELETE FROM local_records_fts WHERE rowid = "
            "(SELECT rowid FROM local_records WHERE storage_key = ?)",
            (record.storage_key,),
        )
        connection.commit()
        assert not backend.check_keyword_index()
        before = backend.keyword_epoch()

        backend.rebuild_keyword_index()

        assert backend.check_keyword_index()
        assert backend.keyword_epoch() > before
        assert backend.search_keyword("findable", 5)
