import sqlite3
import string

from searchkernel.indices.keyword_scoring import (
    looks_like_artifact_query,
    normalize_artifact_value,
    normalize_field_text,
    sanitize_fts_query,
    score_artifact_match,
    score_field_aware_match,
    score_header_locality,
    score_title_locality,
    split_header_segments,
)


def test_sanitize_fts_query_removes_match_operators():
    assert sanitize_fts_query('worker.enabled - "debug"') == "worker enabled debug"
    assert sanitize_fts_query('"alpha beta"') == '"alpha beta"'
    assert sanitize_fts_query("alph*") == "alph*"
    assert sanitize_fts_query("alpha OR beta") == 'alpha "OR" beta'
    assert sanitize_fts_query("C++") == "C"
    assert sanitize_fts_query("how do I authenticate API requests?") == (
        "how do I authenticate API requests"
    )
    assert sanitize_fts_query(
        "The MCP server has a method called list_tools that returns Tool objects."
    ) == "The MCP server has a method called list_tools that returns Tool objects"
    assert sanitize_fts_query("   ") == '""'


def test_sanitize_fts_query_accepts_arbitrary_punctuation_for_fts5():
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE VIRTUAL TABLE search USING fts5(body)")
    connection.execute("INSERT INTO search(body) VALUES (?)", ("alpha beta",))

    for punctuation in string.punctuation:
        sanitized = sanitize_fts_query(f"alpha{punctuation}beta")
        connection.execute(
            "SELECT rowid FROM search WHERE search MATCH ?", (sanitized,)
        ).fetchall()


def test_looks_like_artifact_query_detects_path_like_values():
    assert looks_like_artifact_query("bootstrap.checkpoint.json")
    assert looks_like_artifact_query("docs/runtime_state")
    assert not looks_like_artifact_query(
        "The MCP server exposes list_tools and inputSchema"
    )
    assert not looks_like_artifact_query("authentication")


def test_field_aware_score_prefers_exact_title():
    title_score = score_field_aware_match(
        "Authentication Guide",
        content="General documentation.",
        title="Authentication Guide",
        headers="Reference",
        source_file="docs/auth.md",
    )
    content_score = score_field_aware_match(
        "Authentication Guide",
        content="Authentication Guide details.",
        title="Reference",
        headers="Reference",
        source_file="docs/auth.md",
    )

    assert title_score > content_score


def test_artifact_score_prefers_matching_source_file():
    source_score = score_artifact_match(
        "bootstrap.checkpoint.json",
        "bootstrap.checkpoint.json",
        content="Runtime notes.",
        title="Checkpoint Notes",
        headers="Runtime",
        source_file="runtime/bootstrap.checkpoint.json",
    )
    content_score = score_artifact_match(
        "bootstrap.checkpoint.json",
        "bootstrap.checkpoint.json",
        content="Uses bootstrap.checkpoint.json.",
        title="Runtime Notes",
        headers="Runtime",
        source_file="notes/runtime.md",
    )

    assert source_score > content_score


def test_keyword_scoring_normalizes_values_and_splits_headers():
    assert normalize_artifact_value(r" Docs\Guide.MD ") == "docs/guide.md"
    assert normalize_field_text("  A   B\nC ") == "a b c"
    assert split_header_segments("  API  > >  Users  > Details ") == [
        "api",
        "users",
        "details",
    ]


def test_keyword_scoring_handles_locality_boundaries_and_misses():
    assert score_title_locality("guide", "guide") == 80.0
    assert score_title_locality("guide", "user guide") == 24.0
    assert score_title_locality("guide", "guidelines") == 14.0
    assert score_title_locality("guide", "reference") == 0.0

    exact = score_header_locality("users", "api > users", ["api", "users"])
    deeper = score_header_locality("users", "api > reference > users", ["api", "reference", "users"])
    assert exact > deeper
    assert score_header_locality("missing", "api > users", ["api", "users"]) == 0.0


def test_keyword_scoring_artifact_matching_covers_source_and_body_fallbacks():
    assert score_artifact_match(
        "docs/guide.md",
        "guide.md",
        "",
        "",
        "",
        "docs/guide.md",
    ) == 120.0
    assert score_artifact_match(
        "guide.md",
        "guide.md",
        "",
        "guide.md",
        "",
        "notes/readme.md",
    ) == 102.0
    assert score_artifact_match(
        "guide.md",
        "guide.md",
        "mentions guide.md",
        "",
        "",
        "notes/readme.md",
    ) == 74.0
