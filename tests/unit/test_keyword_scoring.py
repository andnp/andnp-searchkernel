from searchkernel.indices.keyword_scoring import (
    looks_like_artifact_query,
    sanitize_fts_query,
    score_artifact_match,
    score_field_aware_match,
)


def test_sanitize_fts_query_removes_match_operators():
    assert sanitize_fts_query('worker.enabled - "debug"') == "worker.enabled debug"
    assert sanitize_fts_query("   ") == '""'


def test_looks_like_artifact_query_detects_path_like_values():
    assert looks_like_artifact_query("bootstrap.checkpoint.json")
    assert looks_like_artifact_query("docs/runtime_state")
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
