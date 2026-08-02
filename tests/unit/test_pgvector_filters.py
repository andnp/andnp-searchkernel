from searchkernel.adapters.stores.pgvector import build_pgvector_filter_sql
from searchkernel.domain import RecordIdentity, RecordStatus


def test_pgvector_filter_sql_covers_identity_lifecycle_metadata_filters() -> None:
    identity = RecordIdentity("workspace", "note", "kept")
    clauses, parameters = build_pgvector_filter_sql(
        {
            "workspace_id": "workspace",
            "source_kinds": ["note"],
            "statuses": [RecordStatus.ACTIVE],
            "candidate_ids": [identity],
            "project_id": "project",
            "excluded_files": {"ignored.md"},
            "excluded_documents": {"ignored"},
        }
    )

    sql_text = " AND ".join(clauses)
    assert "r.status = ANY(%s)" in sql_text
    assert "r.workspace_id = %s" in sql_text
    assert "r.source_kind = ANY(%s)" in sql_text
    assert "r.record_id = ANY(%s)" in sql_text
    assert "r.source_id = ANY(%s)" not in sql_text
    assert "metadata->>'project_id'" in sql_text
    assert "metadata->>'file_path'" in sql_text
    assert "metadata->>'doc_id'" in sql_text
    assert parameters[0] == ["active"]
    assert identity.storage_key in parameters[3]
    assert parameters[3] == [identity.storage_key]


def test_pgvector_candidate_filter_ignores_bare_source_ids() -> None:
    clauses, parameters = build_pgvector_filter_sql({"candidate_ids": ["same"]})

    assert clauses == ["FALSE"]
    assert parameters == []


def test_pgvector_filter_sql_rejects_empty_candidate_filter() -> None:
    clauses, parameters = build_pgvector_filter_sql({"candidate_ids": []})

    assert clauses == ["FALSE"]
    assert parameters == []


def test_pgvector_filter_sql_supports_single_metadata_field() -> None:
    clauses, parameters = build_pgvector_filter_sql(
        {"metadata_equals": {"issue_type": "Bug"}}
    )

    sql_text = " AND ".join(clauses)
    assert "metadata->>'issue_type' = %s" in sql_text
    assert "Bug" in parameters


def test_pgvector_filter_sql_supports_multiple_metadata_fields() -> None:
    clauses, parameters = build_pgvector_filter_sql(
        {
            "metadata_equals": {
                "issue_type": "Bug",
                "component": "core",
                "team": "backend",
            }
        }
    )

    sql_text = " AND ".join(clauses)
    assert "metadata->>'issue_type' = %s" in sql_text
    assert "metadata->>'component' = %s" in sql_text
    assert "metadata->>'team' = %s" in sql_text
    assert "Bug" in parameters
    assert "core" in parameters
    assert "backend" in parameters


def test_pgvector_filter_sql_metadata_equals_with_project_id() -> None:
    clauses, parameters = build_pgvector_filter_sql(
        {
            "project_id": "proj-123",
            "metadata_equals": {"issue_type": "Feature", "component": "api"},
        }
    )

    sql_text = " AND ".join(clauses)
    assert "metadata->>'project_id' = ANY(%s)" in sql_text
    assert "metadata->>'issue_type' = %s" in sql_text
    assert "metadata->>'component' = %s" in sql_text
    # project_id uses ANY so it's wrapped in a list
    assert ["proj-123"] in parameters
    assert "Feature" in parameters
    assert "api" in parameters


def test_pgvector_filter_sql_metadata_equals_ignores_none_values() -> None:
    clauses, parameters = build_pgvector_filter_sql(
        {"metadata_equals": {"issue_type": "Bug", "component": None}}
    )

    sql_text = " AND ".join(clauses)
    assert "metadata->>'issue_type' = %s" in sql_text
    assert "metadata->>'component'" not in sql_text
    assert "Bug" in parameters


def test_pgvector_keyword_filter_sql_supports_keyword_filter_aliases() -> None:
    clauses, parameters = build_pgvector_filter_sql(
        {
            "source_filter": ["note"],
            "project_filter": ["keep"],
            "paths": ["docs/guide.md"],
            "document_id": "doc-1",
            "excluded_projects": ["drop"],
            "excluded_paths": ["ignored.md"],
            "excluded_documents": ["blocked"],
            "metadata_equals": {"kind": "guide"},
        }
    )

    sql_text = " AND ".join(clauses)
    assert "r.source_kind = ANY(%s)" in sql_text
    assert "metadata->>'project_id' = ANY(%s)" in sql_text
    assert "metadata->>'file_path'" in sql_text
    assert "metadata->>'doc_id'" in sql_text
    assert "metadata->>'kind' = %s" in sql_text
    assert ["note"] in parameters
    assert ["keep"] in parameters
    assert ["doc-1"] in parameters
