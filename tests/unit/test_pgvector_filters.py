import pytest

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


def test_pgvector_filter_sql_rejects_unsafe_metadata_field_name() -> None:
    import pytest

    with pytest.raises(ValueError, match="metadata_equals field"):
        build_pgvector_filter_sql(
            {"metadata_equals": {"issue_type' = 'x'; DROP TABLE records; --": "Bug"}}
        )


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


def test_pgvector_filter_sql_supports_metadata_in_single_field() -> None:
    clauses, parameters = build_pgvector_filter_sql(
        {"metadata_in": {"status_name": ["Done", "Closed"]}}
    )

    sql_text = " AND ".join(clauses)
    assert "metadata->>'status_name' = ANY(%s)" in sql_text
    assert ["Done", "Closed"] in parameters


def test_pgvector_filter_sql_metadata_in_multiple_fields_are_anded() -> None:
    clauses, parameters = build_pgvector_filter_sql(
        {
            "metadata_in": {
                "status_name": ["Done", "Closed"],
                "issue_type": ["Bug"],
            }
        }
    )

    sql_text = " AND ".join(clauses)
    assert sql_text.count("= ANY(%s)") >= 2
    assert "metadata->>'status_name' = ANY(%s)" in sql_text
    assert "metadata->>'issue_type' = ANY(%s)" in sql_text
    assert ["Done", "Closed"] in parameters
    assert ["Bug"] in parameters


def test_pgvector_filter_sql_rejects_empty_metadata_in_value_list() -> None:
    clauses, parameters = build_pgvector_filter_sql(
        {"metadata_in": {"status_name": []}}
    )

    assert clauses == ["FALSE"]
    assert parameters == []


def test_pgvector_filter_sql_rejects_unsafe_metadata_in_field_name() -> None:
    with pytest.raises(ValueError, match="metadata_in field"):
        build_pgvector_filter_sql(
            {"metadata_in": {"a'; DROP TABLE records; --": ["x"]}}
        )


def test_pgvector_filter_sql_metadata_in_parameterizes_values() -> None:
    """Values must travel as bound parameters, never interpolated into SQL."""
    dangerous = "Done'); DROP TABLE records; --"
    clauses, parameters = build_pgvector_filter_sql(
        {"metadata_in": {"status_name": [dangerous]}}
    )

    sql_text = " AND ".join(clauses)
    assert dangerous not in sql_text
    assert [dangerous] in parameters


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


def test_pgvector_source_scoped_filter_uses_parameterized_array_overlap() -> None:
    """Authorization values stay parameters while array overlap is SQL-side."""
    allowed = "allowed'); DROP TABLE records; --"
    clauses, parameters = build_pgvector_filter_sql(
        {
            "source_scoped_filters": {
                "note": {"metadata_contains_any": {"acl": [allowed]}}
            }
        }
    )

    sql_text = " AND ".join(clauses)
    assert "jsonb_array_elements" in sql_text
    assert allowed not in sql_text
    assert [allowed] in parameters


def test_pgvector_source_scoped_filter_matches_only_string_array_values() -> None:
    """PostgreSQL authorization matches the shared string-array contract."""
    clauses, _ = build_pgvector_filter_sql(
        {
            "source_scoped_filters": {
                "note": {"metadata_contains_any": {"acl": ["1"]}}
            }
        }
    )

    sql_text = " AND ".join(clauses)
    assert "jsonb_typeof(scoped_value.value) = 'string'" in sql_text
    assert "scoped_value.value #>> '{}' = ANY(%s)" in sql_text


def test_pgvector_source_scoped_filter_rejects_unsafe_field_names() -> None:
    """Authorization JSON field names are validated before SQL construction."""
    with pytest.raises(ValueError, match="metadata_contains_any field"):
        build_pgvector_filter_sql(
            {
                "source_scoped_filters": {
                    "note": {
                        "metadata_contains_any": {
                            "acl'); DROP TABLE records; --": ["allowed"]
                        }
                    }
                }
            }
        )


def test_pgvector_source_scoped_filter_combines_identity_and_non_empty_metadata() -> None:
    """PostgreSQL applies workspace identity and membership presence pre-limit."""
    clauses, parameters = build_pgvector_filter_sql(
        {
            "source_scoped_filters": {
                "gdrive": {
                    "workspace_ids": ["workspace-a"],
                    "metadata_non_empty": ["scope_memberships"],
                }
            }
        }
    )

    sql_text = " AND ".join(clauses)
    assert "r.workspace_id = ANY(%s)" in sql_text
    assert "jsonb_array_length" in sql_text
    assert ["workspace-a"] in parameters
    assert "workspace-a" not in sql_text
