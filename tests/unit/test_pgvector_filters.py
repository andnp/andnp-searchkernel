from searchkernel.adapters.stores.pgvector import build_pgvector_filter_sql
from searchkernel.domain import RecordIdentity, RecordStatus


def test_pgvector_filter_sql_covers_identity_lifecycle_metadata_filters() -> None:
    identity = RecordIdentity("workspace", "note", "kept")
    clauses, parameters = build_pgvector_filter_sql(
        {
            "workspace_id": "workspace",
            "source_kinds": ["note"],
            "statuses": [RecordStatus.ACTIVE],
            "candidate_ids": [identity, "other"],
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
    assert "metadata->>'project_id'" in sql_text
    assert "metadata->>'file_path'" in sql_text
    assert "metadata->>'doc_id'" in sql_text
    assert parameters[0] == ["active"]
    assert identity.storage_key in parameters[3]
    assert "other" in parameters[4]


def test_pgvector_filter_sql_rejects_empty_candidate_filter() -> None:
    clauses, parameters = build_pgvector_filter_sql({"candidate_ids": []})

    assert clauses == ["FALSE"]
    assert parameters == []
