"""Behavior conformance tests for representative search backends."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from benchmarks.evaluate_labeled_retrieval import load_labeled_fixture
from searchkernel.domain import Record, RecordIdentity
from tests.integration.backend_conformance import (
    BackendConformanceTarget,
    conformance_schema,
    seed_faiss_target,
    seed_local_target,
    seed_postgres_target,
    storage_keys,
)

FIXTURE = Path(__file__).parents[1] / "fixtures" / "representative_retrieval_corpus.json"


@pytest.fixture
def representative_records() -> list[Record]:
    """Load the canonical representative records for one isolated test."""
    records, _golden = load_labeled_fixture(FIXTURE)
    return records


@pytest.fixture
def local_target(
    tmp_path: Path, representative_records: list[Record]
) -> Iterator[BackendConformanceTarget]:
    """Provide local keyword and vector stores backed by a temporary database."""
    target = seed_local_target(representative_records, tmp_path / "records.db")
    try:
        yield target
    finally:
        target.close()


@pytest.fixture
def faiss_target(
    tmp_path: Path, representative_records: list[Record]
) -> Iterator[BackendConformanceTarget]:
    """Provide FAISS only when its optional dependency is installed."""
    try:
        import faiss  # noqa: F401
    except ImportError as error:
        pytest.skip(f"FAISS unavailable: {error}")

    target = seed_faiss_target(
        representative_records,
        tmp_path / "records.db",
        tmp_path / "faiss",
    )
    try:
        yield target
    finally:
        target.close()


@pytest.fixture
def postgres_target(
    request: pytest.FixtureRequest,
    pg_dsn: str,
    pg_cleanup_executor,
    representative_records: list[Record],
) -> Iterator[BackendConformanceTarget]:
    """Provide PostgreSQL only when the existing Docker fixture is available."""
    target = seed_postgres_target(
        representative_records,
        pg_dsn,
        conformance_schema(request.config),
    )
    try:
        yield target
    finally:
        pg_cleanup_executor.submit(target.close)


def _shared_identity_isolation_contract(
    target: BackendConformanceTarget,
) -> None:
    """Require workspace-scoped searches to preserve canonical identity."""
    queries = {
        "engineering": "search rollout brief",
        "product": "product search brief",
        "personal": "personal search brief",
    }
    for workspace, query in queries.items():
        expected = RecordIdentity(workspace, "issue", "shared-brief").storage_key
        hits = target.keyword(
            query,
            10,
            {"workspace_id": workspace, "source_kinds": ["issue"]},
        )
        assert set(storage_keys(hits)) == {expected}


def _shared_filter_contract(
    target: BackendConformanceTarget,
    records: list[Record],
) -> None:
    """Require workspace, source-kind, status, and candidate filters."""
    docs_key = RecordIdentity("product", "docs", "ranking-guide").storage_key
    assert set(
        storage_keys(
            target.keyword(
                "ranking",
                10,
                {"workspace_id": "product", "source_kinds": ["docs"]},
            )
        )
    ) == {docs_key}

    assert target.keyword("search", 10, {"statuses": ["archived"]}) == []
    candidate = next(record for record in records if record.source_id == "shared-brief")
    candidate_hits = target.keyword(
        "search rollout brief",
        10,
        {"candidate_ids": [candidate.identity]},
    )
    assert set(storage_keys(candidate_hits)) == {candidate.storage_key}


def _shared_boundary_contract(
    target: BackendConformanceTarget,
    records: list[Record],
) -> None:
    """Require empty, zero, and over-sized result boundaries."""
    assert target.keyword("lunar filesystem", 10) == []
    assert target.keyword("search", 0) == []
    oversized = target.keyword("search", len(records) + 1)
    assert oversized
    assert len(oversized) == len(target.keyword("search", len(records) + 2))
    assert target.vector([1.0, 0.0, 0.0, 0.0], 0) == []
    assert len(target.vector([1.0, 0.0, 0.0, 0.0], len(records) + 1)) == len(records)


def _shared_tie_order_contract(
    target: BackendConformanceTarget,
    records: list[Record],
) -> None:
    """Require stable identity results when vector scores tie.

    Backends may choose different deterministic tie orders, but no identity
    may be lost or reordered between repeated calls to the same backend.
    """
    personal_keys = sorted(
        record.storage_key for record in records if record.workspace_id == "personal"
    )
    filters = {"workspace_id": "personal"}
    first = storage_keys(target.vector([0.0, 0.0, 1.0, 1.0], len(personal_keys), filters))
    second = storage_keys(target.vector([0.0, 0.0, 1.0, 1.0], len(personal_keys), filters))

    assert set(first) == set(personal_keys)
    assert second == first


def _shared_mutation_contract(
    target: BackendConformanceTarget,
    records: list[Record],
) -> None:
    """Require vector deletion and re-upsert visibility for canonical IDs."""
    target_record = next(record for record in records if record.source_id == "shared-brief")
    filters = {"candidate_ids": [target_record.identity]}
    query = [1.0, 0.0, 0.0, 0.0]

    assert storage_keys(target.vector(query, 1, filters)) == (target_record.storage_key,)
    target.delete([target_record.identity])
    assert target.vector(query, 1, filters) == []
    target.upsert([target_record])
    assert storage_keys(target.vector(query, 1, filters)) == (target_record.storage_key,)


def test_local_conformance_preserves_canonical_identity_isolation(
    local_target: BackendConformanceTarget,
) -> None:
    """Local retrieval must isolate duplicate source IDs by workspace."""
    _shared_identity_isolation_contract(local_target)


def test_local_conformance_enforces_filters_and_boundaries(
    local_target: BackendConformanceTarget,
    representative_records: list[Record],
) -> None:
    """Local retrieval must enforce filters and result-size boundaries."""
    _shared_filter_contract(local_target, representative_records)
    _shared_boundary_contract(local_target, representative_records)


def test_local_conformance_orders_vector_ties_deterministically(
    local_target: BackendConformanceTarget,
    representative_records: list[Record],
) -> None:
    """Local exact vector ties must use stable canonical storage ordering."""
    _shared_tie_order_contract(local_target, representative_records)


def test_local_conformance_exposes_delete_upsert_visibility(
    local_target: BackendConformanceTarget,
    representative_records: list[Record],
) -> None:
    """Local vector mutations must update canonical search visibility."""
    _shared_mutation_contract(local_target, representative_records)


def test_faiss_conformance_reports_optional_environment(
    faiss_target: BackendConformanceTarget,
    representative_records: list[Record],
) -> None:
    """FAISS status is explicit and available vector behavior is shared."""
    _shared_identity_isolation_contract(faiss_target)
    _shared_boundary_contract(faiss_target, representative_records)
    _shared_tie_order_contract(faiss_target, representative_records)


def test_postgres_conformance_reports_optional_environment(
    postgres_target: BackendConformanceTarget,
    representative_records: list[Record],
) -> None:
    """PostgreSQL status is explicit and available behavior is shared."""
    _shared_identity_isolation_contract(postgres_target)
    _shared_filter_contract(postgres_target, representative_records)
    _shared_boundary_contract(postgres_target, representative_records)
    _shared_tie_order_contract(postgres_target, representative_records)
