from datetime import UTC, datetime

import pytest

from searchkernel.domain import RecordStatus
from searchkernel.ports.federation import (
    CallerAuthorizationContext,
    SearchDiagnostics,
    SearchHit,
    SearchHitProvenance,
    SearchRequest,
    SearchResponse,
    SearchSource,
    SourceCapabilities,
    SourceIdentity,
)


def test_request_json_round_trip_preserves_authorization_and_metadata() -> None:
    request = SearchRequest(
        query="incident review",
        top_k=5,
        filters={"workspace": "personal", "archived": False},
        source_selection=("memory", "jira"),
        caller=CallerAuthorizationContext(
            caller_id="devkit",
            tenant_id="andy",
            scopes=("search:read",),
            claims={"trusted": True},
        ),
        deadline_at=datetime(2026, 8, 2, 20, 0, tzinfo=UTC),
        cancellation_id="cancel-1",
        request_id="request-1",
        trace_id="trace-1",
    )

    assert SearchRequest.from_json(request.to_json()) == request


def test_response_json_round_trip_preserves_record_identity_and_provenance() -> None:
    response = SearchResponse(
        source=SourceIdentity("memory", "daemon", "andy"),
        hits=(
            SearchHit(
                workspace_id="andy",
                source_kind="memory",
                source_id="memory-1",
                title="Incident review",
                snippet="A concise citation-safe result.",
                rerank_text="Bounded text for an optional shared reranker.",
                uri="memory://memory-1",
                source_rank=1,
                native_score=-0.25,
                created_at=datetime(2026, 8, 1, 12, 0, tzinfo=UTC),
                updated_at=datetime(2026, 8, 2, 12, 0, tzinfo=UTC),
                lifecycle=RecordStatus.ACTIVE,
                metadata={"kind": "note"},
                provenance=SearchHitProvenance(
                    source=SourceIdentity("memory", "daemon", "andy"),
                    request_id="request-1",
                    retrieval_method="keyword",
                ),
            ),
        ),
        index_epoch="epoch-7",
        elapsed_ms=12.5,
        partial=True,
        warnings=("jira timed out",),
        diagnostics=SearchDiagnostics(
            candidate_count=4,
            candidate_counts={"keyword": 3, "vector": 4},
            failures=("graph unavailable",),
            stage_timings_ms={"keyword": 1.5, "vector": 2.25},
        ),
        capabilities=SourceCapabilities(supports_rerank_text=True),
    )

    restored = SearchResponse.from_json(response.to_json())
    assert restored == response
    assert restored.hits[0].identity.storage_key == (
        'record:["andy","memory","memory-1"]'
    )


def test_contract_validation_rejects_unknown_fields_and_unbounded_text() -> None:
    with pytest.raises(ValueError, match="unknown fields"):
        SearchRequest.from_dict({"query": "x", "unexpected": True})

    with pytest.raises(ValueError, match="rerank_text"):
        SearchHit(
            source_kind="note",
            source_id="note-1",
            title="Note",
            snippet="snippet",
            source_rank=1,
            rerank_text="x" * 4_097,
        )


@pytest.mark.parametrize(
    ("factory", "error", "message"),
    [
        (lambda: SearchRequest("query", top_k=0), ValueError, "top_k"),
        (lambda: SearchRequest("query", contract_version="v2"), ValueError, "contract"),
        (
            lambda: SearchRequest("query", source_selection=("memory", "memory")),
            ValueError,
            "duplicates",
        ),
        (
            lambda: SearchHit(
                source_kind="note",
                source_id="note-1",
                title="Note",
                snippet="snippet",
                source_rank=1,
                native_score=float("nan"),
            ),
            ValueError,
            "finite",
        ),
    ],
)
def test_contract_validation_rejects_invalid_values(
    factory: object,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        factory()  # type: ignore[operator]


def test_contract_json_is_canonical_and_rejects_malformed_payload() -> None:
    first = SearchRequest("query", filters={"b": 2, "a": 1})
    second = SearchRequest("query", filters={"a": 1, "b": 2})

    assert first.to_json() == second.to_json()
    with pytest.raises(ValueError, match="valid JSON"):
        SearchRequest.from_json("{")


def test_search_source_is_a_local_protocol() -> None:
    class LocalSource:
        async def search(self, request: SearchRequest) -> SearchResponse:
            return SearchResponse(source=SourceIdentity("note", "local"))

        def capabilities(self) -> SourceCapabilities:
            return SourceCapabilities()

    assert isinstance(LocalSource(), SearchSource)
