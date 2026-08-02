from datetime import UTC, datetime

import pytest

from searchkernel.domain import RecordStatus
from searchkernel.ports.federation import (
    CallerAuthorizationContext,
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


def test_search_source_is_a_local_protocol() -> None:
    class LocalSource:
        async def search(self, request: SearchRequest) -> SearchResponse:
            return SearchResponse(source=SourceIdentity("note", "local"))

        def capabilities(self) -> SourceCapabilities:
            return SourceCapabilities()

    assert isinstance(LocalSource(), SearchSource)
