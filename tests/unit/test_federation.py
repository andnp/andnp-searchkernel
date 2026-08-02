import asyncio
from dataclasses import dataclass, field

import pytest

from searchkernel.ports.federation import (
    SearchHit,
    SearchRequest,
    SearchResponse,
    SourceCapabilities,
    SourceIdentity,
)
from searchkernel.runtime.federation import (
    FederationConfig,
    FederationExecutor,
    RegisteredSearchSource,
)


@dataclass
class FakeSource:
    identity: SourceIdentity
    response: SearchResponse | None = None
    delay: float = 0.0
    error: Exception | None = None
    source_capabilities: SourceCapabilities = field(
        default_factory=SourceCapabilities
    )

    def capabilities(self) -> SourceCapabilities:
        return self.source_capabilities

    async def search(self, request: SearchRequest) -> SearchResponse:
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        if self.response is None:
            return SearchResponse(source=self.identity)
        return self.response


def hit(
    source_kind: str,
    source_id: str,
    rank: int,
    *,
    uri: str | None = None,
    native_score: float | None = None,
    rerank_text: str | None = None,
) -> SearchHit:
    return SearchHit(
        source_kind=source_kind,
        source_id=source_id,
        title=source_id,
        snippet=source_id,
        source_rank=rank,
        uri=uri,
        native_score=native_score,
        rerank_text=rerank_text,
    )


def source(
    name: str,
    hits: tuple[SearchHit, ...],
    **kwargs: object,
) -> FakeSource:
    identity = SourceIdentity(name, name)
    return FakeSource(
        identity=identity,
        response=SearchResponse(source=identity, hits=hits),
        **kwargs,
    )


@pytest.mark.asyncio
async def test_federation_fuses_local_ranks_deduplicates_uri_and_preserves_provenance():
    first = source(
        "first",
        (
            hit("first", "one", 1, uri="https://EXAMPLE.test/one/", native_score=0.01),
            hit("first", "two", 2, native_score=1000),
        ),
    )
    second = source(
        "second",
        (
            hit("second", "duplicate", 1, uri="https://example.test/one"),
            hit("second", "three", 2),
        ),
    )

    response = await FederationExecutor(
        [
            RegisteredSearchSource(first.identity, first),
            RegisteredSearchSource(second.identity, second),
        ]
    ).search(SearchRequest("query", top_k=10, request_id="request-1"))

    assert [item.source_id for item in response.hits] == ["one", "two", "three"]
    assert response.hits[0].provenance.source == first.identity
    assert response.hits[0].provenance.request_id == "request-1"
    assert response.fusion_scores[response.hits[0].identity.storage_key] > (
        response.fusion_scores[response.hits[1].identity.storage_key]
    )


@pytest.mark.asyncio
async def test_federation_bounds_concurrency_and_reports_timeout_and_unavailable():
    slow = source("slow", (), delay=0.05)
    failed = source("failed", (), error=RuntimeError("offline"))
    fast = source("fast", (hit("fast", "ok", 1),))
    executor = FederationExecutor(
        [
            RegisteredSearchSource(slow.identity, slow),
            RegisteredSearchSource(failed.identity, failed),
            RegisteredSearchSource(fast.identity, fast),
        ],
        config=FederationConfig(max_concurrency=1, per_source_timeout_s=0.01),
    )

    response = await executor.search(SearchRequest("query", top_k=3))

    assert [item.source_id for item in response.hits] == ["ok"]
    assert response.partial
    assert {(item.source, item.status) for item in response.degradations} == {
        (slow.identity, "timeout"),
        (failed.identity, "unavailable"),
    }


@pytest.mark.asyncio
async def test_federation_filters_sources_by_selection_and_capabilities():
    selected = source("selected", (hit("selected", "ok", 1),))
    filtered = source(
        "filtered",
        (hit("filtered", "no", 1),),
        source_capabilities=SourceCapabilities(supports_filters=False),
    )

    response = await FederationExecutor(
        [
            RegisteredSearchSource(selected.identity, selected),
            RegisteredSearchSource(filtered.identity, filtered),
        ]
    ).search(
        SearchRequest(
            "query",
            source_selection=("selected",),
            filters={"workspace": "andy"},
        )
    )

    assert [item.source_id for item in response.hits] == ["ok"]
    assert not response.degradations


@pytest.mark.asyncio
async def test_federation_reranker_is_optional_for_hits_without_text():
    class Reranker:
        def __init__(self) -> None:
            self.documents: list[str] = []

        def rerank(self, query: str, documents: list[str]) -> list[float]:
            self.documents = documents
            return list(reversed(range(len(documents))))

    reranker = Reranker()
    one = source(
        "one",
        (hit("one", "a", 1, rerank_text="bounded text"),),
    )
    two = source("two", (hit("two", "b", 1),))

    response = await FederationExecutor(
        [
            RegisteredSearchSource(one.identity, one),
            RegisteredSearchSource(two.identity, two),
        ],
        reranker=reranker,
        config=FederationConfig(max_rerank_text_length=4),
    ).search(SearchRequest("query"))

    assert reranker.documents == ["boun"]
    assert [item.source_id for item in response.hits] == ["a", "b"]
