import asyncio
from dataclasses import dataclass, field

import pytest

import searchkernel.runtime.federation as federation_runtime
from searchkernel.ports.federation import (
    SearchDiagnostics,
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
    requests: list[SearchRequest] = field(default_factory=list)

    def capabilities(self) -> SourceCapabilities:
        return self.source_capabilities

    async def search(self, request: SearchRequest) -> SearchResponse:
        self.requests.append(request)
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
async def test_federation_preserves_source_diagnostics() -> None:
    source_identity = SourceIdentity("memory", "local")
    source_response = SearchResponse(
        source=source_identity,
        diagnostics=SearchDiagnostics(
            candidate_count=2,
            candidate_counts={"keyword": 2},
            failures=("vector unavailable",),
            stage_timings_ms={"keyword": 1.0},
        ),
    )
    response = await FederationExecutor(
        [FakeSource(source_identity, response=source_response)]
    ).search(SearchRequest("query"))

    assert response.diagnostics == (source_response.diagnostics,)


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
async def test_federation_selects_requested_sources_and_reports_capability_degradation():
    selected = source("selected", (hit("selected", "ok", 1),))
    unsupported = source(
        "unsupported",
        (hit("unsupported", "no", 1),),
        source_capabilities=SourceCapabilities(supports_filters=False),
    )
    unselected = source("unselected", (hit("unselected", "no", 1),))

    response = await FederationExecutor(
        [
            RegisteredSearchSource(selected.identity, selected),
            RegisteredSearchSource(unsupported.identity, unsupported),
            RegisteredSearchSource(unselected.identity, unselected),
        ]
    ).search(
        SearchRequest(
            "query",
            source_selection=("selected", "unsupported"),
            filters={"workspace": "andy"},
            top_k=10,
        )
    )

    assert [item.source_id for item in response.hits] == ["ok"]
    assert [request.top_k for request in selected.requests] == [10]
    assert not unselected.requests
    assert [(item.source, item.status) for item in response.degradations] == [
        (unsupported.identity, "unavailable")
    ]


@pytest.mark.asyncio
async def test_federation_surfaces_source_partial_response_and_warning():
    identity = SourceIdentity("partial", "partial")
    source_response = SearchResponse(
        source=identity,
        hits=(hit("partial", "result", 1),),
        partial=True,
        warnings=("index warming",),
    )
    partial_source = FakeSource(identity=identity, response=source_response)

    response = await FederationExecutor(
        [RegisteredSearchSource(identity, partial_source)]
    ).search(SearchRequest("query"))

    assert response.hits[0].source_id == "result"
    assert response.partial
    assert response.source_responses == (source_response,)
    assert response.warnings == (
        "index warming",
        "partial:partial partial: source returned partial results",
    )
    assert response.degradations[0].status == "partial"


@pytest.mark.asyncio
async def test_federation_rank_fusion_uses_local_rank_not_native_score():
    first = source(
        "first",
        (
            hit("first", "first-1", 1, native_score=-1),
            hit("first", "first-2", 2, native_score=10_000),
        ),
    )
    second = source(
        "second",
        (
            hit("second", "second-1", 1, native_score=-1),
            hit("first", "first-2", 2, native_score=-10_000),
        ),
    )

    response = await FederationExecutor(
        [
            RegisteredSearchSource(first.identity, first),
            RegisteredSearchSource(second.identity, second),
        ]
    ).search(SearchRequest("query", top_k=4))

    assert [item.source_id for item in response.hits] == [
        "first-2",
        "first-1",
        "second-1",
    ]


@pytest.mark.asyncio
async def test_federation_deduplicates_identity_and_canonical_uri():
    first = source(
        "first",
        (
            hit("first", "same", 1, uri="https://example.test/item/"),
            hit("first", "uri-only", 2, uri="https://EXAMPLE.test/other/"),
        ),
    )
    second = source(
        "second",
        (
            hit("first", "same", 1, uri="https://example.test/item"),
            hit("second", "different-id", 2, uri="https://example.test/other"),
        ),
    )

    response = await FederationExecutor(
        [
            RegisteredSearchSource(first.identity, first),
            RegisteredSearchSource(second.identity, second),
        ]
    ).search(SearchRequest("query", top_k=10))

    assert [(item.source_kind, item.source_id) for item in response.hits] == [
        ("first", "same"),
        ("first", "uri-only"),
    ]


@pytest.mark.asyncio
async def test_federation_order_is_deterministic_for_equal_scores():
    first = source("first", (hit("first", "first", 1),))
    second = source("second", (hit("second", "second", 1),))
    executor = FederationExecutor(
        [
            RegisteredSearchSource(first.identity, first),
            RegisteredSearchSource(second.identity, second),
        ]
    )

    responses = [
        await executor.search(SearchRequest("query", top_k=10))
        for _ in range(3)
    ]

    assert all(response.hits == responses[0].hits for response in responses)
    assert [item.source_id for item in responses[0].hits] == ["first", "second"]


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


@pytest.mark.asyncio
async def test_federation_reranker_is_bounded_by_candidates_and_text_length():
    class Reranker:
        def __init__(self) -> None:
            self.documents: list[str] = []

        def rerank(self, query: str, documents: list[str]) -> list[float]:
            self.documents = documents
            return [0.0, 1.0]

    reranker = Reranker()
    ranked = source(
        "ranked",
        (
            hit("ranked", "one", 1, rerank_text="one text"),
            hit("ranked", "two", 2, rerank_text="two text"),
            hit("ranked", "three", 3, rerank_text="three text"),
        ),
    )

    response = await FederationExecutor(
        [RegisteredSearchSource(ranked.identity, ranked)],
        reranker=reranker,
        config=FederationConfig(
            rerank_candidate_limit=2,
            max_rerank_text_length=4,
        ),
    ).search(SearchRequest("query"))

    assert reranker.documents == ["one ", "two "]
    assert [item.source_id for item in response.hits] == ["two", "one", "three"]


@pytest.mark.asyncio
async def test_federation_adds_end_to_end_provenance_to_every_hit():
    first = source(
        "first",
        (
            hit("first", "one", 1),
            hit("first", "two", 2),
        ),
    )
    request = SearchRequest("query", request_id="request-42")

    response = await FederationExecutor(
        [RegisteredSearchSource(first.identity, first)]
    ).search(request)

    assert [
        (item.provenance.source, item.provenance.request_id)
        for item in response.hits
    ] == [(first.identity, "request-42")] * 2


@pytest.mark.asyncio
async def test_federation_creates_only_bounded_worker_tasks(monkeypatch):
    created = 0
    original_create_task = asyncio.create_task

    def count_task(coro):
        nonlocal created
        created += 1
        return original_create_task(coro)

    monkeypatch.setattr(federation_runtime.asyncio, "create_task", count_task)
    sources = [source(str(index), ()) for index in range(20)]
    executor = FederationExecutor(
        [RegisteredSearchSource(item.identity, item) for item in sources],
        config=FederationConfig(max_concurrency=3),
    )

    await executor.search(SearchRequest("query"))

    assert created == 3


@pytest.mark.asyncio
async def test_federation_stream_marks_provisional_results_and_one_authoritative_result():
    fast = source("fast", (hit("fast", "quick", 1),), delay=0.001)
    slow = source("slow", (hit("slow", "later", 1),), delay=0.02)
    executor = FederationExecutor(
        [
            RegisteredSearchSource(fast.identity, fast),
            RegisteredSearchSource(slow.identity, slow),
        ],
        config=FederationConfig(max_concurrency=2),
    )

    events = [
        event
        async for event in executor.search_events(SearchRequest("query", top_k=5))
    ]

    assert [event.kind for event in events] == [
        "source",
        "provisional",
        "source",
        "provisional",
        "authoritative",
    ]
    assert events[0].source == fast.identity
    assert events[0].source_response is not None
    assert all(not event.authoritative for event in events[:-1])
    assert [
        event.result.authoritative
        for event in events
        if event.kind == "provisional" and event.result is not None
    ] == [False, False]
    assert events[-1].authoritative
    assert events[-1].result is not None
    assert events[-1].result.authoritative
    assert [item.source_id for item in events[-1].result.hits] == ["quick", "later"]
    assert [item.source for item in events[-1].result.source_responses] == [
        fast.identity,
        slow.identity,
    ]
