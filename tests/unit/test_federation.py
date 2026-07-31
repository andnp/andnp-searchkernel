"""Unit tests for search_anything: fan-out timeouts + retrieve-then-rerank-once."""

import asyncio
from collections.abc import Iterable
from typing import Any

import pytest

from searchkernel.domain import ScoredRef
from searchkernel.ports import Reranker, SearchableSource
from searchkernel.runtime.federation import (
    FederationSearchError,
    search_anything,
)
from searchkernel.runtime.registry import SourceRegistry


class _FastSource:
    source_kind = "fast"

    async def search(
        self, query: str, k: int, filters: dict[str, Any] | None = None
    ) -> Iterable[ScoredRef]:
        return [
            ScoredRef(
                source_id="fast-1",
                score=0.9,
                source_kind="fast",
                metadata={"text": "fast source result"},
            )
        ]


class _SlowSource:
    source_kind = "slow"

    async def search(
        self, query: str, k: int, filters: dict[str, Any] | None = None
    ) -> Iterable[ScoredRef]:
        await asyncio.sleep(1.0)
        return [
            ScoredRef(
                source_id="slow-1",
                score=0.9,
                source_kind="slow",
                metadata={"text": "slow source result"},
            )
        ]


class _FailingSource:
    source_kind = "failing"

    async def search(
        self, query: str, k: int, filters: dict[str, Any] | None = None
    ) -> Iterable[ScoredRef]:
        raise RuntimeError("source is down")


class _StubReranker:
    """Deterministic reranker: score = position of the text in a fixed table."""

    model_name = "stub-reranker"

    def __init__(self, scores_by_text: dict[str, float]):
        self._scores_by_text = scores_by_text

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        return [self._scores_by_text.get(doc, 0.0) for doc in documents]


def _identity_reranker() -> _StubReranker:
    return _StubReranker({})


@pytest.mark.asyncio
async def test_slow_source_times_out_others_still_return():
    registry = SourceRegistry()
    registry.register(_FastSource())
    registry.register(_SlowSource())
    reranker = _StubReranker({"fast source result": 1.0})

    results = await search_anything(
        "query",
        registry=registry,
        reranker=reranker,
        per_source_timeout_s=0.05,
    )

    assert [r.source_id for r in results] == ["fast-1"]


@pytest.mark.asyncio
async def test_failing_source_yields_no_results_others_still_return():
    registry = SourceRegistry()
    registry.register(_FastSource())
    registry.register(_FailingSource())
    reranker = _StubReranker({"fast source result": 1.0})

    results = await search_anything(
        "query",
        registry=registry,
        reranker=reranker,
    )

    assert [r.source_id for r in results] == ["fast-1"]


@pytest.mark.asyncio
async def test_no_sources_registered_returns_empty():
    registry = SourceRegistry()

    results = await search_anything(
        "query",
        registry=registry,
        reranker=_identity_reranker(),
    )

    assert results == []


@pytest.mark.asyncio
async def test_top_n_truncates_the_fused_list():
    registry = SourceRegistry()
    registry.register(_FastSource())
    reranker = _StubReranker({"fast source result": 1.0})

    results = await search_anything(
        "query",
        registry=registry,
        reranker=reranker,
        top_n=0,
    )

    assert results == []


@pytest.mark.asyncio
async def test_sources_param_restricts_fan_out_to_named_sources():
    registry = SourceRegistry()
    registry.register(_FastSource())
    registry.register(_FailingSource())
    reranker = _StubReranker({"fast source result": 1.0})

    results = await search_anything(
        "query",
        registry=registry,
        reranker=reranker,
        sources=["fast"],
    )

    assert [r.source_id for r in results] == ["fast-1"]


def test_reranker_protocol_conformance_of_stub():
    assert isinstance(_StubReranker({}), Reranker)


def test_searchable_source_protocol_conformance_of_stubs():
    assert isinstance(_FastSource(), SearchableSource)
    assert isinstance(_SlowSource(), SearchableSource)
    assert isinstance(_FailingSource(), SearchableSource)


@pytest.mark.asyncio
async def test_duplicate_results_are_fused_once_with_source_details():
    class _Source:
        def __init__(self, source_kind, results):
            self.source_kind = source_kind
            self._results = results

        async def search(self, query, k, filters=None):
            return self._results

    registry = SourceRegistry()
    registry.register(
        _Source(
            "first",
            [
                ScoredRef(
                    source_id="shared",
                    score=0.9,
                    source_kind="first",
                    metadata={"text": "shared result", "origin": "first"},
                )
            ],
        )
    )
    registry.register(
        _Source(
            "second",
            [
                ScoredRef(
                    source_id="shared",
                    score=0.2,
                    source_kind="second",
                    metadata={"text": "shared result", "origin": "second"},
                )
            ],
        )
    )

    results = await search_anything("query", registry=registry)

    assert len(results) == 2
    assert {result.source_kind for result in results} == {"first", "second"}
    assert all(result.source_id == "shared" for result in results)


@pytest.mark.asyncio
async def test_rrf_preserves_source_order_for_equal_rank_scores():
    class _Source:
        def __init__(self, source_kind, source_id):
            self.source_kind = source_kind
            self._source_id = source_id

        async def search(self, query, k, filters=None):
            return [
                ScoredRef(
                    source_id=self._source_id,
                    score=0.1,
                    source_kind=self.source_kind,
                )
            ]

    registry = SourceRegistry()
    registry.register(_Source("first", "first-1"))
    registry.register(_Source("second", "second-1"))

    results = await search_anything("query", registry=registry)

    assert [result.source_id for result in results] == ["first-1", "second-1"]
    assert [result.score for result in results] == pytest.approx([1 / 61, 1 / 61])


@pytest.mark.asyncio
async def test_failed_reranker_falls_back_to_rrf():
    class _FailingReranker:
        model_name = "failing-reranker"

        def rerank(self, query, documents):
            raise RuntimeError("reranker unavailable")

    registry = SourceRegistry()
    registry.register(_FastSource())

    results = await search_anything(
        "query",
        registry=registry,
        reranker=_FailingReranker(),
    )

    assert [result.source_id for result in results] == ["fast-1"]
    assert results[0].score == pytest.approx(1 / 61)
    assert results[0].metadata["source_score"] == 0.9


@pytest.mark.asyncio
async def test_no_reranker_path_returns_rrf_scores():
    registry = SourceRegistry()
    registry.register(_FastSource())

    results = await search_anything("query", registry=registry)

    assert [result.source_id for result in results] == ["fast-1"]
    assert results[0].score == pytest.approx(1 / 61)
    assert results[0].metadata["source_score"] == 0.9


@pytest.mark.asyncio
async def test_rerank_once_reorders_by_reranker_not_source_score():
    """The merged set is ordered by the single rerank pass, not each source's
    own score -- a source's high self-reported score should not win if the
    reranker judges it less relevant than a lower-scored candidate."""
    registry = SourceRegistry()

    class _SourceA:
        source_kind = "a"

        async def search(self, query, k, filters=None):
            return [
                ScoredRef(
                    source_id="a-1",
                    score=0.99,  # highest self-reported score
                    source_kind="a",
                    metadata={"text": "irrelevant filler text"},
                )
            ]

    class _SourceB:
        source_kind = "b"

        async def search(self, query, k, filters=None):
            return [
                ScoredRef(
                    source_id="b-1",
                    score=0.1,  # lowest self-reported score
                    source_kind="b",
                    metadata={"text": "the most relevant answer"},
                )
            ]

    registry.register(_SourceA())
    registry.register(_SourceB())
    reranker = _StubReranker(
        {
            "irrelevant filler text": 0.05,
            "the most relevant answer": 0.95,
        }
    )

    results = await search_anything(
        "query",
        registry=registry,
        reranker=reranker,
    )

    assert [r.source_id for r in results] == ["b-1", "a-1"]
    assert results[0].score == 0.95
    assert results[0].metadata["source_score"] == 0.1


@pytest.mark.asyncio
async def test_reranker_requires_text_or_explicit_hydrator() -> None:
    class _Source:
        source_kind = "empty"

        async def search(self, query, k, filters=None):
            return [ScoredRef("id", 0.5, "empty")]

    registry = SourceRegistry()
    registry.register(_Source())

    with pytest.raises(FederationSearchError, match="candidate text"):
        await search_anything(
            "query",
            registry=registry,
            reranker=_identity_reranker(),
            failure_mode="strict",
        )

    results = await search_anything(
        "query",
        registry=registry,
        reranker=_identity_reranker(),
        candidate_hydrator=lambda candidate: "hydrated text",
    )
    assert results[0].source_id == "id"


@pytest.mark.asyncio
async def test_lenient_candidate_hydrator_failure_returns_diagnostic() -> None:
    class _Source:
        source_kind = "empty"

        async def search(self, query, k, filters=None):
            return [ScoredRef("id", 0.5, "empty")]

    registry = SourceRegistry()
    registry.register(_Source())
    diagnostics = []

    async def hydrate(_candidate):
        raise RuntimeError("hydrate failed")

    results = await search_anything(
        "query",
        registry=registry,
        reranker=_identity_reranker(),
        candidate_hydrator=hydrate,
        diagnostics=diagnostics,
        failure_mode="lenient",
    )

    assert [result.source_id for result in results] == ["id"]
    assert diagnostics[0].stage == "rerank"
    assert diagnostics[0].exception_type == "RuntimeError"
