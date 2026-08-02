"""Source candidate acquisition for the record search pipeline."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

from searchkernel.domain import RecordHit, Vector
from searchkernel.ports import (
    AsyncKeywordStore,
    AsyncVectorStore,
    KeywordStore,
    VectorStore,
)
from searchkernel.search.query_plan import QueryPlan

if TYPE_CHECKING:
    from searchkernel.search.record_pipeline import (
        RecordSearchPolicy,
        RecordSearchQueryContext,
    )

QueryEmbedding = Callable[[str], Awaitable[tuple[Vector, str, int]]]


class CandidateAcquirer:
    """Acquire normalized keyword and vector candidate rankings."""

    def __init__(
        self,
        *,
        keyword_store: KeywordStore | AsyncKeywordStore | None,
        vector_store: VectorStore | AsyncVectorStore | None,
        policy: RecordSearchPolicy,
        query_embedding: QueryEmbedding,
    ) -> None:
        self._keyword_store = keyword_store
        self._vector_store = vector_store
        self._policy = policy
        self._query_embedding = query_embedding

    async def keyword(
        self,
        query: str,
        acquisition_limit: int,
        filters: dict[str, object],
    ) -> list[RecordHit]:
        store = self._keyword_store
        if store is None:
            return []
        search = cast(Callable[..., Any], store.search)
        return _normalize_hits(
            await _call_async(search, query, acquisition_limit, filters),
        )

    async def vector(
        self,
        embedding: tuple[Vector, str, int] | None,
        acquisition_limit: int,
        filters: dict[str, object],
        rankings: Mapping[str, Sequence[RecordHit]],
        *,
        context: RecordSearchQueryContext,
        query: str | None = None,
        plan: QueryPlan | None = None,
    ) -> list[RecordHit]:
        if embedding is None:
            if query is None:
                raise ValueError("query is required when embedding is absent")
            embedding = await self._query_embedding(query)
        vector, model_name, dim = embedding
        vector_filters = filters
        if self._policy.vector_candidate_ids is not None:
            candidate_ids = self._policy.vector_candidate_ids(
                list(rankings.get("keyword", ())),
                context,
            )
            if candidate_ids is not None:
                vector_filters = dict(filters)
                vector_filters["candidate_ids"] = list(candidate_ids)
        elif plan is not None and plan.vector_candidates_keyword_bounded:
            keyword_ranking = rankings.get("keyword", ())
            if keyword_ranking:
                vector_filters = dict(filters)
                vector_filters["candidate_storage_keys"] = [
                    hit.storage_key for hit in keyword_ranking
                ]
        vector_store = self._vector_store
        if vector_store is None:
            return []
        vector_ranking = _normalize_hits(
            await _search_vector_store(
                vector_store,
                vector,
                acquisition_limit,
                model_name=model_name,
                dim=dim,
                filters=vector_filters,
            ),
            sort=False,
        )
        if self._policy.vector_ranking_order is not None:
            vector_ranking = _normalize_hits(
                self._policy.vector_ranking_order(
                    vector_ranking,
                    context,
                ),
                sort=False,
            )
        return vector_ranking


def _normalize_hits(
    results: Sequence[RecordHit],
    *,
    sort: bool = True,
) -> list[RecordHit]:
    best: dict[str, RecordHit] = {}
    for hit in results:
        current = best.get(hit.storage_key)
        if current is None or hit.score > current.score:
            best[hit.storage_key] = hit
    normalized = list(best.values())
    if sort:
        normalized.sort(key=lambda hit: (-hit.score, hit.storage_key))
    return normalized


async def _search_vector_store(
    store: VectorStore | AsyncVectorStore,
    vector: Vector,
    k: int,
    *,
    model_name: str,
    dim: int,
    filters: dict[str, object] | None,
) -> Sequence[RecordHit]:
    async_search = getattr(store, "async_search", None)
    if callable(async_search):
        return await _call_async(
            cast(Callable[..., Any], async_search),
            vector,
            k,
            model_name=model_name,
            dim=dim,
            filters=filters,
        )
    return await _call_async(
        cast(Callable[..., Any], store.search),
        vector,
        k,
        model_name=model_name,
        dim=dim,
        filters=filters,
    )


async def _call_async[T](
    function: Callable[..., T | Awaitable[T]],
    *args: Any,
    **kwargs: Any,
) -> T:
    if inspect.iscoroutinefunction(function):
        value = function(*args, **kwargs)
    else:
        value = await asyncio.to_thread(function, *args, **kwargs)
    if inspect.isawaitable(value):
        return await value
    return value
