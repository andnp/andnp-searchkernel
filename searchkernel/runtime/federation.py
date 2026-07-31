"""search_anything: the federation entrypoint for unified search.

Sources are queried concurrently, fused by rank position, deduplicated, then
optionally reranked once over the fused candidate set.
"""

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import (
    dataclass,
)
from dataclasses import (
    replace as dataclass_replace,
)
from typing import Any, Literal

from searchkernel.domain import ScoredRef
from searchkernel.ports.rerank import Reranker
from searchkernel.runtime.fanout import FanoutDiagnostic, gather_with_timeout
from searchkernel.runtime.registry import SourceRegistry
from searchkernel.search.fusion import fuse_reciprocal_rank

DEFAULT_PER_SOURCE_K = 10
DEFAULT_PER_SOURCE_TIMEOUT_S = 5.0
DEFAULT_RRF_K = 60.0


class FederationSearchError(RuntimeError):
    """Raised when strict federation cannot complete a retrieval stage."""


@dataclass
class FederationDiagnostic:
    stage: Literal["source", "rerank"]
    message: str
    exception_type: str = "Exception"


async def search_anything(
    query: str,
    *,
    registry: SourceRegistry,
    reranker: Reranker | None = None,
    sources: list[str] | None = None,
    top_n: int = 10,
    per_source_k: int = DEFAULT_PER_SOURCE_K,
    per_source_timeout_s: float = DEFAULT_PER_SOURCE_TIMEOUT_S,
    source_weights: dict[str, float] | None = None,
    filters: dict[str, Any] | None = None,
    failure_mode: Literal["strict", "lenient"] = "lenient",
    diagnostics: list[FederationDiagnostic] | None = None,
    candidate_hydrator: (
        Callable[[ScoredRef], str | None | Awaitable[str | None]] | None
    ) = None,
) -> list[ScoredRef]:
    """Fuse the registered sources into one reranked list of ScoredRefs.

    Args:
        query: The search query string.
        registry: SourceRegistry holding the candidate SearchableSources.
        reranker: Optionally scores the fused candidate texts once,
                  cross-source. RRF scores are returned if it is unavailable.
        sources: Optional subset of source_kind names to fan out to.
                 If None, every registered source is queried.
        top_n: Maximum number of results to return.
        per_source_k: How many candidates to request from each source.
        per_source_timeout_s: Timeout applied independently to each source.
        filters: Optional source-specific filters (opaque to core).

    Returns:
        A single ranked list of deduplicated ScoredRefs, ordered by reranker
        score when available or RRF score otherwise, truncated to top_n.
    """
    selected = registry.select(sources)
    if not selected:
        return []

    fanout_diagnostics: list[FanoutDiagnostic] = []
    per_source_results = await gather_with_timeout(
        [source.search(query, per_source_k, filters) for source in selected],
        per_timeout_s=per_source_timeout_s,
        failure_mode=failure_mode,
        diagnostics=fanout_diagnostics,
    )
    if diagnostics is not None:
        diagnostics.extend(
            FederationDiagnostic(
                stage="source",
                message=f"source {item.index}: {item.message}",
                exception_type=item.exception_type,
            )
            for item in fanout_diagnostics
        )

    rankings: dict[str, list[str]] = {}
    candidates: dict[str, ScoredRef] = {}
    source_scores: dict[str, dict[str, float]] = {}
    source_metadata: dict[str, dict[str, dict[str, Any]]] = {}
    first_seen: dict[str, int] = {}

    for results in per_source_results:
        if not results:
            continue

        source_name: str | None = None
        ranking: list[str] = []
        seen_in_source: set[str] = set()
        for candidate in results:
            if source_name is None:
                source_name = candidate.source_kind
            identity = candidate.storage_key
            if identity in seen_in_source:
                continue
            seen_in_source.add(identity)
            ranking.append(identity)

            if identity not in candidates:
                candidates[identity] = candidate
                first_seen[identity] = len(first_seen)

            source_scores.setdefault(identity, {})[
                candidate.source_kind
            ] = candidate.score
            source_metadata.setdefault(identity, {})[
                candidate.source_kind
            ] = dict(candidate.metadata)

        if ranking and source_name is not None:
            rankings[source_name] = ranking

    if not candidates:
        return []

    fused_scores = fuse_reciprocal_rank(
        rankings,
        k=DEFAULT_RRF_K,
        strategy_weights=source_weights,
    )
    fused = [
        dataclass_replace(
            candidates[source_id],
            score=rrf_score,
            metadata=_preserve_source_details(
                candidates[source_id],
                source_scores[source_id],
                source_metadata[source_id],
            ),
        )
        for source_id, rrf_score in sorted(
            fused_scores.items(),
            key=lambda item: (-item[1], first_seen[item[0]]),
        )
    ]

    if reranker is None:
        return fused[:top_n]

    texts: list[str] = []
    for candidate in fused:
        text = _candidate_text(candidate)
        if not text.strip() and candidate_hydrator is not None:
            try:
                text = await _maybe_await(candidate_hydrator(candidate)) or ""
            except Exception as error:  # noqa: BLE001 - heterogeneous hydrators
                _record_or_raise("rerank", error, failure_mode, diagnostics)
                return fused[:top_n]
        if not text.strip():
            error = FederationSearchError(
                f"rerank requires candidate text for "
                f"{candidate.source_kind}:{candidate.source_id}"
            )
            _record_or_raise("rerank", error, failure_mode, diagnostics)
            return fused[:top_n]
        texts.append(text)
    try:
        rerank_scores = await _maybe_await(reranker.rerank(query, texts))
    except Exception as error:  # noqa: BLE001 - heterogeneous reranker adapters
        _record_or_raise("rerank", error, failure_mode, diagnostics)
        return fused[:top_n]

    if len(rerank_scores) != len(fused):
        error = FederationSearchError(
            f"reranker returned {len(rerank_scores)} scores for {len(fused)} candidates"
        )
        _record_or_raise("rerank", error, failure_mode, diagnostics)
        return fused[:top_n]

    reranked = [
        dataclass_replace(candidate, score=rerank_score)
        for candidate, rerank_score in zip(fused, rerank_scores)
    ]
    reranked.sort(key=lambda ref: ref.score, reverse=True)
    return reranked[:top_n]


def _candidate_text(candidate: ScoredRef) -> str:
    text = candidate.metadata.get("text", "")
    return text if isinstance(text, str) else str(text)


async def _maybe_await[T](value: T | Awaitable[T]) -> T:
    if inspect.isawaitable(value):
        return await value
    return value


def _record_or_raise(
    stage: Literal["source", "rerank"],
    error: Exception,
    failure_mode: Literal["strict", "lenient"],
    diagnostics: list[FederationDiagnostic] | None,
) -> None:
    if failure_mode == "strict":
        raise error
    if diagnostics is not None:
        diagnostics.append(FederationDiagnostic(stage, str(error), type(error).__name__))


def _preserve_source_details(
    candidate: ScoredRef,
    source_scores: dict[str, float],
    source_metadata: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    metadata = {**candidate.metadata, "source_score": candidate.score}
    if len(source_scores) > 1:
        metadata["source_scores"] = dict(source_scores)
        metadata["source_metadata"] = {
            source_kind: dict(details)
            for source_kind, details in source_metadata.items()
        }
    return metadata
