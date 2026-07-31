"""search_anything: the federation entrypoint for unified search.

Sources are queried concurrently, fused by rank position, deduplicated, then
optionally reranked once over the fused candidate set.
"""

import logging
from dataclasses import replace as dataclass_replace
from typing import Any

from searchkernel.domain import ScoredRef
from searchkernel.ports.rerank import Reranker
from searchkernel.runtime.fanout import gather_with_timeout
from searchkernel.runtime.registry import SourceRegistry
from searchkernel.search.fusion import fuse_reciprocal_rank

logger = logging.getLogger(__name__)

DEFAULT_PER_SOURCE_K = 10
DEFAULT_PER_SOURCE_TIMEOUT_S = 5.0
DEFAULT_RRF_K = 60.0


async def search_anything(
    query: str,
    *,
    registry: SourceRegistry,
    reranker: Reranker | None = None,
    sources: list[str] | None = None,
    top_n: int = 10,
    per_source_k: int = DEFAULT_PER_SOURCE_K,
    per_source_timeout_s: float = DEFAULT_PER_SOURCE_TIMEOUT_S,
    filters: dict[str, Any] | None = None,
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

    per_source_results = await gather_with_timeout(
        [source.search(query, per_source_k, filters) for source in selected],
        per_timeout_s=per_source_timeout_s,
    )

    rankings: list[list[str]] = []
    candidates: dict[str, ScoredRef] = {}
    source_scores: dict[str, dict[str, float]] = {}
    source_metadata: dict[str, dict[str, dict[str, Any]]] = {}
    first_seen: dict[str, int] = {}

    for results in per_source_results:
        if not results:
            continue

        ranking: list[str] = []
        seen_in_source: set[str] = set()
        for candidate in results:
            if candidate.source_id in seen_in_source:
                continue
            seen_in_source.add(candidate.source_id)
            ranking.append(candidate.source_id)

            if candidate.source_id not in candidates:
                candidates[candidate.source_id] = candidate
                first_seen[candidate.source_id] = len(first_seen)

            source_scores.setdefault(candidate.source_id, {})[
                candidate.source_kind
            ] = candidate.score
            source_metadata.setdefault(candidate.source_id, {})[
                candidate.source_kind
            ] = dict(candidate.metadata)

        if ranking:
            rankings.append(ranking)

    if not candidates:
        return []

    fused_scores = fuse_reciprocal_rank(rankings, k=DEFAULT_RRF_K)
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

    texts = [_candidate_text(candidate) for candidate in fused]
    try:
        rerank_scores = reranker.rerank(query, texts)
    except Exception:
        logger.warning("Reranker failed; returning RRF results", exc_info=True)
        return fused[:top_n]

    if len(rerank_scores) != len(fused):
        logger.warning(
            "Reranker returned %d scores for %d candidates; returning RRF results",
            len(rerank_scores),
            len(fused),
        )
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
