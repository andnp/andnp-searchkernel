"""Transport-neutral federation over the :class:`SearchSource` port."""

from __future__ import annotations

import asyncio
import inspect
import math
import time
from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Literal, cast
from urllib.parse import urlsplit, urlunsplit

from searchkernel.ports.federation import (
    FEDERATION_CONTRACT_VERSION,
    MAX_RERANK_TEXT_LENGTH,
    FederationEventKind,
    SearchHit,
    SearchRequest,
    SearchResponse,
    SearchSource,
    SourceCapabilities,
    SourceIdentity,
)
from searchkernel.ports.rerank import Reranker
from searchkernel.search.fusion import fuse_reciprocal_rank

DEFAULT_MAX_CONCURRENCY = 8
DEFAULT_PER_SOURCE_TIMEOUT_S = 5.0
DEFAULT_RRF_K = 60.0

DegradationStatus = Literal["unavailable", "timeout", "partial", "rerank"]


@dataclass(frozen=True, slots=True)
class RegisteredSearchSource:
    """A source together with the identity used for routing and diagnostics."""

    identity: SourceIdentity
    source: SearchSource


@dataclass(frozen=True, slots=True)
class FederationDiagnostic:
    """A source or reranker degradation captured without failing the query."""

    source: SourceIdentity | None
    status: DegradationStatus
    message: str
    exception_type: str = "Exception"


@dataclass(frozen=True, slots=True)
class FederatedSearchResponse:
    """Fused hits plus source responses and explicit degradation details."""

    hits: tuple[SearchHit, ...] = ()
    partial: bool = False
    degradations: tuple[FederationDiagnostic, ...] = ()
    warnings: tuple[str, ...] = ()
    source_responses: tuple[SearchResponse, ...] = ()
    fusion_scores: Mapping[str, float] = field(default_factory=dict)
    elapsed_ms: float = 0.0
    authoritative: bool = True

    @property
    def degraded(self) -> bool:
        return self.partial or bool(self.degradations)

    @property
    def results(self) -> tuple[SearchHit, ...]:
        """Alias matching the result naming used by other search APIs."""
        return self.hits


@dataclass(frozen=True, slots=True)
class FederationEvent:
    """An opt-in progressive update with an explicit finality contract."""

    kind: FederationEventKind
    source: SourceIdentity | None = None
    source_response: SearchResponse | None = None
    result: FederatedSearchResponse | None = None
    diagnostic: FederationDiagnostic | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("source", "provisional", "authoritative"):
            raise ValueError(f"unsupported federation event kind: {self.kind}")
        if self.kind == "source":
            if self.source is None or self.result is not None:
                raise ValueError("source events require a source and no result")
            return
        if self.result is None or self.source is not None:
            raise ValueError(f"{self.kind} events require only a result")
        if self.kind == "provisional" and self.result.authoritative:
            raise ValueError("provisional results must not be authoritative")
        if self.kind == "authoritative" and not self.result.authoritative:
            raise ValueError("authoritative events require an authoritative result")

    @property
    def event_type(self) -> FederationEventKind:
        """Alias for consumers that use event-type terminology."""
        return self.kind

    @property
    def authoritative(self) -> bool:
        """Whether this event carries the one final fused result."""
        return self.kind == "authoritative"

    @property
    def provisional(self) -> bool:
        """Whether this event carries an explicitly non-final result."""
        return self.kind == "provisional"

    @property
    def response(self) -> SearchResponse | None:
        """Compatibility alias for the source response payload."""
        return self.source_response


@dataclass(frozen=True, slots=True)
class FederationConfig:
    """Bounds and ranking policy for a federation execution."""

    max_concurrency: int = DEFAULT_MAX_CONCURRENCY
    per_source_timeout_s: float = DEFAULT_PER_SOURCE_TIMEOUT_S
    per_source_top_k: int | None = None
    rrf_k: float = DEFAULT_RRF_K
    rerank_candidate_limit: int = 100
    max_rerank_text_length: int = MAX_RERANK_TEXT_LENGTH

    def __post_init__(self) -> None:
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be positive")
        if self.per_source_timeout_s <= 0:
            raise ValueError("per_source_timeout_s must be positive")
        if self.per_source_top_k is not None and self.per_source_top_k < 1:
            raise ValueError("per_source_top_k must be positive")
        if self.rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
        if self.rerank_candidate_limit < 1:
            raise ValueError("rerank_candidate_limit must be positive")
        if self.max_rerank_text_length < 1:
            raise ValueError("max_rerank_text_length must be positive")


@dataclass(frozen=True, slots=True)
class _SourceHit:
    source_index: int
    source: SourceIdentity
    hit: SearchHit


class _UnionFind:
    def __init__(self) -> None:
        self._parent: list[int] = []

    def add(self) -> int:
        index = len(self._parent)
        self._parent.append(index)
        return index

    def find(self, index: int) -> int:
        parent = self._parent[index]
        if parent != index:
            self._parent[index] = self.find(parent)
        return self._parent[index]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self._parent[right_root] = left_root


class FederationExecutor:
    """Execute concurrent source searches and fuse their ordered results."""

    def __init__(
        self,
        sources: (
            Mapping[str | SourceIdentity, SearchSource]
            | Iterable[
                SearchSource
                | RegisteredSearchSource
                | tuple[SourceIdentity, SearchSource]
            ]
        ),
        *,
        config: FederationConfig | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.config = config or FederationConfig()
        self.reranker = reranker
        self._sources = tuple(_normalize_sources(sources))

    @property
    def sources(self) -> tuple[RegisteredSearchSource, ...]:
        return self._sources

    async def search(self, request: SearchRequest) -> FederatedSearchResponse:
        """Search eligible sources, fuse by local rank, and return top-k hits."""
        started = time.perf_counter()
        selected, skipped = self._select_sources(request)
        diagnostics = list(skipped)
        responses: list[SearchResponse] = []
        source_hits: list[list[_SourceHit]] = []
        outcomes: list[tuple[SearchResponse | None, FederationDiagnostic | None] | None] = [
            None
        ] * len(selected)
        async for index, outcome in self._run_sources(selected, request):
            outcomes[index] = outcome
        for outcome in outcomes:
            assert outcome is not None
            self._add_outcome(
                outcome,
                request,
                responses,
                source_hits,
                diagnostics,
            )
        return await self._build_response(
            request,
            started,
            responses,
            source_hits,
            diagnostics,
            authoritative=True,
        )

    async def stream(self, request: SearchRequest) -> AsyncIterator[FederationEvent]:
        """Yield opt-in source/provisional updates, then one final result."""
        started = time.perf_counter()
        selected, skipped = self._select_sources(request)
        diagnostics = list(skipped)
        responses: list[SearchResponse] = []
        source_hits: list[list[_SourceHit]] = []
        outcomes: list[tuple[SearchResponse | None, FederationDiagnostic | None] | None] = [
            None
        ] * len(selected)
        async for index, outcome in self._run_sources(selected, request):
            registration = selected[index]
            outcomes[index] = outcome
            response, source_diagnostic = outcome
            yield FederationEvent(
                kind="source",
                source=registration.identity,
                source_response=response,
                diagnostic=source_diagnostic,
            )
            self._add_outcome(
                outcome,
                request,
                responses,
                source_hits,
                diagnostics,
            )
            if response is not None:
                provisional = await self._build_response(
                    request,
                    started,
                    responses,
                    source_hits,
                    diagnostics,
                    authoritative=False,
                    rerank=False,
                )
                yield FederationEvent(kind="provisional", result=provisional)

        final_diagnostics = list(skipped)
        final_responses: list[SearchResponse] = []
        final_source_hits: list[list[_SourceHit]] = []
        for outcome in outcomes:
            assert outcome is not None
            self._add_outcome(
                outcome,
                request,
                final_responses,
                final_source_hits,
                final_diagnostics,
            )
        authoritative = await self._build_response(
            request,
            started,
            final_responses,
            final_source_hits,
            final_diagnostics,
            authoritative=True,
        )
        yield FederationEvent(kind="authoritative", result=authoritative)

    async def search_events(self, request: SearchRequest) -> AsyncIterator[FederationEvent]:
        """Named alias for :meth:`stream` for event-oriented callers."""
        async for event in self.stream(request):
            yield event

    async def events(self, request: SearchRequest) -> AsyncIterator[FederationEvent]:
        """Short alias for the opt-in progressive event stream."""
        async for event in self.stream(request):
            yield event

    async def execute(self, request: SearchRequest) -> FederatedSearchResponse:
        """Explicit executor alias for callers that prefer command semantics."""
        return await self.search(request)

    def _select_sources(
        self,
        request: SearchRequest,
    ) -> tuple[list[RegisteredSearchSource], list[FederationDiagnostic]]:
        selected: list[RegisteredSearchSource] = []
        diagnostics: list[FederationDiagnostic] = []
        requested = set(request.source_selection)
        for registration in self._sources:
            if requested and not _matches_selection(registration.identity, requested):
                continue
            capabilities = _safe_capabilities(registration.source)
            if FEDERATION_CONTRACT_VERSION not in capabilities.contract_versions:
                diagnostics.append(
                    FederationDiagnostic(
                        source=registration.identity,
                        status="unavailable",
                        message=(
                            f"source does not support {FEDERATION_CONTRACT_VERSION}"
                        ),
                    )
                )
                continue
            if request.filters and not capabilities.supports_filters:
                diagnostics.append(
                    FederationDiagnostic(
                        source=registration.identity,
                        status="unavailable",
                        message="source does not support requested filters",
                    )
                )
                continue
            selected.append(registration)
        return selected, diagnostics

    async def _search_source(
        self,
        registration: RegisteredSearchSource,
        request: SearchRequest,
    ) -> tuple[SearchResponse | None, FederationDiagnostic | None]:
        capabilities = _safe_capabilities(registration.source)
        top_k = request.top_k
        if self.config.per_source_top_k is not None:
            top_k = min(top_k, self.config.per_source_top_k)
        top_k = min(top_k, capabilities.max_top_k)
        source_request = replace(
            request,
            top_k=top_k,
            deadline_at=_source_deadline(
                request.deadline_at,
                self.config.per_source_timeout_s,
            ),
        )
        timeout = _remaining_timeout(source_request.deadline_at)
        if timeout <= 0:
            return None, FederationDiagnostic(
                source=registration.identity,
                status="timeout",
                message="source deadline elapsed before execution",
                exception_type="TimeoutError",
            )
        try:
            result = await asyncio.wait_for(
                registration.source.search(source_request),
                timeout=timeout,
            )
            if not isinstance(result, SearchResponse):
                raise TypeError("SearchSource.search must return SearchResponse")
            return result, None
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            return None, FederationDiagnostic(
                source=registration.identity,
                status="timeout",
                message=f"source timed out after {timeout:.3g}s",
                exception_type="TimeoutError",
            )
        except Exception as error:  # noqa: BLE001 - source failures are isolated
            return None, FederationDiagnostic(
                source=registration.identity,
                status="unavailable",
                message=str(error) or type(error).__name__,
                exception_type=type(error).__name__,
            )

    async def _run_sources(
        self,
        selected: Sequence[RegisteredSearchSource],
        request: SearchRequest,
    ) -> AsyncIterator[tuple[int, tuple[SearchResponse | None, FederationDiagnostic | None]]]:
        """Run sources with at most ``max_concurrency`` active worker tasks."""
        if not selected:
            return
        queue: asyncio.Queue[
            tuple[int, tuple[SearchResponse | None, FederationDiagnostic | None]]
        ] = asyncio.Queue(maxsize=self.config.max_concurrency)
        next_index = 0

        async def worker() -> None:
            nonlocal next_index
            while next_index < len(selected):
                index = next_index
                next_index += 1
                try:
                    outcome = await self._search_source(selected[index], request)
                except asyncio.CancelledError:
                    raise
                except Exception as error:  # noqa: BLE001 - isolate worker failures
                    outcome = (
                        None,
                        FederationDiagnostic(
                            source=selected[index].identity,
                            status="unavailable",
                            message=str(error) or type(error).__name__,
                            exception_type=type(error).__name__,
                        ),
                    )
                await queue.put((index, outcome))

        tasks = [
            asyncio.create_task(worker())
            for _ in range(min(self.config.max_concurrency, len(selected)))
        ]
        try:
            for _ in selected:
                yield await queue.get()
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    def _add_outcome(
        self,
        outcome: tuple[SearchResponse | None, FederationDiagnostic | None],
        request: SearchRequest,
        responses: list[SearchResponse],
        source_hits: list[list[_SourceHit]],
        diagnostics: list[FederationDiagnostic],
    ) -> None:
        response, source_diagnostic = outcome
        if source_diagnostic is not None:
            diagnostics.append(source_diagnostic)
        if response is None:
            return
        responses.append(response)
        if response.partial:
            diagnostics.append(
                FederationDiagnostic(
                    source=response.source,
                    status="partial",
                    message="source returned partial results",
                )
            )
        source_hits.append(
            [
                _SourceHit(
                    source_index=index,
                    source=response.source,
                    hit=_with_provenance(hit, response.source, request),
                )
                for index, hit in enumerate(
                    sorted(
                        response.hits,
                        key=lambda item: (item.source_rank, _identity_key(item)),
                    )
                )
            ]
        )

    async def _build_response(
        self,
        request: SearchRequest,
        started: float,
        responses: Sequence[SearchResponse],
        source_hits: Sequence[Sequence[_SourceHit]],
        diagnostics: Sequence[FederationDiagnostic],
        *,
        authoritative: bool,
        rerank: bool = True,
    ) -> FederatedSearchResponse:
        fused_hits, fusion_scores = _fuse_hits(source_hits, self.config.rrf_k)
        all_diagnostics = list(diagnostics)
        if rerank:
            rerank_diagnostic = await self._rerank(
                request.query,
                fused_hits,
                fusion_scores,
            )
            if rerank_diagnostic is not None:
                all_diagnostics.append(rerank_diagnostic)
        warnings = tuple(
            warning for response in responses for warning in response.warnings
        ) + tuple(_diagnostic_warning(diagnostic) for diagnostic in all_diagnostics)
        return FederatedSearchResponse(
            hits=tuple(fused_hits[: request.top_k]),
            partial=bool(all_diagnostics),
            degradations=tuple(all_diagnostics),
            warnings=warnings,
            source_responses=tuple(responses),
            fusion_scores=fusion_scores,
            elapsed_ms=_elapsed_ms(started),
            authoritative=authoritative,
        )

    async def _rerank(
        self,
        query: str,
        hits: list[SearchHit],
        fusion_scores: Mapping[str, float],
    ) -> FederationDiagnostic | None:
        if self.reranker is None or not hits:
            return None
        eligible_positions = [
            index
            for index, hit in enumerate(hits[: self.config.rerank_candidate_limit])
            if hit.rerank_text
        ]
        if not eligible_positions:
            return None
        texts = [
            (hits[index].rerank_text or "")[: self.config.max_rerank_text_length]
            for index in eligible_positions
        ]
        try:
            scores = self.reranker.rerank(query, texts)
            if inspect.isawaitable(scores):
                scores = await scores
            scores = list(scores)
            if len(scores) != len(eligible_positions):
                raise ValueError(
                    f"reranker returned {len(scores)} scores for "
                    f"{len(eligible_positions)} candidates"
                )
            ranked = sorted(
                zip(eligible_positions, scores, strict=True),
                key=lambda item: (
                    _validate_rerank_score(item[1]),
                    -fusion_scores[_identity_key(hits[item[0]])],
                    item[0],
                ),
            )
            original_hits = hits.copy()
            for position, (original_index, _) in zip(
                eligible_positions,
                ranked,
                strict=True,
            ):
                hits[position] = original_hits[original_index]
            return None
        except Exception as error:  # noqa: BLE001 - reranker is optional
            return FederationDiagnostic(
                source=None,
                status="rerank",
                message=str(error) or type(error).__name__,
                exception_type=type(error).__name__,
            )


def _normalize_sources(
    sources: (
        Mapping[str | SourceIdentity, SearchSource]
        | Iterable[
            SearchSource
            | RegisteredSearchSource
            | tuple[SourceIdentity, SearchSource]
        ]
    ),
) -> list[RegisteredSearchSource]:
    if isinstance(sources, Mapping):
        items = sources.items()
    else:
        items = (_source_item(registration) for registration in sources)
    normalized: list[RegisteredSearchSource] = []
    for identity, source in items:
        if isinstance(identity, SourceIdentity):
            source_identity = identity
        else:
            source_identity = SourceIdentity(str(identity), str(identity))
        if not isinstance(source, SearchSource):
            raise TypeError("sources must implement SearchSource")
        normalized.append(RegisteredSearchSource(source_identity, source))
    return normalized


def _source_item(
    registration: (
        SearchSource
        | RegisteredSearchSource
        | tuple[SourceIdentity, SearchSource]
    ),
) -> tuple[SourceIdentity, SearchSource]:
    if isinstance(registration, RegisteredSearchSource):
        return registration.identity, registration.source
    if isinstance(registration, tuple):
        return registration
    identity = getattr(registration, "identity", None)
    if not isinstance(identity, SourceIdentity):
        identity = getattr(registration, "source_identity", None)
    if not isinstance(identity, SourceIdentity):
        raise TypeError(
            "bare SearchSource registrations must expose a SourceIdentity "
            "as identity or source_identity"
        )
    return identity, registration


def _safe_capabilities(source: SearchSource) -> SourceCapabilities:
    capabilities = source.capabilities()
    if not isinstance(capabilities, SourceCapabilities):
        raise TypeError("SearchSource.capabilities must return SourceCapabilities")
    return capabilities


def _matches_selection(identity: SourceIdentity, requested: set[str]) -> bool:
    available = {identity.source_kind, identity.source_id}
    if identity.workspace_id is not None:
        available.add(identity.workspace_id)
    return bool(available & requested)


def _source_deadline(
    request_deadline: datetime | None,
    timeout_s: float,
) -> datetime:
    bounded = datetime.now(UTC).timestamp() + timeout_s
    timeout_deadline = datetime.fromtimestamp(bounded, UTC)
    if request_deadline is None:
        return timeout_deadline
    return min(request_deadline, timeout_deadline)


def _remaining_timeout(deadline: datetime | None) -> float:
    if deadline is None:
        return float("inf")
    return max(0.0, deadline.timestamp() - datetime.now(UTC).timestamp())


def _with_provenance(
    hit: SearchHit,
    source: SourceIdentity,
    request: SearchRequest,
) -> SearchHit:
    provenance = hit.provenance
    if provenance.source is not None and provenance.request_id is not None:
        return hit
    return replace(
        hit,
        provenance=replace(
            provenance,
            source=provenance.source or source,
            request_id=provenance.request_id or request.request_id or None,
        ),
    )


def _fuse_hits(
    source_hits: Sequence[Sequence[_SourceHit]],
    rrf_k: float,
) -> tuple[list[SearchHit], dict[str, float]]:
    union_find = _UnionFind()
    identity_nodes: dict[str, int] = {}
    uri_nodes: dict[str, int] = {}
    entries: list[list[tuple[int, _SourceHit, int]]] = []

    for source_index, hits in enumerate(source_hits):
        source_entries: list[tuple[int, _SourceHit, int]] = []
        for rank, entry in enumerate(hits, start=1):
            identity_key = _identity_key(entry.hit)
            uri_key = _uri_key(entry.hit.uri)
            node = union_find.add()
            existing = identity_nodes.get(identity_key)
            if existing is not None:
                union_find.union(node, existing)
            else:
                identity_nodes[identity_key] = node
            if uri_key is not None:
                existing = uri_nodes.get(uri_key)
                if existing is not None:
                    union_find.union(node, existing)
                else:
                    uri_nodes[uri_key] = node
            source_entries.append((rank, entry, node))
        entries.append(source_entries)

    rankings: list[list[str]] = []
    candidates: dict[int, SearchHit] = {}
    candidate_keys: dict[int, str] = {}
    first_seen: dict[int, tuple[int, int, str]] = {}
    for source_index, source_entries in enumerate(entries):
        ranking: list[str] = []
        seen: set[int] = set()
        for rank, entry, node in source_entries:
            root = union_find.find(node)
            if root in seen:
                continue
            seen.add(root)
            ranking.append(str(root))
            if root not in candidates:
                candidates[root] = entry.hit
                candidate_keys[root] = _identity_key(entry.hit)
        if ranking:
            rankings.append(ranking)

    first_seen = {}
    for source_index, source_entries in enumerate(entries):
        for rank, entry, node in source_entries:
            root = union_find.find(node)
            first_seen.setdefault(
                root,
                (source_index, rank, _identity_key(entry.hit)),
            )

    scores = fuse_reciprocal_rank(rankings, k=rrf_k)
    ranked_roots = sorted(
        (int(key) for key in scores),
        key=lambda root: (
            -scores[str(root)],
            first_seen.get(root, (0, 0, candidate_keys.get(root, ""))),
        ),
    )
    ranked_hits = [candidates[root] for root in ranked_roots]
    hit_scores = {
        _identity_key(hit): scores[str(root)]
        for root, hit in zip(ranked_roots, ranked_hits, strict=True)
    }
    return ranked_hits, hit_scores


def _identity_key(hit: SearchHit) -> str:
    return hit.identity.storage_key


def _validate_rerank_score(value: object) -> float:
    score = float(cast(float, value))
    if not math.isfinite(score):
        raise ValueError("reranker scores must be finite")
    return -score


def _uri_key(uri: str | None) -> str | None:
    if not uri:
        return None
    parts = urlsplit(uri.strip())
    if not parts.scheme and not parts.netloc:
        return uri.rstrip("/")
    return urlunsplit(
        (
            parts.scheme.lower(),
            parts.netloc.lower(),
            parts.path.rstrip("/") or "/",
            parts.query,
            parts.fragment,
        )
    )


def _diagnostic_warning(diagnostic: FederationDiagnostic) -> str:
    source = (
        f"{diagnostic.source.source_kind}:{diagnostic.source.source_id}"
        if diagnostic.source is not None
        else "reranker"
    )
    return f"{source} {diagnostic.status}: {diagnostic.message}"


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000


FederatedSearchExecutor = FederationExecutor

__all__ = [
    "DEFAULT_MAX_CONCURRENCY",
    "DEFAULT_PER_SOURCE_TIMEOUT_S",
    "DEFAULT_RRF_K",
    "FederatedSearchExecutor",
    "FederatedSearchResponse",
    "FederationConfig",
    "FederationDiagnostic",
    "FederationEvent",
    "FederationEventKind",
    "FederationExecutor",
    "RegisteredSearchSource",
]
