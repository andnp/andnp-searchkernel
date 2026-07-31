"""Optional source-balanced post-fusion candidate selection."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from searchkernel.domain import ScoredRef
from searchkernel.ports.retrieval import RetrievalFields


@dataclass(frozen=True, slots=True)
class DiversityDiagnostic:
    """Deterministic explanation for a diversity policy decision."""

    reason: str
    source_id: str | None = None
    source_kind: str | None = None


@dataclass(frozen=True, slots=True)
class SourceDiversityPolicy:
    """Optional caps, reservations, and MMR settings for fused results."""

    enabled: bool = False
    max_per_source: int | None = None
    source_slot_caps: Mapping[str, int] = field(default_factory=dict)
    source_slot_reservations: Mapping[str, int] = field(default_factory=dict)
    max_per_document: int | None = None
    max_per_entity: int | None = None
    mmr_lambda: float | None = None
    min_score_ratio: float = 0.5
    relevance_floor: float | None = None

    def __post_init__(self) -> None:
        _validate_optional_cap("max_per_source", self.max_per_source)
        _validate_optional_cap("max_per_document", self.max_per_document)
        _validate_optional_cap("max_per_entity", self.max_per_entity)
        for name, values in (
            ("source_slot_caps", self.source_slot_caps),
            ("source_slot_reservations", self.source_slot_reservations),
        ):
            for source_kind, value in values.items():
                if not isinstance(source_kind, str) or not source_kind:
                    raise ValueError(f"{name} keys must be non-empty strings")
                if not isinstance(value, int) or value < 0:
                    raise ValueError(f"{name} values must be non-negative integers")
        for source_kind, reservation in self.source_slot_reservations.items():
            cap = self.source_slot_caps.get(source_kind, self.max_per_source)
            if cap is not None and reservation > cap:
                raise ValueError(
                    f"reservation for {source_kind!r} exceeds its source cap"
                )
        if not 0.0 <= self.min_score_ratio <= 1.0:
            raise ValueError("min_score_ratio must be between zero and one")
        if self.relevance_floor is not None and not math.isfinite(
            self.relevance_floor
        ):
            raise ValueError("relevance_floor must be finite")
        if self.mmr_lambda is not None and not 0.0 <= self.mmr_lambda <= 1.0:
            raise ValueError("mmr_lambda must be between zero and one")


def apply_source_diversity(
    candidates: Sequence[ScoredRef],
    *,
    top_n: int,
    policy: SourceDiversityPolicy | None = None,
    embeddings: Mapping[str, Sequence[float]] | None = None,
    diagnostics: list[DiversityDiagnostic] | None = None,
    document_key: Callable[[ScoredRef], str | None] | None = None,
    entity_key: Callable[[ScoredRef], str | None] | None = None,
) -> list[ScoredRef]:
    """Apply a guarded, deterministic diversity pass to fused candidates."""
    if top_n < 0:
        raise ValueError("top_n must be non-negative")
    effective_policy = policy or SourceDiversityPolicy()
    ordered = list(candidates)
    if not effective_policy.enabled:
        _diagnose(diagnostics, "disabled")
        return ordered[:top_n]
    if top_n == 0 or not ordered:
        _diagnose(diagnostics, "empty")
        return []

    if not _has_constraints(effective_policy):
        _diagnose(diagnostics, "no_constraints")
        return ordered[:top_n]

    best_score = max(candidate.score for candidate in ordered)
    score_floor = best_score * effective_policy.min_score_ratio
    if effective_policy.relevance_floor is not None:
        score_floor = max(score_floor, effective_policy.relevance_floor)
    eligible = [
        candidate
        for candidate in ordered
        if candidate.score >= score_floor
    ]
    if len(eligible) < min(top_n, len(ordered)):
        _diagnose(
            diagnostics,
            f"relevance_guard:{len(eligible)}/{min(top_n, len(ordered))}",
        )

    selected: list[ScoredRef] = []
    selected_keys: set[str] = set()
    source_counts: dict[str, int] = {}
    document_counts: dict[str, int] = {}
    entity_counts: dict[str, int] = {}

    for source_kind in sorted(effective_policy.source_slot_reservations):
        reservation = effective_policy.source_slot_reservations[source_kind]
        for candidate in eligible:
            if candidate.source_kind != source_kind:
                continue
            if len(selected) >= top_n:
                break
            if source_counts.get(source_kind, 0) >= reservation:
                break
            if _try_select(
                candidate,
                selected,
                selected_keys,
                source_counts,
                document_counts,
                entity_counts,
                effective_policy,
                diagnostics,
                document_key,
                entity_key,
            ):
                _diagnose(
                    diagnostics,
                    f"reserved:{source_kind}",
                    candidate,
                )

    remaining = [candidate for candidate in eligible if candidate.storage_key not in selected_keys]
    while remaining and len(selected) < top_n:
        candidate = _next_candidate(
            remaining,
            selected,
            effective_policy.mmr_lambda,
            embeddings,
        )
        remaining.remove(candidate)
        if _try_select(
            candidate,
            selected,
            selected_keys,
            source_counts,
            document_counts,
            entity_counts,
            effective_policy,
            diagnostics,
            document_key,
            entity_key,
        ):
            continue

    return selected


def _try_select(
    candidate: ScoredRef,
    selected: list[ScoredRef],
    selected_keys: set[str],
    source_counts: dict[str, int],
    document_counts: dict[str, int],
    entity_counts: dict[str, int],
    policy: SourceDiversityPolicy,
    diagnostics: list[DiversityDiagnostic] | None,
    document_key: Callable[[ScoredRef], str | None] | None,
    entity_key: Callable[[ScoredRef], str | None] | None,
) -> bool:
    identity = candidate.storage_key
    if identity in selected_keys:
        return False

    source_cap = policy.source_slot_caps.get(
        candidate.source_kind,
        policy.max_per_source,
    )
    if source_cap is not None and source_counts.get(candidate.source_kind, 0) >= source_cap:
        _diagnose(diagnostics, "source_cap", candidate)
        return False

    document = _resolve_group_key(candidate, "document", document_key)
    if (
        document is not None
        and policy.max_per_document is not None
        and document_counts.get(document, 0) >= policy.max_per_document
    ):
        _diagnose(diagnostics, "document_cap", candidate)
        return False

    entity = _resolve_group_key(candidate, "entity", entity_key)
    if (
        entity is not None
        and policy.max_per_entity is not None
        and entity_counts.get(entity, 0) >= policy.max_per_entity
    ):
        _diagnose(diagnostics, "entity_cap", candidate)
        return False

    selected.append(candidate)
    selected_keys.add(identity)
    source_counts[candidate.source_kind] = (
        source_counts.get(candidate.source_kind, 0) + 1
    )
    if document is not None:
        document_counts[document] = document_counts.get(document, 0) + 1
    if entity is not None:
        entity_counts[entity] = entity_counts.get(entity, 0) + 1
    return True


def _next_candidate(
    candidates: Sequence[ScoredRef],
    selected: Sequence[ScoredRef],
    mmr_lambda: float | None,
    embeddings: Mapping[str, Sequence[float]] | None,
) -> ScoredRef:
    if mmr_lambda is None:
        return candidates[0]
    if embeddings is None:
        return candidates[0]
    ranked = sorted(
        enumerate(candidates),
        key=lambda item: (
            -_mmr_score(item[1], selected, mmr_lambda, embeddings),
            item[0],
            item[1].storage_key,
        ),
    )
    return ranked[0][1]


def _mmr_score(
    candidate: ScoredRef,
    selected: Sequence[ScoredRef],
    mmr_lambda: float,
    embeddings: Mapping[str, Sequence[float]],
) -> float:
    relevance = candidate.score
    candidate_embedding = _embedding_for(candidate, embeddings)
    if candidate_embedding is None or not selected:
        return relevance
    redundancy = max(
        (
            _cosine_similarity(candidate_embedding, other_embedding)
            for other in selected
            if (other_embedding := _embedding_for(other, embeddings)) is not None
        ),
        default=0.0,
    )
    return mmr_lambda * relevance - (1.0 - mmr_lambda) * redundancy


def _embedding_for(
    candidate: ScoredRef,
    embeddings: Mapping[str, Sequence[float]],
) -> Sequence[float] | None:
    return embeddings.get(candidate.storage_key) or embeddings.get(candidate.source_id)


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _resolve_group_key(
    candidate: ScoredRef,
    group: str,
    key_fn: Callable[[ScoredRef], str | None] | None,
) -> str | None:
    if key_fn is not None:
        return key_fn(candidate)
    retrieval_fields = candidate.metadata.get("retrieval_fields")
    if isinstance(retrieval_fields, RetrievalFields):
        return retrieval_fields.parent_id if group == "document" else None
    if isinstance(retrieval_fields, Mapping):
        value = retrieval_fields.get("parent_id") if group == "document" else None
        return str(value) if value is not None else None
    aliases = (
        ("document_id", "doc_id", "parent_id")
        if group == "document"
        else ("entity_id", "entity")
    )
    for alias in aliases:
        value = candidate.metadata.get(alias)
        if value is not None:
            return str(value)
    return None


def _has_constraints(policy: SourceDiversityPolicy) -> bool:
    return any(
        (
            policy.max_per_source is not None,
            bool(policy.source_slot_caps),
            bool(policy.source_slot_reservations),
            policy.max_per_document is not None,
            policy.max_per_entity is not None,
            policy.mmr_lambda is not None,
        )
    )


def _validate_optional_cap(name: str, value: int | None) -> None:
    if value is not None and (not isinstance(value, int) or value < 1):
        raise ValueError(f"{name} must be a positive integer or None")


def _diagnose(
    diagnostics: list[DiversityDiagnostic] | None,
    reason: str,
    candidate: ScoredRef | None = None,
) -> None:
    if diagnostics is None:
        return
    diagnostics.append(
        DiversityDiagnostic(
            reason=reason,
            source_id=candidate.source_id if candidate is not None else None,
            source_kind=candidate.source_kind if candidate is not None else None,
        )
    )


__all__ = [
    "DiversityDiagnostic",
    "SourceDiversityPolicy",
    "apply_source_diversity",
]
