import pytest

from searchkernel.domain import ScoredRef
from searchkernel.search.diversity import (
    SourceDiversityPolicy,
    apply_source_diversity,
)


def _ref(source_kind: str, source_id: str, score: float, **metadata) -> ScoredRef:
    return ScoredRef(
        source_id=source_id,
        score=score,
        source_kind=source_kind,
        metadata=metadata,
    )


def test_source_caps_and_reservations_are_deterministic():
    candidates = [
        _ref("large", "large-1", 1.0),
        _ref("large", "large-2", 0.9),
        _ref("small", "small-1", 0.8),
        _ref("large", "large-3", 0.7),
    ]
    diagnostics = []

    results = apply_source_diversity(
        candidates,
        top_n=3,
        policy=SourceDiversityPolicy(
            enabled=True,
            max_per_source=2,
            source_slot_reservations={"small": 1},
        ),
        diagnostics=diagnostics,
    )

    assert [(result.source_kind, result.source_id) for result in results] == [
        ("small", "small-1"),
        ("large", "large-1"),
        ("large", "large-2"),
    ]
    assert diagnostics[0].reason == "reserved:small"


def test_document_and_entity_caps_limit_repeated_results():
    candidates = [
        _ref("notes", "one", 1.0, document_id="doc", entity_id="entity"),
        _ref("notes", "two", 0.9, document_id="doc", entity_id="other"),
        _ref("notes", "three", 0.8, document_id="other", entity_id="entity"),
        _ref("notes", "four", 0.7, document_id="other", entity_id="other"),
    ]

    results = apply_source_diversity(
        candidates,
        top_n=4,
        policy=SourceDiversityPolicy(
            enabled=True,
            max_per_document=1,
            max_per_entity=1,
        ),
    )

    assert [result.source_id for result in results] == ["one", "four"]


def test_mmr_is_deterministic_and_prefers_novel_relevant_candidates():
    candidates = [
        _ref("one", "first", 1.0),
        _ref("one", "similar", 0.95),
        _ref("two", "novel", 0.8),
    ]
    embeddings = {
        "first": [1.0, 0.0],
        "similar": [0.99, 0.01],
        "novel": [0.0, 1.0],
    }
    policy = SourceDiversityPolicy(enabled=True, mmr_lambda=0.2)

    first = apply_source_diversity(
        candidates,
        top_n=2,
        policy=policy,
        embeddings=embeddings,
    )
    second = apply_source_diversity(
        candidates,
        top_n=2,
        policy=policy,
        embeddings=embeddings,
    )

    assert [result.source_id for result in first] == ["first", "novel"]
    assert [result.source_id for result in second] == ["first", "novel"]


def test_relevance_guard_does_not_force_irrelevant_source():
    diagnostics = []
    results = apply_source_diversity(
        [
            _ref("dominant", "relevant", 1.0),
            _ref("other", "irrelevant", 0.1),
        ],
        top_n=2,
        policy=SourceDiversityPolicy(
            enabled=True,
            max_per_source=1,
            min_score_ratio=0.8,
        ),
        diagnostics=diagnostics,
    )

    assert [result.source_id for result in results] == ["relevant"]
    assert any(item.reason.startswith("relevance_guard:") for item in diagnostics)


def test_disabled_policy_preserves_order_and_reports_reason():
    diagnostics = []
    results = apply_source_diversity(
        [_ref("source", "one", 1.0), _ref("source", "two", 0.5)],
        top_n=1,
        policy=SourceDiversityPolicy(enabled=False),
        diagnostics=diagnostics,
    )

    assert [result.source_id for result in results] == ["one"]
    assert [item.reason for item in diagnostics] == ["disabled"]


def test_mmr_reports_embedding_fallback():
    diagnostics = []

    apply_source_diversity(
        [_ref("source", "one", 1.0), _ref("source", "two", 0.9)],
        top_n=2,
        policy=SourceDiversityPolicy(enabled=True, mmr_lambda=0.5),
        diagnostics=diagnostics,
    )

    assert [item.reason for item in diagnostics] == [
        "mmr:skipped:embeddings_unavailable"
    ]


def test_invalid_diversity_settings_are_rejected():
    with pytest.raises(ValueError, match="mmr_lambda"):
        SourceDiversityPolicy(enabled=True, mmr_lambda=1.1)

    with pytest.raises(ValueError, match="reservation"):
        SourceDiversityPolicy(
            enabled=True,
            max_per_source=1,
            source_slot_reservations={"source": 2},
        )
