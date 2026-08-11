from dataclasses import FrozenInstanceError
from typing import cast

import pytest

from searchkernel.ports import (
    FreshnessPolicy,
    FreshnessSnapshot,
    FreshnessStatus,
    VersionSnapshot,
    VersionToken,
    validate_fresh_hit,
)
from searchkernel.ports.epochs import SearchEpochs


class FakeProvider:
    def __init__(self, snapshot: object) -> None:
        self.snapshot = snapshot

    def current_snapshot(self) -> FreshnessSnapshot:
        return cast(FreshnessSnapshot, self.snapshot)


def _snapshot(
    *, policy_identity: str = "hybrid", policy_version: str = "1", keyword: int = 1
) -> FreshnessSnapshot:
    return FreshnessSnapshot(
        policy=FreshnessPolicy(policy_identity, policy_version),
        versions=VersionSnapshot(
            (VersionToken("keyword", keyword), VersionToken("vector", 2))
        ),
    )


def test_unchanged_snapshot_is_a_fresh_hit() -> None:
    """An identical policy and version snapshot permits the cached result."""
    snapshot = _snapshot()

    decision = validate_fresh_hit(snapshot, FakeProvider(snapshot))

    assert decision.status is FreshnessStatus.FRESH
    assert decision.use_cached is True
    assert decision.invalidate is False
    assert decision.is_fresh is True


def test_changed_authoritative_snapshot_invalidates_hit() -> None:
    """A changed provider token rejects the cached result as stale."""
    decision = validate_fresh_hit(
        _snapshot(), FakeProvider(_snapshot(keyword=2))
    )

    assert decision.status is FreshnessStatus.STALE
    assert decision.use_cached is False
    assert decision.invalidate is True
    assert decision.reason == "authoritative version snapshot changed"


def test_policy_identity_or_version_mismatch_invalidates_hit() -> None:
    """A changed policy identity invalidates results made by another policy."""
    decision = validate_fresh_hit(
        _snapshot(), FakeProvider(_snapshot(policy_identity="semantic"))
    )

    assert decision.status is FreshnessStatus.STALE
    assert decision.invalidate is True
    assert decision.reason == "freshness policy identity or version changed"


def test_allowed_stale_read_is_explicitly_degraded() -> None:
    """Permission to read stale data reports degraded status explicitly."""
    decision = validate_fresh_hit(
        _snapshot(), FakeProvider(_snapshot(keyword=2)), allow_stale=True
    )

    assert decision.status is FreshnessStatus.DEGRADED
    assert decision.use_cached is True
    assert decision.invalidate is True
    assert decision.is_fresh is False


def test_malformed_provider_data_is_never_a_cache_hit() -> None:
    """Malformed provider output is invalid and cannot be treated as fresh."""
    decision = validate_fresh_hit(_snapshot(), FakeProvider({"keyword": 1}))

    assert decision.status is FreshnessStatus.INVALID
    assert decision.use_cached is False
    assert decision.invalidate is True
    assert "FreshnessSnapshot" in (decision.reason or "")


def test_epoch_adapter_and_objects_are_immutable() -> None:
    """Existing epoch snapshots adapt to immutable version tokens."""
    versions = VersionSnapshot.from_epochs(SearchEpochs(keyword=1, vector=2, graph=3))

    assert versions.tokens == (
        VersionToken("keyword", 1),
        VersionToken("vector", 2),
        VersionToken("graph", 3),
    )
    with pytest.raises(FrozenInstanceError):
        versions.__setattr__("tokens", ())


@pytest.mark.parametrize("value", [True, object()])
def test_provider_version_data_requires_typed_immutable_tokens(
    value: object,
) -> None:
    """Provider mappings reject values outside the typed token contract."""
    with pytest.raises(TypeError):
        VersionSnapshot.from_mapping({"keyword": value})
