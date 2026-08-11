"""Provider-neutral freshness contracts for derivative search caches."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from searchkernel.ports.epochs import SearchEpochs

VersionValue = int | str


@dataclass(frozen=True, slots=True)
class FreshnessPolicy:
    """Identity and version of the policy that produced a cached result."""

    identity: str
    version: str

    def __post_init__(self) -> None:
        if not isinstance(self.identity, str):
            raise TypeError("freshness policy identity must be a string")
        if not self.identity:
            raise ValueError("freshness policy identity must not be empty")
        if not isinstance(self.version, str):
            raise TypeError("freshness policy version must be a string")
        if not self.version:
            raise ValueError("freshness policy version must not be empty")


@dataclass(frozen=True, slots=True)
class VersionToken:
    """One provider-issued, authoritative version token."""

    name: str
    value: VersionValue

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("version token name must be a string")
        if not self.name:
            raise ValueError("version token name must not be empty")
        if isinstance(self.value, bool) or not isinstance(self.value, (int, str)):
            raise TypeError("version token value must be an integer or string")
        if isinstance(self.value, str) and not self.value:
            raise ValueError("version token value must not be empty")


@dataclass(frozen=True, slots=True)
class VersionSnapshot:
    """An immutable set of authoritative provider version tokens."""

    tokens: tuple[VersionToken, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.tokens, tuple):
            raise TypeError("version snapshot tokens must be a tuple")
        if any(not isinstance(token, VersionToken) for token in self.tokens):
            raise TypeError("version snapshot tokens must be VersionToken objects")
        names = [token.name for token in self.tokens]
        if len(names) != len(set(names)):
            raise ValueError("version snapshot token names must be unique")

    @classmethod
    def from_epochs(cls, epochs: SearchEpochs) -> VersionSnapshot:
        """Adapt the existing search epoch snapshot to version tokens."""
        return cls(
            tuple(
                VersionToken(lane, epochs.for_lane(lane))
                for lane in ("keyword", "vector", "graph")
            )
        )

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> VersionSnapshot:
        """Validate provider data and copy it into an immutable snapshot."""
        tokens: list[VersionToken] = []
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, (int, str)):
                raise TypeError("version token value must be an integer or string")
            tokens.append(VersionToken(name, value))
        return cls(tuple(tokens))


@dataclass(frozen=True, slots=True)
class FreshnessSnapshot:
    """Policy and authoritative versions captured for a cached result."""

    policy: FreshnessPolicy
    versions: VersionSnapshot

    def __post_init__(self) -> None:
        if not isinstance(self.policy, FreshnessPolicy):
            raise TypeError("freshness snapshot policy must be a FreshnessPolicy")
        if not isinstance(self.versions, VersionSnapshot):
            raise TypeError("freshness snapshot versions must be a VersionSnapshot")


@runtime_checkable
class FreshnessProvider(Protocol):
    """Provider boundary for reading the authoritative current snapshot."""

    def current_snapshot(self) -> FreshnessSnapshot:
        ...


class FreshnessStatus(StrEnum):
    """Outcome of validating a cached result against provider state."""

    FRESH = "fresh"
    STALE = "stale"
    DEGRADED = "degraded"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class FreshnessDecision:
    """Immutable cache-use and invalidation decision."""

    status: FreshnessStatus
    use_cached: bool
    invalidate: bool
    reason: str | None = None

    @property
    def is_fresh(self) -> bool:
        """Whether the cached result passed the freshness contract."""
        return self.status is FreshnessStatus.FRESH


def validate_fresh_hit(
    cached_snapshot: object,
    provider: FreshnessProvider,
    *,
    allow_stale: bool = False,
) -> FreshnessDecision:
    """Validate a cached snapshot and explicitly report degraded stale reads."""
    try:
        cached = _validated_snapshot(cached_snapshot)
        current = _validated_snapshot(provider.current_snapshot())
    except (TypeError, ValueError) as exc:
        return FreshnessDecision(
            FreshnessStatus.INVALID,
            use_cached=False,
            invalidate=True,
            reason=str(exc),
        )

    if cached == current:
        return FreshnessDecision(
            FreshnessStatus.FRESH,
            use_cached=True,
            invalidate=False,
        )

    reason = (
        "freshness policy identity or version changed"
        if cached.policy != current.policy
        else "authoritative version snapshot changed"
    )
    return FreshnessDecision(
        FreshnessStatus.DEGRADED if allow_stale else FreshnessStatus.STALE,
        use_cached=allow_stale,
        invalidate=True,
        reason=reason,
    )


def _validated_snapshot(value: object) -> FreshnessSnapshot:
    if not isinstance(value, FreshnessSnapshot):
        raise TypeError("freshness provider must return a FreshnessSnapshot")
    return value
