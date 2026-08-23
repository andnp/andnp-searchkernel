"""Evaluation-only capability descriptors for backend evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, Self


class CapabilityState(StrEnum):
    """Observed state of one backend capability."""

    SUPPORTED = "supported"
    UNAVAILABLE = "unavailable"
    UNSUPPORTED = "unsupported"


CapabilityName = Literal[
    "keyword",
    "exact_vector",
    "approximate_vector",
    "graph",
    "filtering",
    "candidate_filtering",
    "deletion",
]

_CAPABILITY_NAMES: tuple[CapabilityName, ...] = (
    "keyword",
    "exact_vector",
    "approximate_vector",
    "graph",
    "filtering",
    "candidate_filtering",
    "deletion",
)


def _require_object(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} keys must be strings")
    return value


def _require_exact_keys(
    value: Mapping[str, object], expected: set[str], name: str
) -> None:
    actual = set(value)
    missing = expected - actual
    unexpected = actual - expected
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append(f"missing {sorted(missing)}")
        if unexpected:
            details.append(f"unexpected {sorted(unexpected)}")
        raise ValueError(f"{name} has invalid fields: {', '.join(details)}")


def _state(value: object, name: str) -> CapabilityState:
    if isinstance(value, CapabilityState):
        return value
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a capability state")
    try:
        return CapabilityState(value)
    except ValueError as error:
        raise ValueError(
            f"{name} must be supported, unavailable, or unsupported"
        ) from error


@dataclass(frozen=True, slots=True)
class OptionalDependency:
    """Availability evidence for one optional backend dependency."""

    name: str
    state: CapabilityState
    reason: str | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("optional dependency name must not be empty")
        if not isinstance(self.state, CapabilityState):
            raise TypeError("optional dependency state must be a CapabilityState")
        if self.reason is not None and not self.reason.strip():
            raise ValueError("optional dependency reason must not be empty")

    def to_dict(self) -> dict[str, object]:
        """Serialize dependency availability for an evidence report."""
        return {"name": self.name, "state": self.state.value, "reason": self.reason}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        """Deserialize one dependency availability record."""
        data = _require_object(value, "optional dependency")
        keys = {"name", "state", "reason"}
        _require_exact_keys(data, keys, "optional dependency")
        name = data["name"]
        if not isinstance(name, str):
            raise TypeError("optional dependency name must be a string")
        reason = data["reason"]
        if reason is not None and not isinstance(reason, str):
            raise TypeError("optional dependency reason must be a string or None")
        return cls(name, _state(data["state"], "optional dependency state"), reason)


@dataclass(frozen=True, slots=True)
class BackendCapabilityDescriptor:
    """Evaluation metadata describing one backend's observable surfaces."""

    backend: str
    keyword: CapabilityState
    exact_vector: CapabilityState
    approximate_vector: CapabilityState
    graph: CapabilityState
    filtering: CapabilityState
    candidate_filtering: CapabilityState
    deletion: CapabilityState
    optional_dependencies: tuple[OptionalDependency, ...] = ()

    def __post_init__(self) -> None:
        if not self.backend.strip():
            raise ValueError("backend must not be empty")
        for name, state in self._capability_states().items():
            if not isinstance(state, CapabilityState):
                raise TypeError(f"{name} must be a CapabilityState")
        dependencies = tuple(self.optional_dependencies)
        if any(not isinstance(item, OptionalDependency) for item in dependencies):
            raise TypeError("optional_dependencies must contain OptionalDependency values")
        names = [item.name for item in dependencies]
        if len(names) != len(set(names)):
            raise ValueError("optional dependency names must be unique")
        object.__setattr__(self, "optional_dependencies", dependencies)

    def _capability_states(self) -> dict[CapabilityName, CapabilityState]:
        return {
            "keyword": self.keyword,
            "exact_vector": self.exact_vector,
            "approximate_vector": self.approximate_vector,
            "graph": self.graph,
            "filtering": self.filtering,
            "candidate_filtering": self.candidate_filtering,
            "deletion": self.deletion,
        }

    def state_for(self, capability: CapabilityName) -> CapabilityState:
        """Return the observed state for one named capability."""
        if capability not in _CAPABILITY_NAMES:
            raise ValueError(f"unknown capability: {capability}")
        return self._capability_states()[capability]

    def supports(self, capability: CapabilityName) -> bool:
        """Return whether a capability is explicitly supported."""
        return self.state_for(capability) is CapabilityState.SUPPORTED

    def to_dict(self) -> dict[str, object]:
        """Serialize the descriptor as JSON-compatible evidence."""
        return {
            "backend": self.backend,
            "capabilities": {
                name: state.value for name, state in self._capability_states().items()
            },
            "optional_dependencies": [
                dependency.to_dict() for dependency in self.optional_dependencies
            ],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        """Deserialize and validate one backend descriptor."""
        data = _require_object(value, "backend capability descriptor")
        _require_exact_keys(
            data,
            {"backend", "capabilities", "optional_dependencies"},
            "backend capability descriptor",
        )
        backend = data["backend"]
        if not isinstance(backend, str):
            raise TypeError("backend must be a string")
        capability_data = _require_object(data["capabilities"], "capabilities")
        _require_exact_keys(
            capability_data,
            set(_CAPABILITY_NAMES),
            "capabilities",
        )
        dependencies = data["optional_dependencies"]
        if not isinstance(dependencies, Sequence) or isinstance(
            dependencies, (str, bytes, bytearray)
        ):
            raise TypeError("optional_dependencies must be an array")
        parsed_dependencies: list[OptionalDependency] = []
        for item in dependencies:
            parsed_dependencies.append(
                OptionalDependency.from_dict(_require_object(item, "optional dependency"))
            )
        parsed_states = {
            name: _state(capability_data[name], f"capabilities.{name}")
            for name in _CAPABILITY_NAMES
        }
        return cls(
            backend=backend,
            optional_dependencies=tuple(parsed_dependencies),
            **parsed_states,
        )
