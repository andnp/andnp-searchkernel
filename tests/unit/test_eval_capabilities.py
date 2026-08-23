"""Tests for evaluation-only backend capability descriptors."""

import pytest

from searchkernel.eval.capabilities import (
    BackendCapabilityDescriptor,
    CapabilityState,
    OptionalDependency,
)


def _descriptor() -> BackendCapabilityDescriptor:
    """Build a representative descriptor with optional dependency evidence."""
    return BackendCapabilityDescriptor(
        backend="faiss-local",
        keyword=CapabilityState.SUPPORTED,
        exact_vector=CapabilityState.SUPPORTED,
        approximate_vector=CapabilityState.UNAVAILABLE,
        graph=CapabilityState.UNSUPPORTED,
        filtering=CapabilityState.SUPPORTED,
        candidate_filtering=CapabilityState.SUPPORTED,
        deletion=CapabilityState.SUPPORTED,
        optional_dependencies=(
            OptionalDependency("faiss-cpu", CapabilityState.UNAVAILABLE, "not installed"),
        ),
    )


def test_descriptor_round_trip_preserves_evidence() -> None:
    """Serialization preserves capability states and dependency diagnostics."""
    original = _descriptor()
    without_reason = OptionalDependency("tree-sitter", CapabilityState.UNSUPPORTED)

    restored = BackendCapabilityDescriptor.from_dict(original.to_dict())

    assert restored == original
    assert restored.to_dict() == original.to_dict()
    assert OptionalDependency.from_dict(without_reason.to_dict()) == without_reason


def test_unavailable_and_unsupported_are_distinct_non_support_states() -> None:
    """Only explicitly supported capabilities satisfy the support predicate."""
    descriptor = _descriptor()

    assert descriptor.state_for("approximate_vector") is CapabilityState.UNAVAILABLE
    assert descriptor.state_for("graph") is CapabilityState.UNSUPPORTED
    assert not descriptor.supports("approximate_vector")
    assert not descriptor.supports("graph")
    assert descriptor.supports("exact_vector")


def test_descriptor_rejects_invalid_or_ambiguous_state_data() -> None:
    """Malformed states and duplicate dependencies cannot enter evidence."""
    payload = _descriptor().to_dict()
    capabilities = payload["capabilities"]
    assert isinstance(capabilities, dict)
    capabilities["graph"] = "maybe"
    with pytest.raises(ValueError, match="supported, unavailable, or unsupported"):
        BackendCapabilityDescriptor.from_dict(payload)

    with pytest.raises(ValueError, match="names must be unique"):
        BackendCapabilityDescriptor(
            backend="local",
            keyword=CapabilityState.SUPPORTED,
            exact_vector=CapabilityState.SUPPORTED,
            approximate_vector=CapabilityState.UNSUPPORTED,
            graph=CapabilityState.UNSUPPORTED,
            filtering=CapabilityState.SUPPORTED,
            candidate_filtering=CapabilityState.SUPPORTED,
            deletion=CapabilityState.SUPPORTED,
            optional_dependencies=(
                OptionalDependency("faiss-cpu", CapabilityState.SUPPORTED),
                OptionalDependency("faiss-cpu", CapabilityState.UNAVAILABLE),
            ),
        )


def test_descriptor_rejects_ambiguous_serialized_shape() -> None:
    """Missing or extra capability fields fail instead of defaulting silently."""
    payload = _descriptor().to_dict()
    capabilities = payload["capabilities"]
    assert isinstance(capabilities, dict)
    del capabilities["graph"]

    with pytest.raises(ValueError, match="missing"):
        BackendCapabilityDescriptor.from_dict(payload)
