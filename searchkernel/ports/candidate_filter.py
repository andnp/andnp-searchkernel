"""Optional candidate-ID filtering capability for source-owned adapters."""

from typing import ClassVar, Protocol, runtime_checkable


@runtime_checkable
class CandidateFilterSupport(Protocol):
    """Opt-in marker for adapters that accept candidate IDs during search.

    This capability describes only whether an adapter supports candidate-ID
    filtering. It does not define the source-specific search signature that
    accepts those IDs.
    """

    supports_candidate_filtering: ClassVar[bool]
