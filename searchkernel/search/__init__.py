from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from searchkernel.search.orchestrator import SearchOrchestrator
    from searchkernel.search.pipeline import SearchPipeline
    from searchkernel.search.record_pipeline import RecordSearchPipeline

__all__ = [
    "RecordSearchPipeline",
    "SearchOrchestrator",
    "SearchPipeline",
]


def __getattr__(name: str):
    if name == "SearchOrchestrator":
        from searchkernel.search.orchestrator import SearchOrchestrator

        return SearchOrchestrator
    if name == "SearchPipeline":
        from searchkernel.search.pipeline import SearchPipeline

        return SearchPipeline
    if name == "RecordSearchPipeline":
        from searchkernel.search.record_pipeline import RecordSearchPipeline

        return RecordSearchPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
