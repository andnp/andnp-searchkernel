"""
Source-agnostic search/indexing kernel.

A domain-agnostic library for building hybrid vector+keyword+graph search systems
with pluggable embedding/LLM/reranker providers.
"""

from searchkernel.search.orchestrator import SearchOrchestrator
from searchkernel.search.pipeline import SearchPipelineConfig
from searchkernel.search.utils import classify_query_type, truncate_content

__version__ = "0.1.0"

__all__ = [
    "SearchOrchestrator",
    "SearchPipelineConfig",
    "classify_query_type",
    "truncate_content",
]
