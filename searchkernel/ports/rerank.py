"""Reranker port: adapters for relevance scoring and reranking.

Pluggable reranking models for refining search results. Implementations can wrap
HuggingFace, Ollama, or other reranking services. Scores are normalized to [0, 1]
with higher values indicating greater relevance.
"""

from typing import Protocol, runtime_checkable

from searchkernel.domain import Record


@runtime_checkable
class Reranker(Protocol):
    """Scores document relevance against a query.

    Attributes:
        model_name: Stable identifier for this reranking model
                    (e.g., "Qwen/Qwen3-Reranker-0.6B").
    """

    model_name: str

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        """
        Score a list of documents for relevance to a query.

        Args:
            query: The search query string.
            documents: List of document texts to score.

        Returns:
            List of relevance scores, one per input document, in the same order.
            Each score is a float in [0, 1]; higher = more relevant.
        """
        ...


@runtime_checkable
class RecordReranker(Protocol):
    """Optional identity-aware Reranker extension.

    The plain Reranker port only ever sees document text, so an
    implementation that could reuse a record's already-computed data (e.g. a
    stored embedding) has no way to know which record a document came from.
    A Reranker that also implements this protocol receives full Records
    instead, and callers (the search pipeline, CascadingReranker) prefer this
    method when it is available, falling back to plain text otherwise.
    """

    def rerank_records(self, query: str, records: list[Record]) -> list[float]:
        """
        Score records for relevance to a query.

        Args:
            query: The search query string.
            records: Records to score.

        Returns:
            List of relevance scores, one per input record, in the same order.
            Each score is a float in [0, 1]; higher = more relevant.
        """
        ...
