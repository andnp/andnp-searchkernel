"""Keyword-scoring port: optional identifier-aware boosting for keyword search.

Plain BM25 keyword search treats every query as natural-language text. Some
corpora contain identifier-shaped queries -- paths, filenames, dotted
tokens -- where a source knows how to detect that shape and reward rows
that match it structurally. That knowledge is source-specific (filesystem
layout, markdown headers, etc.), so it is expressed here as an optional
capability rather than built into the generic keyword store.
"""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class KeywordArtifactScorer(Protocol):
    """Detects identifier-shaped queries and scores rows against them."""

    def looks_like_identifier_query(self, query: str) -> bool:
        """Whether the whole query reads as an identifier/path rather than natural language.

        Governs whether keyword search overfetches candidates and reranks
        them at all.
        """
        ...

    def identifier_tokens(self, query: str) -> Sequence[str]:
        """Identifier-shaped sub-tokens embedded in a (possibly multi-word) query.

        Used to focus the keyword search on those sub-tokens. Can be empty
        even when `looks_like_identifier_query` is True for the query as a
        whole -- the two questions are asked for different reasons and are
        not guaranteed to agree.
        """
        ...

    def score(
        self,
        query: str,
        *,
        title: str,
        body: str,
        indexed_text: str | None,
        headers: str,
        uri: str,
    ) -> float:
        """Additional boost to add on top of a row's base relevance score.

        ``body`` and ``indexed_text`` are both supplied because a record may
        carry text prepared for indexing that differs from its body, and an
        implementation may reasonably want either: matching a whole document
        is a different question from matching the span that was indexed.
        Collapsing them here would force that choice on every implementation.

        Return 0.0 for no boost.
        """
        ...
