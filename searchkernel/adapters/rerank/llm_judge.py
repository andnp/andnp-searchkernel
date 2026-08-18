"""LLM-judge reranker: an ADDITIVE Reranker implementation backed by any
text-completion callable, not a specific provider or API. The model call
itself is supplied by the caller as a plain prompt-in/text-out function, so no
LLM client library is a dependency here.

Asks for a graded 0-10 relevance rating rather than a yes/no verdict: a
reranker's job is to order documents, and a binary judgment ties every
relevant document at the same score, leaving the pipeline's storage-key
tie-break (effectively alphabetical order) to do the actual ranking. A 0-10
integer scale was chosen over a 0-1 decimal because models produce it more
consistently -- no "0.7" vs "7/10" vs "70%" ambiguity in what's *asked for*,
even though a few of those forms still show up in answers and are parsed
below.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial

_MARKDOWN_EMPHASIS = "*`"
_DECIMAL_RATING = re.compile(r"\d+\.\d+")
_RATING_OF_TEN = re.compile(r"(-?\d{1,3})\s*/\s*10\b")
_BARE_RATING = re.compile(r"-?\d{1,3}\b")
_MAX_RATING = 10


class LLMJudgeReranker:
    """Scores documents by asking an LLM whether each is relevant to the query.

    One completion call per document, so scoring is latency-bound by however
    slow the backing model is. Documents are judged concurrently (bounded by
    max_concurrency) instead of one at a time, since a slow completion
    endpoint would otherwise make judging N documents N times slower than
    judging one.
    """

    def __init__(
        self,
        complete: Callable[[str], str],
        *,
        model_name: str,
        max_concurrency: int = 8,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError(f"max_concurrency must be >= 1, got {max_concurrency}")
        self._complete = complete
        self.model_name = model_name
        self._max_concurrency = max_concurrency

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        workers = min(self._max_concurrency, len(documents))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(partial(self._score, query), documents))

    def _score(self, query: str, document: str) -> float:
        response = self._complete(self._build_prompt(query, document))
        return self._parse_score(response)

    @staticmethod
    def _build_prompt(query: str, document: str) -> str:
        return (
            "You judge how relevant a document is to a search query.\n\n"
            f"Query: {query}\n\n"
            f"Document: {document}\n\n"
            "Rate the relevance on a scale from 0 (completely irrelevant) "
            "to 10 (perfectly relevant). Answer with exactly one integer, "
            "and nothing else."
        )

    @staticmethod
    def _parse_score(response: str) -> float:
        # Models answer "Score: 7", "**7**", or "7/10" however plainly the
        # prompt asks for one integer, and one stray decoration would
        # otherwise abort the whole rerank, since every document is judged
        # separately.
        stripped = response.strip()
        cleaned = stripped.translate(str.maketrans("", "", _MARKDOWN_EMPHASIS))
        # A fractional rating ("7.5") is rejected rather than guessed at: the
        # prompt asks for an integer, so silently truncating to "7" would
        # discard precision the model may have intended differently (e.g.
        # rounding to 8), and there is no documented convention to resolve it.
        if _DECIMAL_RATING.search(cleaned):
            raise ValueError(
                f"judge reranker got an unparseable response: {response!r}"
            )
        match = _RATING_OF_TEN.search(cleaned) or _BARE_RATING.search(cleaned)
        if match is None:
            raise ValueError(
                f"judge reranker got an unparseable response: {response!r}"
            )
        rating = int(match.group(1)) if match.re is _RATING_OF_TEN else int(match.group())
        clamped = max(0, min(_MAX_RATING, rating))
        return clamped / _MAX_RATING
