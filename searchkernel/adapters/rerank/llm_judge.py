"""LLM-judge reranker: an ADDITIVE Reranker implementation backed by any
text-completion callable, not a specific provider or API. Mirrors
HuggingFaceReranker's yes/no relevance-judgment approach, but the model call
itself is supplied by the caller as a plain prompt-in/text-out function, so no
LLM client library is a dependency here.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial

_ANSWER_PUNCTUATION = ".,;:!?\"'*`)]}"


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
            "You judge whether a document is relevant to a search query.\n\n"
            f"Query: {query}\n\n"
            f"Document: {document}\n\n"
            "Is this document relevant to the query? Answer with exactly "
            "one word, 'Yes' or 'No'."
        )

    @staticmethod
    def _parse_score(response: str) -> float:
        # Models answer "Yes." or "Yes, this is relevant" however plainly the
        # prompt asks for one word, and one stray comma would otherwise abort
        # the whole rerank, since every document is judged separately.
        stripped = response.strip()
        first_word = stripped.split()[0] if stripped else ""
        answer = first_word.strip(_ANSWER_PUNCTUATION).casefold()
        if answer == "yes":
            return 1.0
        if answer == "no":
            return 0.0
        raise ValueError(f"judge reranker got an unparseable response: {response!r}")
