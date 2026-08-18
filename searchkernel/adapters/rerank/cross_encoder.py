"""Cross-encoder reranker adapter for the Reranker port.

This is an ADDITIVE port implementation. No live path is affected.
"""

from __future__ import annotations

from collections.abc import Callable


class CrossEncoderReranker:
    """Reranker backed by an injected cross-encoder scoring function.

    Unlike LLMJudgeReranker (a causal LM asked to judge yes/no relevance),
    a cross-encoder has a trained classification head and produces a
    relevance score in a single forward pass -- no generation, no
    yes/no-token parsing. The scoring function itself is injected so this
    adapter has no hard dependency on any specific ML library; see
    `sentence_transformers_cross_encoder` in this module for a real one.
    """

    def __init__(
        self, score: Callable[[str, list[str]], list[float]], *, model_name: str
    ) -> None:
        self._score = score
        self.model_name = model_name

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        return self._score(query, documents)
