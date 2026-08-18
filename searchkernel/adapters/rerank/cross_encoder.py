"""Cross-encoder reranker adapter for the Reranker port.

This is an ADDITIVE port implementation. No live path is affected.
"""

from __future__ import annotations

import functools
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


@functools.cache
def sentence_transformers_cross_encoder(
    model_name: str = "BAAI/bge-reranker-v2-m3",
) -> Callable[[str, list[str]], list[float]]:
    """Load a real cross-encoder model once and cache it by model_name.

    Memoized because callers may build multiple reranker instances (e.g.
    one runtime per data source) that should all share one loaded model
    rather than each loading their own copy into memory.
    """
    import torch
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_name, activation_fn=torch.nn.Sigmoid())

    def score(query: str, documents: list[str]) -> list[float]:
        pairs = [(query, doc) for doc in documents]
        raw_scores = model.predict(pairs)
        return [float(s) for s in raw_scores]

    return score
