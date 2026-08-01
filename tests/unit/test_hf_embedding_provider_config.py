from __future__ import annotations

from typing import ClassVar
from unittest.mock import patch

import numpy as np
import pytest

from searchkernel.adapters.embedding import HuggingFaceEmbeddingProvider


class _FakeModel:
    prompts: ClassVar[dict[str, str]] = {}

    def __init__(self, *args, **kwargs) -> None:
        self.calls: list[dict[str, object]] = []

    def get_embedding_dimension(self) -> int:
        return 1

    def encode(self, texts, **kwargs):
        self.calls.append(kwargs)
        return np.ones((len(texts), 1))


def test_embedding_batch_size_is_used_for_documents_and_queries() -> None:
    with patch("sentence_transformers.SentenceTransformer", _FakeModel):
        provider = HuggingFaceEmbeddingProvider(batch_size=4)

    provider.embed(["document"])
    provider.embed_queries(["query"])

    assert provider._model.calls == [
        {
            "batch_size": 4,
            "normalize_embeddings": True,
            "convert_to_numpy": True,
        },
        {
            "prompt": "Instruct: Given a web search query, retrieve relevant passages\nQuery: ",
            "batch_size": 4,
            "normalize_embeddings": True,
            "convert_to_numpy": True,
        },
    ]


def test_embedding_batch_size_must_be_positive() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        HuggingFaceEmbeddingProvider(batch_size=0)
