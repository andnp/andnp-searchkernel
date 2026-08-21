"""Focused batching tests for the HuggingFace reranker adapter."""

import math
from collections.abc import ItemsView

import pytest
import torch

from searchkernel.adapters.rerank import HuggingFaceReranker


class FakeTokenizedInputs:
    def __init__(self, **values: torch.Tensor) -> None:
        self.values: dict[str, object] = {
            key: value for key, value in values.items()
        }

    def items(self) -> ItemsView[str, object]:
        return self.values.items()

    def __getitem__(self, key: str) -> object:
        return self.values[key]

    def to(self, device: str, /) -> "FakeTokenizedInputs":
        del device
        return self


class FakeTokenizer:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(
        self,
        text: str | list[str],
        *,
        return_tensors: str,
        padding: bool,
        truncation: bool,
        max_length: int,
    ) -> FakeTokenizedInputs:
        del return_tensors, padding, truncation, max_length
        text_batch = [text] if isinstance(text, str) else text
        self.calls.append(text_batch)
        lengths = torch.tensor([len(text) % 4 + 1 for text in text_batch])
        max_length = int(lengths.max().item())
        input_ids = torch.zeros((len(text_batch), max_length), dtype=torch.long)
        for index, document in enumerate(text_batch):
            input_ids[index, : lengths[index]] = len(document) % 10
        attention_mask = torch.arange(max_length).expand(len(text_batch), -1) < lengths[:, None]
        return FakeTokenizedInputs(input_ids=input_ids, attention_mask=attention_mask)

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        del text, add_special_tokens
        return [1]


class FakeModelOutput:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


class FakeModel:
    def __init__(self, yes_token_id: int, no_token_id: int) -> None:
        self.yes_token_id = yes_token_id
        self.no_token_id = no_token_id
        self.batch_sizes: list[int] = []

    def __call__(self, **inputs: object) -> FakeModelOutput:
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        assert isinstance(input_ids, torch.Tensor)
        assert isinstance(attention_mask, torch.Tensor)
        self.batch_sizes.append(input_ids.shape[0])
        logits = torch.zeros((*input_ids.shape, 3))
        final_positions = attention_mask.sum(dim=1) - 1
        row_indices = torch.arange(input_ids.shape[0])
        final_values = input_ids[row_indices, final_positions].float()
        logits[:, :, self.yes_token_id] = final_values[:, None]
        logits[:, :, self.no_token_id] = 0
        return FakeModelOutput(logits)

    def to(self, device: str, /) -> "FakeModel":
        del device
        return self

    def eval(self) -> "FakeModel":
        return self


def make_reranker(tokenizer: FakeTokenizer, model: FakeModel) -> HuggingFaceReranker:
    reranker = object.__new__(HuggingFaceReranker)
    reranker._tokenizer = tokenizer
    reranker._model = model
    reranker._device = "cpu"
    reranker._yes_token_id = model.yes_token_id
    reranker._no_token_id = model.no_token_id
    return reranker


def test_rerank_tokenizes_and_forwards_each_configured_batch() -> None:
    tokenizer = FakeTokenizer()
    model = FakeModel(yes_token_id=1, no_token_id=2)
    reranker = make_reranker(tokenizer, model)
    documents = [f"document {index}" for index in range(9)]

    scores = reranker.rerank("query", documents)

    assert [len(call) for call in tokenizer.calls] == [8, 1]
    assert model.batch_sizes == [8, 1]
    assert len(scores) == len(documents)
    assert scores == reranker.rerank("query", documents)


def test_rerank_uses_each_document_final_token_for_stable_scores() -> None:
    tokenizer = FakeTokenizer()
    model = FakeModel(yes_token_id=1, no_token_id=2)
    reranker = make_reranker(tokenizer, model)
    documents = ["a", "long document"]

    scores = reranker.rerank("query", documents)

    assert scores == pytest.approx([
        1 / (1 + math.exp(-(len(reranker._build_prompt("query", document)) % 10)))
        for document in documents
    ])
