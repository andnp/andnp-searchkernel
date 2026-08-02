"""Focused batching tests for the HuggingFace reranker adapter."""

import math

import pytest
import torch

from searchkernel.adapters.rerank import HuggingFaceReranker


class FakeTokenizedInputs(dict[str, torch.Tensor]):
    def to(self, device: str, /) -> "FakeTokenizedInputs":
        return self


class FakeTokenizer:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(
        self,
        texts: list[str],
        *,
        return_tensors: str,
        padding: bool,
        truncation: bool,
        max_length: int,
    ) -> FakeTokenizedInputs:
        del return_tensors, padding, truncation, max_length
        self.calls.append(texts)
        lengths = torch.tensor([len(text) % 4 + 1 for text in texts])
        max_length = int(lengths.max().item())
        input_ids = torch.zeros((len(texts), max_length), dtype=torch.long)
        for index, text in enumerate(texts):
            input_ids[index, : lengths[index]] = len(text) % 10
        attention_mask = torch.arange(max_length).expand(len(texts), -1) < lengths[:, None]
        return FakeTokenizedInputs(input_ids=input_ids, attention_mask=attention_mask)


class FakeModel:
    def __init__(self, yes_token_id: int, no_token_id: int) -> None:
        self.yes_token_id = yes_token_id
        self.no_token_id = no_token_id
        self.batch_sizes: list[int] = []

    def __call__(self, **inputs: object) -> object:
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
        return type("FakeOutput", (), {"logits": logits})()


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
