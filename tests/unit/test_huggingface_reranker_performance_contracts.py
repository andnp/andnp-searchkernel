"""Behavior contracts for configurable HuggingFace reranker batching."""

from __future__ import annotations

import sys
from collections.abc import ItemsView
from types import SimpleNamespace

import pytest
import torch

from searchkernel.adapters.rerank import HuggingFaceReranker


class _FakeTokenizedInputs:
    def __init__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> None:
        self._values: dict[str, object] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def items(self) -> ItemsView[str, object]:
        return self._values.items()

    def __getitem__(self, key: str) -> object:
        return self._values[key]

    def to(self, device: str, /) -> _FakeTokenizedInputs:
        del device
        return self


class _FakeTokenizer:
    def __call__(
        self,
        text: str | list[str],
        *,
        return_tensors: str,
        padding: bool,
        truncation: bool,
        max_length: int,
    ) -> _FakeTokenizedInputs:
        del return_tensors, padding, truncation, max_length
        prompts = [text] if isinstance(text, str) else text
        lengths = torch.tensor([len(prompt) % 3 + 1 for prompt in prompts])
        width = int(lengths.max().item())
        input_ids = torch.zeros((len(prompts), width), dtype=torch.long)
        attention_mask = torch.arange(width).expand(len(prompts), -1) < lengths[:, None]
        return _FakeTokenizedInputs(input_ids, attention_mask)

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        del add_special_tokens
        return [1 if text == "Yes" else 2]


class _FakeModel:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def __call__(self, **inputs: object) -> SimpleNamespace:
        input_ids = inputs["input_ids"]
        assert isinstance(input_ids, torch.Tensor)
        self.batch_sizes.append(input_ids.shape[0])
        logits = torch.zeros((*input_ids.shape, 3))
        logits[:, :, 1] = 1
        return SimpleNamespace(logits=logits)

    def to(self, device: str, /) -> _FakeModel:
        del device
        return self

    def eval(self) -> _FakeModel:
        return self


def _make_reranker(
    monkeypatch: pytest.MonkeyPatch, *, batch_size: int = 8
) -> tuple[HuggingFaceReranker, _FakeModel]:
    tokenizer = _FakeTokenizer()
    model = _FakeModel()
    fake_transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda _name: tokenizer),
        AutoModelForCausalLM=SimpleNamespace(
            from_pretrained=lambda _name, *, torch_dtype: model
        ),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    return HuggingFaceReranker(batch_size=batch_size), model


def test_default_batch_size_preserves_current_batching_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default still forwards eight documents per inference batch."""
    reranker, model = _make_reranker(monkeypatch)

    scores = reranker.rerank("query", [f"document {index}" for index in range(9)])

    assert model.batch_sizes == [8, 1]
    assert scores == [0.7310585975646973] * 9


def test_configured_batch_size_preserves_order_and_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A configured batch size changes grouping without changing results."""
    configured, configured_model = _make_reranker(monkeypatch, batch_size=3)
    default, _ = _make_reranker(monkeypatch)
    documents = [f"document {index}" for index in range(7)]

    configured_scores = configured.rerank("query", documents)
    default_scores = default.rerank("query", documents)

    assert configured_model.batch_sizes == [3, 3, 1]
    assert configured_scores == default_scores


def test_empty_input_returns_without_model_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty document list returns no scores and makes no model call."""
    reranker, model = _make_reranker(monkeypatch, batch_size=2)

    assert reranker.rerank("query", []) == []
    assert model.batch_sizes == []


def test_short_input_uses_one_partial_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fewer documents than the batch size are forwarded as one short batch."""
    reranker, model = _make_reranker(monkeypatch, batch_size=4)

    scores = reranker.rerank("query", ["first", "second"])

    assert model.batch_sizes == [2]
    assert len(scores) == 2


@pytest.mark.parametrize("batch_size", [0, -1])
def test_batch_size_must_be_positive(batch_size: int) -> None:
    """Non-positive batch sizes are rejected before model loading."""
    with pytest.raises(ValueError, match="batch_size must be positive"):
        HuggingFaceReranker(batch_size=batch_size)
