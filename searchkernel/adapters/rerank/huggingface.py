"""HuggingFace Qwen3-Reranker adapter for the Reranker port.

In-process reranking via Qwen3-Reranker-0.6B, a causal language model
that judges yes/no document relevance to a query.

This is an ADDITIVE port implementation. No live path is affected.
"""

from __future__ import annotations

from collections.abc import ItemsView
from typing import TYPE_CHECKING, Protocol, Self, TypeGuard

if TYPE_CHECKING:
    from torch import Tensor


class _TokenizedInputs(Protocol):
    def items(self) -> ItemsView[str, object]: ...

    def to(self, device: str, /) -> Self: ...


class _Tokenizer(Protocol):
    def __call__(
        self,
        text: str,
        *,
        return_tensors: str,
        padding: bool,
        truncation: bool,
        max_length: int,
    ) -> _TokenizedInputs: ...

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...


class _ModelOutput(Protocol):
    logits: Tensor


class _CausalLM(Protocol):
    def __call__(self, **inputs: object) -> _ModelOutput: ...

    def to(self, device: str, /) -> Self: ...

    def eval(self) -> Self: ...


def _is_tokenizer(value: object) -> TypeGuard[_Tokenizer]:
    return callable(value) and callable(getattr(value, "encode", None))


def _is_causal_lm(value: object) -> TypeGuard[_CausalLM]:
    return all(
        callable(getattr(value, attribute, None)) for attribute in ("__call__", "to", "eval")
    )


class HuggingFaceReranker:
    """Reranker backed by Qwen3-Reranker-0.6B via HuggingFace transformers.

    Qwen3-Reranker is a causal LM used as a yes/no relevance judge.
    For each (query, document) pair:
    1. Build a prompt in the Qwen3-Reranker instruct format
    2. Forward pass; take logits at the final position
    3. Compute softmax over yes/no token ids
    4. Score = P(yes)
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Reranker-0.6B",
        device: str | None = None,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self._device = device or ("cuda" if self._has_cuda() else "cpu")

        # Load tokenizer and model
        tokenizer: object = AutoTokenizer.from_pretrained(model_name)
        model: object = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
        )
        if not _is_tokenizer(tokenizer) or not _is_causal_lm(model):
            raise TypeError("Loaded HuggingFace components do not match the adapter protocol")
        self._tokenizer = tokenizer
        self._model = model.to(self._device)
        self._model.eval()

        # Resolve yes/no token ids via the tokenizer
        # Qwen3-Reranker expects tokenization of single-token yes/no answers
        yes_token_ids = self._tokenizer.encode("Yes", add_special_tokens=False)
        no_token_ids = self._tokenizer.encode("No", add_special_tokens=False)
        if len(yes_token_ids) != 1 or len(no_token_ids) != 1:
            raise ValueError("HuggingFace tokenizer must encode Yes and No as one token each")
        self._yes_token_id = yes_token_ids[0]
        self._no_token_id = no_token_ids[0]

    @staticmethod
    def _has_cuda() -> bool:
        """Check if CUDA is available."""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        """
        Score documents for relevance to a query.

        Args:
            query: The search query string.
            documents: List of document texts to score.

        Returns:
            List of relevance scores in [0, 1], one per document, in order.
            Higher = more relevant.
        """
        scores = []

        # Process in reasonable batches to avoid OOM
        batch_size = 8
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i : i + batch_size]
            batch_scores = self._score_batch(query, batch_docs)
            scores.extend(batch_scores)

        return scores

    def _score_batch(self, query: str, documents: list[str]) -> list[float]:
        """
        Score a batch of documents.

        Args:
            query: The search query.
            documents: Batch of documents to score.

        Returns:
            List of relevance scores in [0, 1].
        """
        import torch

        scores = []

        for doc in documents:
            # Build prompt in Qwen3-Reranker instruct format:
            # System: You are a helpful document evaluator...
            # User: [Query] [Document] Does this document answer the query? Answer Yes or No.
            # Assistant: <yes_or_no>
            prompt = self._build_prompt(query, doc)

            # Tokenize
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self._device)

            # Forward pass; get logits at final position
            with torch.no_grad():
                outputs = self._model(
                    **dict(inputs.items()),
                    output_hidden_states=False,
                )
                logits = outputs.logits[0, -1, :]  # (vocab_size,)

            # Softmax over yes/no tokens
            yes_logit = logits[self._yes_token_id]
            no_logit = logits[self._no_token_id]

            # Compute P(yes)
            exp_yes = torch.exp(yes_logit)
            exp_no = torch.exp(no_logit)
            prob_yes = exp_yes / (exp_yes + exp_no)

            score = prob_yes.item()
            if not isinstance(score, float) or not 0.0 <= score <= 1.0:
                raise RuntimeError("HuggingFace reranker returned a score outside [0, 1]")
            scores.append(score)

        return scores

    def _build_prompt(self, query: str, document: str) -> str:
        """
        Build a prompt in Qwen3-Reranker instruct format.

        Args:
            query: The search query.
            document: The document to evaluate.

        Returns:
            A formatted prompt string.
        """
        # Qwen3-Reranker instruct format:
        # System prompt + user query + document + question
        system_prompt = (
            "You are a helpful document evaluator. "
            "Your task is to determine if a document is relevant to a given query."
        )

        user_prompt = (
            f"Query: {query}\n\n"
            f"Document: {document}\n\n"
            f"Is this document relevant to the query? Answer with 'Yes' or 'No' only."
        )

        # Format as a chat message for the model
        return f"system\n{system_prompt}\nuser\n{user_prompt}\nassistant\n"
