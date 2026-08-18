import pytest

from searchkernel.adapters.rerank.llm_judge import LLMJudgeReranker
from searchkernel.ports.rerank import Reranker


def _judge(responses: list[str]) -> tuple[LLMJudgeReranker, list[str]]:
    prompts: list[str] = []
    responses_iter = iter(responses)

    def complete(prompt: str) -> str:
        prompts.append(prompt)
        return next(responses_iter)

    return LLMJudgeReranker(complete, model_name="fake-judge"), prompts


def test_implements_reranker_protocol() -> None:
    reranker, _ = _judge(["Yes"])
    assert isinstance(reranker, Reranker)


def test_model_name_is_the_caller_supplied_name() -> None:
    reranker, _ = _judge([])
    assert reranker.model_name == "fake-judge"


def test_scores_one_document_per_call_in_order() -> None:
    reranker, prompts = _judge(["Yes", "No"])

    scores = reranker.rerank("query", ["relevant doc", "irrelevant doc"])

    assert scores == [1.0, 0.0]
    assert len(prompts) == 2
    assert "relevant doc" in prompts[0]
    assert "irrelevant doc" in prompts[1]


def test_answer_parsing_is_case_insensitive_and_trims_whitespace() -> None:
    reranker, _ = _judge(["  yES \n"])
    assert reranker.rerank("query", ["doc"]) == [1.0]


def test_unparseable_response_raises() -> None:
    reranker, _ = _judge(["I'm not sure"])
    with pytest.raises(ValueError, match="unparseable"):
        reranker.rerank("query", ["doc"])


def test_empty_response_raises() -> None:
    reranker, _ = _judge([""])
    with pytest.raises(ValueError, match="unparseable"):
        reranker.rerank("query", ["doc"])
