import threading

import pytest

from searchkernel.adapters.rerank.llm_judge import LLMJudgeReranker
from searchkernel.ports.rerank import Reranker


def _judge(responses_by_document: dict[str, str]) -> tuple[LLMJudgeReranker, list[str]]:
    """Maps responses by document content, not call order -- scoring is concurrent."""
    prompts: list[str] = []
    lock = threading.Lock()

    def complete(prompt: str) -> str:
        with lock:
            prompts.append(prompt)
        document = next(doc for doc in responses_by_document if doc in prompt)
        return responses_by_document[document]

    return LLMJudgeReranker(complete, model_name="fake-judge"), prompts


def test_implements_reranker_protocol() -> None:
    reranker, _ = _judge({"doc": "Yes"})
    assert isinstance(reranker, Reranker)


def test_model_name_is_the_caller_supplied_name() -> None:
    reranker, _ = _judge({})
    assert reranker.model_name == "fake-judge"


def test_scores_preserve_input_document_order() -> None:
    reranker, prompts = _judge({"cats document": "Yes", "dogs document": "No"})

    scores = reranker.rerank("query", ["cats document", "dogs document"])

    assert scores == [1.0, 0.0]
    assert len(prompts) == 2
    assert any("cats document" in prompt for prompt in prompts)
    assert any("dogs document" in prompt for prompt in prompts)


def test_rerank_judges_documents_concurrently() -> None:
    """Two documents in flight at once, not one-at-a-time -- the whole point
    of judging concurrently is to avoid paying N sequential completion calls.
    """
    barrier = threading.Barrier(2, timeout=2.0)

    def complete(_prompt: str) -> str:
        barrier.wait()
        return "Yes"

    reranker = LLMJudgeReranker(complete, model_name="fake-judge")

    scores = reranker.rerank("query", ["doc-a", "doc-b"])

    assert scores == [1.0, 1.0]


def test_answer_parsing_is_case_insensitive_and_trims_whitespace() -> None:
    reranker, _ = _judge({"doc": "  yES \n"})
    assert reranker.rerank("query", ["doc"]) == [1.0]


def test_unparseable_response_raises() -> None:
    reranker, _ = _judge({"doc": "I'm not sure"})
    with pytest.raises(ValueError, match="unparseable"):
        reranker.rerank("query", ["doc"])


def test_empty_response_raises() -> None:
    reranker, _ = _judge({"doc": ""})
    with pytest.raises(ValueError, match="unparseable"):
        reranker.rerank("query", ["doc"])


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("Yes.", 1.0),
        ("Yes,", 1.0),
        ("Yes, this is relevant.", 1.0),
        ("**Yes**", 1.0),
        ("No.", 0.0),
        ("no!", 0.0),
    ],
)
def test_judge_reranker_tolerates_punctuated_answers(
    response: str, expected: float
) -> None:
    """Models punctuate however plainly the prompt asks for one word."""
    reranker = LLMJudgeReranker(lambda _prompt: response, model_name="test")

    assert reranker.rerank("query", ["document"]) == [expected]
