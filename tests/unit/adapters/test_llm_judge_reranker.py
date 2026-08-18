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
    reranker, _ = _judge({"doc": "7"})
    assert isinstance(reranker, Reranker)


def test_model_name_is_the_caller_supplied_name() -> None:
    reranker, _ = _judge({})
    assert reranker.model_name == "fake-judge"


def test_scores_preserve_input_document_order() -> None:
    reranker, prompts = _judge({"cats document": "9", "dogs document": "2"})

    scores = reranker.rerank("query", ["cats document", "dogs document"])

    assert scores == [0.9, 0.2]
    assert len(prompts) == 2
    assert any("cats document" in prompt for prompt in prompts)
    assert any("dogs document" in prompt for prompt in prompts)


def test_graded_scores_produce_a_genuine_order() -> None:
    """The whole point of a graded score is that documents with different
    ratings sort into the order the ratings imply, not just into distinct
    buckets -- a binary yes/no score can't do this.
    """
    reranker, _ = _judge(
        {
            "low relevance doc": "2",
            "high relevance doc": "9",
            "mid relevance doc": "5",
        }
    )

    scores = reranker.rerank(
        "query", ["low relevance doc", "high relevance doc", "mid relevance doc"]
    )
    ranked = sorted(
        zip(["low relevance doc", "high relevance doc", "mid relevance doc"], scores),
        key=lambda pair: pair[1],
        reverse=True,
    )

    assert [doc for doc, _ in ranked] == [
        "high relevance doc",
        "mid relevance doc",
        "low relevance doc",
    ]


def test_rerank_judges_documents_concurrently() -> None:
    """Two documents in flight at once, not one-at-a-time -- the whole point
    of judging concurrently is to avoid paying N sequential completion calls.
    """
    barrier = threading.Barrier(2, timeout=2.0)

    def complete(_prompt: str) -> str:
        barrier.wait()
        return "10"

    reranker = LLMJudgeReranker(complete, model_name="fake-judge")

    scores = reranker.rerank("query", ["doc-a", "doc-b"])

    assert scores == [1.0, 1.0]


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("7", 0.7),
        ("0", 0.0),
        ("10", 1.0),
        (" 7 \n", 0.7),
        ("7.", 0.7),
        ("7,", 0.7),
        ("7/10", 0.7),
        ("7 / 10", 0.7),
        ("10/10", 1.0),
        ("**7**", 0.7),
        ("`7`", 0.7),
        ("Score: 7", 0.7),
        ("Rating: 7", 0.7),
        ("Score: 7/10", 0.7),
        ("I would rate this a 7 out of 10.", 0.7),
    ],
)
def test_judge_reranker_parses_realistic_response_formats(
    response: str, expected: float
) -> None:
    """Models emit the rating however plainly the prompt asks for a bare
    integer -- decorated with markdown, a denominator, or a label.
    """
    reranker = LLMJudgeReranker(lambda _prompt: response, model_name="test")

    assert reranker.rerank("query", ["document"]) == [expected]


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("11", 1.0),
        ("15", 1.0),
        ("-1", 0.0),
        ("-5", 0.0),
    ],
)
def test_out_of_range_ratings_are_clamped_to_the_valid_scale(
    response: str, expected: float
) -> None:
    reranker = LLMJudgeReranker(lambda _prompt: response, model_name="test")

    assert reranker.rerank("query", ["document"]) == [expected]


@pytest.mark.parametrize("response", ["", "I'm not sure", "maybe", "seven", "n/a"])
def test_unparseable_response_raises(response: str) -> None:
    reranker = LLMJudgeReranker(lambda _prompt: response, model_name="test")

    with pytest.raises(ValueError, match="unparseable"):
        reranker.rerank("query", ["document"])


def test_fractional_rating_raises_rather_than_guessing_rounding() -> None:
    """The prompt asks for an integer; a decimal like '7.5' is ambiguous
    about intended rounding, so it is rejected rather than silently
    truncated.
    """
    reranker = LLMJudgeReranker(lambda _prompt: "7.5", model_name="test")

    with pytest.raises(ValueError, match="unparseable"):
        reranker.rerank("query", ["document"])


@pytest.mark.parametrize("response", ["0", "5", "10"])
def test_scores_are_bounded_to_the_zero_one_range(response: str) -> None:
    reranker = LLMJudgeReranker(lambda _prompt: response, model_name="test")

    (score,) = reranker.rerank("query", ["document"])

    assert 0.0 <= score <= 1.0


@pytest.mark.parametrize("max_concurrency", [0, -1])
def test_construction_rejects_non_positive_max_concurrency(max_concurrency: int) -> None:
    with pytest.raises(ValueError, match="max_concurrency"):
        LLMJudgeReranker(
            lambda _prompt: "5", model_name="test", max_concurrency=max_concurrency
        )
