import pytest

from searchkernel.adapters.rerank.cross_encoder import (
    CrossEncoderReranker,
    sentence_transformers_cross_encoder,
)
from searchkernel.ports.rerank import Reranker


def _fake_scorer() -> tuple[CrossEncoderReranker, list[tuple[str, list[str]]]]:
    calls: list[tuple[str, list[str]]] = []

    def score(query: str, documents: list[str]) -> list[float]:
        calls.append((query, documents))
        return [float(len(doc)) / 100 for doc in documents]

    return CrossEncoderReranker(score, model_name="fake-cross-encoder"), calls


def test_implements_reranker_protocol() -> None:
    reranker, _ = _fake_scorer()
    assert isinstance(reranker, Reranker)


def test_model_name_is_the_caller_supplied_name() -> None:
    reranker, _ = _fake_scorer()
    assert reranker.model_name == "fake-cross-encoder"


def test_empty_documents_returns_empty_list_without_calling_scorer() -> None:
    reranker, calls = _fake_scorer()

    scores = reranker.rerank("query", [])

    assert scores == []
    assert calls == []


def test_scores_pass_through_in_input_order() -> None:
    reranker, calls = _fake_scorer()

    scores = reranker.rerank("query", ["ab", "abcd", "a"])

    assert scores == [0.02, 0.04, 0.01]
    assert calls == [("query", ["ab", "abcd", "a"])]


class TestSentenceTransformersCrossEncoderMemoization:
    def test_same_model_name_loads_the_model_only_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sentence_transformers_cross_encoder.cache_clear()
        load_count = 0

        class _FakeCrossEncoder:
            def __init__(self, model_name: str, **_kwargs: object) -> None:
                nonlocal load_count
                load_count += 1

            def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
                return [0.5 for _ in pairs]

        monkeypatch.setattr(
            "sentence_transformers.CrossEncoder", _FakeCrossEncoder
        )

        sentence_transformers_cross_encoder("fake/memo-model")
        sentence_transformers_cross_encoder("fake/memo-model")

        assert load_count == 1
        sentence_transformers_cross_encoder.cache_clear()


class TestSentenceTransformersCrossEncoderRealModel:
    """Real-model tests: build the reranker on the actual bge-reranker-v2-m3
    weights and confirm the [0, 1] score contract and ordering hold end to
    end, not just against a fake scorer.
    """

    @pytest.fixture
    def reranker(self) -> CrossEncoderReranker:
        sentence_transformers_cross_encoder.cache_clear()
        return CrossEncoderReranker(
            sentence_transformers_cross_encoder(),
            model_name="BAAI/bge-reranker-v2-m3",
        )

    @pytest.mark.slow
    @pytest.mark.real_embeddings
    def test_rerank_scores_in_valid_range(self, reranker: CrossEncoderReranker) -> None:
        query = "What is Python?"
        documents = ["Python is a programming language.", "A snake is a reptile."]
        scores = reranker.rerank(query, documents)

        assert all(0.0 <= s <= 1.0 for s in scores), f"Scores out of range: {scores}"

    @pytest.mark.slow
    @pytest.mark.real_embeddings
    def test_rerank_relevant_higher_than_irrelevant(
        self, reranker: CrossEncoderReranker
    ) -> None:
        query = "What is Python programming language?"
        relevant = (
            "Python is a high-level, interpreted programming language "
            "created by Guido van Rossum."
        )
        irrelevant = "A snake is a legless reptile found in many parts of the world."

        scores = reranker.rerank(query, [relevant, irrelevant])
        relevant_score, irrelevant_score = scores

        assert (
            relevant_score > irrelevant_score
        ), f"Expected relevant ({relevant_score}) > irrelevant ({irrelevant_score})"

    @pytest.mark.slow
    @pytest.mark.real_embeddings
    def test_rerank_maintains_order(self, reranker: CrossEncoderReranker) -> None:
        query = "machine learning"
        documents = [
            "Machine learning is a subset of AI.",
            "Cats are adorable pets.",
            "Deep learning uses neural networks.",
        ]
        scores = reranker.rerank(query, documents)

        assert len(scores) == 3
        assert scores[0] > scores[1]  # ML > Cats
        assert scores[2] > scores[1]  # Deep learning > Cats
