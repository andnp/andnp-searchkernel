"""Behavioral coverage for deterministic test embeddings."""

import asyncio
import math

import pytest

from searchkernel.embeddings import (
    TEST_FAKE_EMBEDDINGS_ENV_VAR,
    DeterministicFakeEmbeddingModel,
    should_use_test_fake_embeddings,
)


def test_fake_embedding_is_repeatable_for_documents_and_queries() -> None:
    """Repeated calls produce the same document and query vectors.

    Determinism keeps offline tests independent of model state or randomness.
    """
    model = DeterministicFakeEmbeddingModel(dimension=8)

    assert model.get_text_embedding("Alpha beta") == model.get_text_embedding(
        "Alpha beta"
    )
    assert model.get_query_embedding("Alpha beta") == model.get_query_embedding(
        "Alpha beta"
    )


@pytest.mark.parametrize("value", [None, "0", "true", "yes"])
def test_fake_embedding_mode_requires_exact_one_environment_value(
    monkeypatch: pytest.MonkeyPatch,
    value: str | None,
) -> None:
    """Only the literal environment value ``1`` enables fake embeddings.

    Unset and truthy-looking alternatives must not silently select test mode.
    """
    if value is None:
        monkeypatch.delenv(TEST_FAKE_EMBEDDINGS_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(TEST_FAKE_EMBEDDINGS_ENV_VAR, value)

    assert should_use_test_fake_embeddings() is False

    monkeypatch.setenv(TEST_FAKE_EMBEDDINGS_ENV_VAR, "1")
    assert should_use_test_fake_embeddings() is True


@pytest.mark.parametrize("text", ["", "!!! ???", "\t\n"])
def test_empty_and_punctuation_texts_still_have_unit_vectors(text: str) -> None:
    """Texts without word tokens map to a valid normalized vector.

    The fallback token prevents empty input from producing a zero vector.
    """
    vector = DeterministicFakeEmbeddingModel(dimension=6).get_text_embedding(text)

    assert len(vector) == 6
    assert math.isclose(
        math.sqrt(sum(value * value for value in vector)), 1.0, abs_tol=1e-6
    )


def test_custom_dimension_controls_document_and_async_query_shapes() -> None:
    """A custom dimension is honored across synchronous and async APIs.

    The public model methods should agree on shape without loading a model.
    """
    model = DeterministicFakeEmbeddingModel(dimension=3)

    async def embed() -> tuple[list[float], list[float]]:
        return (
            await model.aget_text_embedding("dimension"),
            await model.aget_query_embedding("dimension"),
        )

    text_vector, query_vector = asyncio.run(embed())

    assert len(model.get_text_embedding("dimension")) == 3
    assert len(text_vector) == len(query_vector) == 3


def test_tokenization_normalizes_case_and_ignores_punctuation() -> None:
    """Case and punctuation do not change the token sequence.

    Equivalent token text should produce identical normalized embeddings.
    """
    model = DeterministicFakeEmbeddingModel(dimension=16)

    assert model.get_text_embedding("Alpha, beta!") == model.get_text_embedding(
        "alpha beta"
    )


def test_repeated_tokens_change_weight_while_collisions_remain_normalized() -> None:
    """Repeated tokens accumulate weight even when all buckets collide.

    A one-dimensional model makes the bucket collision deterministic and easy
    to observe while preserving the unit-norm contract.
    """
    model = DeterministicFakeEmbeddingModel(dimension=1)

    once = model.get_text_embedding("alpha")
    repeated = model.get_text_embedding("alpha alpha")
    collided = model.get_text_embedding("alpha beta")

    assert once == repeated == collided == [1.0]
