"""Tests for deterministic benchmark corpora."""

import pytest

from searchkernel.eval.synthetic import (
    make_1k_corpus,
    make_10k_corpus,
    make_100k_corpus,
)


def test_make_1k_corpus_is_deterministic() -> None:
    """The routine corpus has stable records, queries, and metadata."""
    first = make_1k_corpus(seed=3)
    second = make_1k_corpus(seed=3)

    assert len(first) == 1_000
    assert first.version == second.version
    assert [record.to_dict() for record in first.records] == [
        record.to_dict() for record in second.records
    ]
    assert first.golden_set.to_dict() == second.golden_set.to_dict()
    assert first.golden_set.entries[0].corpus_version == first.version


@pytest.mark.slow
@pytest.mark.parametrize(
    ("factory", "expected_size"),
    [(make_10k_corpus, 10_000), (make_100k_corpus, 100_000)],
)
def test_large_corpora_have_expected_sizes(factory, expected_size: int) -> None:
    """Medium and large helpers are available for explicit benchmark runs."""
    assert len(factory()) == expected_size
