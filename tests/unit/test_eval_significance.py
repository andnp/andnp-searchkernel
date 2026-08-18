"""Unit tests for paired significance testing."""

import math

import pytest

from searchkernel.eval.significance import PairedComparison, compare_paired


class TestCompareePairedValidation:
    """Input validation before any statistics are computed."""

    def test_length_mismatch_raises(self):
        """Un-paired sequences cannot be compared query-for-query."""
        with pytest.raises(ValueError):
            compare_paired([0.1, 0.2, 0.3], [0.1, 0.2])

    def test_empty_input_raises(self):
        """There is nothing to test with zero queries."""
        with pytest.raises(ValueError):
            compare_paired([], [])


class TestCompareePairedKnownAnswers:
    """Cases whose statistical answer is known in advance."""

    def test_constant_uniform_improvement_is_significant(self):
        """Every query improves by the exact same nonzero amount.

        This is a known-answer case: a perfectly consistent improvement
        across many queries must be reported as significant with a large
        positive effect size. If the sign-flip math were wrong (e.g. treating
        the samples as independent, or miscomputing the two-sided extremity
        check), this is the kind of case that would silently pass through as
        "not significant" -- so it directly exercises whether the statistics,
        not just the code path, are correct.
        """
        n = 40
        baseline = [0.5] * n
        candidate = [0.6] * n  # +0.1 on every single query, no variance

        result = compare_paired(baseline, candidate, iterations=5_000, seed=0)

        assert result.mean_difference == pytest.approx(0.1)
        assert result.p_value < 0.01
        assert result.effect_size == math.inf

    def test_identical_inputs_give_p_value_one(self):
        """No difference anywhere means the null hypothesis holds trivially."""
        scores = [0.1, 0.4, 0.7, 0.9, 0.3]

        result = compare_paired(scores, scores, iterations=2_000, seed=0)

        assert result.mean_difference == 0.0
        assert result.p_value == 1.0
        assert result.effect_size == 0.0

    def test_pure_noise_is_not_significant(self):
        """Symmetric random noise around zero should not look significant."""
        import numpy as np

        rng = np.random.default_rng(123)
        baseline = rng.uniform(0.0, 1.0, size=200).tolist()
        noise = rng.normal(loc=0.0, scale=0.05, size=200)
        candidate = (np.asarray(baseline) + noise).tolist()

        result = compare_paired(baseline, candidate, iterations=5_000, seed=1)

        assert result.p_value > 0.05

    def test_clear_improvement_with_variance_is_significant(self):
        """A consistent directional improvement with per-query noise still wins."""
        import numpy as np

        rng = np.random.default_rng(7)
        baseline = rng.uniform(0.3, 0.6, size=150).tolist()
        # Candidate is baseline plus a reliable positive shift with some jitter.
        jitter = rng.normal(loc=0.15, scale=0.03, size=150)
        candidate = (np.asarray(baseline) + jitter).tolist()

        result = compare_paired(baseline, candidate, iterations=5_000, seed=2)

        assert result.mean_difference > 0.1
        assert result.p_value < 0.01
        assert result.effect_size > 0.5


class TestCompareePairedDegenerateCases:
    """Explicit handling for edge inputs that are not simply invalid."""

    def test_single_query_gives_p_value_one(self):
        """One query can never establish statistical significance."""
        result = compare_paired([0.2], [0.9], iterations=1_000, seed=0)

        assert result.n_queries == 1
        assert result.p_value == 1.0
        assert result.effect_size == 0.0

    def test_zero_variance_nonzero_mean_gives_infinite_effect_size(self):
        """Every query shifts by the same nonzero amount: unbounded Cohen's d."""
        result = compare_paired([1.0, 1.0, 1.0], [1.2, 1.2, 1.2], iterations=1_000, seed=0)

        assert result.effect_size == math.inf

    def test_zero_variance_zero_mean_gives_zero_effect_size(self):
        """All differences are exactly zero: p == 1.0 and effect size is 0, not NaN."""
        result = compare_paired([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], iterations=1_000, seed=0)

        assert result.p_value == 1.0
        assert result.effect_size == 0.0
        assert not math.isnan(result.effect_size)


class TestCompareePairedDeterminism:
    """The test must be a usable CI gate: same seed, same result."""

    def test_same_seed_gives_identical_result(self):
        """Repeated calls with the same seed must be byte-for-byte identical."""
        baseline = [0.1, 0.4, 0.55, 0.3, 0.7, 0.2]
        candidate = [0.2, 0.5, 0.5, 0.4, 0.6, 0.3]

        first = compare_paired(baseline, candidate, iterations=3_000, seed=42)
        second = compare_paired(baseline, candidate, iterations=3_000, seed=42)

        assert first == second

    def test_different_seed_keeps_same_conclusion(self):
        """The significance conclusion should be stable across seeds even if
        the exact p-value estimate is not identical."""
        baseline = [0.1] * 60
        candidate = [0.35] * 60

        first = compare_paired(baseline, candidate, iterations=4_000, seed=1)
        second = compare_paired(baseline, candidate, iterations=4_000, seed=2)

        assert first.p_value < 0.05
        assert second.p_value < 0.05


def test_paired_comparison_is_a_frozen_dataclass():
    """The result is meant to be handed around as an immutable record."""
    result = compare_paired([0.1, 0.2], [0.2, 0.3])

    assert isinstance(result, PairedComparison)
    with pytest.raises(AttributeError):
        result.p_value = 0.0  # type: ignore[misc]
