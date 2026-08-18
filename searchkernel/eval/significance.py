"""Paired significance testing for retrieval evaluation comparisons.

Two aggregate metric numbers never settle whether a candidate configuration
is genuinely better than a baseline or the delta is noise. This module
answers that with a paired sign-flip permutation test plus a paired Cohen's
d effect size, computed over the *same* queries evaluated by both
configurations (see :mod:`searchkernel.eval.runner` for how to obtain
per-query scores from an :class:`~searchkernel.eval.runner.EvalReport`).

Pure functions, no I/O. Requires numpy; deliberately avoids scipy.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class PairedComparison:
    """Whether one configuration genuinely beat another.

    ``p_value`` comes from a two-sided sign-flip permutation test over the
    per-query differences. ``effect_size`` is a paired Cohen's d
    (``mean_difference / std(differences)``): a small p-value on a trivial
    effect size means "real but doesn't matter." ``ci_low``/``ci_high`` are
    a percentile bootstrap confidence interval on the mean difference,
    computed with the same iteration count and RNG stream as the
    permutation test for determinism.
    """

    mean_difference: float
    p_value: float
    effect_size: float
    n_queries: int
    ci_low: float
    ci_high: float
    confidence: float
    iterations: int


def compare_paired(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    iterations: int = 10_000,
    seed: int = 0,
    confidence: float = 0.95,
) -> PairedComparison:
    """Test whether ``candidate`` beats ``baseline`` on paired per-query scores.

    ``baseline[i]`` and ``candidate[i]`` must be the same query's score under
    the two configurations; this is a paired test, not an independent-samples
    test, because throwing away the pairing loses most of the statistical
    power a same-queries comparison gives you for free.

    Method: a two-sided sign-flip permutation test. Under the null
    hypothesis that baseline and candidate are exchangeable per query, the
    sign of each per-query difference is equally likely to be positive or
    negative. We resample the null distribution of the mean difference by
    flipping each per-query difference's sign uniformly at random
    ``iterations`` times, and measure how often that null mean is at least
    as extreme (in absolute value) as the one actually observed. This
    respects the pairing directly and needs scipy for nothing. A bootstrap
    over the per-query differences would also be a legitimate choice here;
    sign-flip permutation was chosen because it tests the natural null
    hypothesis for a paired A/B comparison (no systematic directional
    effect) without assuming the difference distribution is symmetric
    enough for a bootstrap CI alone to double as a hypothesis test.

    Determinism: uses ``np.random.default_rng(seed)`` exclusively, never the
    global numpy random state, so the same seed always returns the same
    result.

    Degenerate cases (see body for exact handling):
      - empty input: raises ``ValueError`` (there is nothing to compare).
      - length mismatch: raises ``ValueError`` (pairing is impossible).
      - all differences exactly zero: falls out of the same permutation
        math naturally, giving ``p_value == 1.0`` and ``effect_size == 0.0``
        (no special-cased branch needed).
      - a single query: only two sign-flip outcomes exist and both have the
        same absolute value as the observation, so ``p_value`` is always
        1.0 -- correctly reflecting that one query can never establish
        significance. ``effect_size`` is 0.0: a single point has no
        estimable spread (not the same as a *known* zero spread), so there
        is not enough data for a standardized effect size.
      - zero variance in the differences (every query moved by the exact
        same nonzero amount): Cohen's d divides by zero standard deviation.
        If the mean is also zero, ``effect_size`` is 0.0. Otherwise the
        standardized effect is genuinely unbounded, so ``effect_size`` is
        signed infinity rather than a fabricated finite number.

    Args:
        baseline: Per-query scores for the baseline configuration.
        candidate: Per-query scores for the candidate configuration, aligned
            index-for-index with ``baseline`` on the same queries.
        iterations: Number of sign-flip resamples (and bootstrap resamples
            for the confidence interval).
        seed: Seed for the deterministic RNG stream.
        confidence: Confidence level for the bootstrap interval on the mean
            difference (e.g. 0.95 for a 95% CI).

    Returns:
        A :class:`PairedComparison`.

    Raises:
        ValueError: If ``baseline`` and ``candidate`` differ in length, or
            both are empty.
    """
    if len(baseline) != len(candidate):
        raise ValueError(
            "baseline and candidate must be paired on the same queries: "
            f"got {len(baseline)} vs {len(candidate)} scores"
        )
    n = len(baseline)
    if n == 0:
        raise ValueError("cannot run a paired comparison over zero queries")

    diffs = np.asarray(candidate, dtype=np.float64) - np.asarray(baseline, dtype=np.float64)
    mean_diff = float(diffs.mean())

    rng = np.random.default_rng(seed)

    signs = rng.choice(np.array([-1.0, 1.0]), size=(iterations, n))
    permuted_means = (signs * diffs).mean(axis=1)
    observed = abs(mean_diff)
    # Tiny tolerance guards against float round-trip noise at the boundary.
    extreme_count = int(np.sum(np.abs(permuted_means) >= observed - 1e-12))
    # Add-one smoothing (North et al. 2002): a finite number of resamples
    # can never justify a p-value of exactly 0.
    p_value = (extreme_count + 1) / (iterations + 1)

    if n == 1:
        # A single point has no estimable spread at all (sample variance
        # with ddof=1 is literally 0/0, not zero) -- there just isn't enough
        # data for a standardized effect size, so report 0.0 rather than
        # treat "unknown" the same as "unbounded".
        effect_size = 0.0
    else:
        std_diff = float(diffs.std(ddof=1))
        if std_diff == 0.0:
            effect_size = 0.0 if mean_diff == 0.0 else math.copysign(math.inf, mean_diff)
        else:
            effect_size = mean_diff / std_diff

    # Percentile bootstrap CI for the mean difference, drawn from the same
    # rng stream so the whole result is reproducible from one seed.
    boot_indices = rng.integers(0, n, size=(iterations, n))
    boot_means = diffs[boot_indices].mean(axis=1)
    alpha = 1.0 - confidence
    ci_low = float(np.quantile(boot_means, alpha / 2))
    ci_high = float(np.quantile(boot_means, 1 - alpha / 2))

    return PairedComparison(
        mean_difference=mean_diff,
        p_value=float(p_value),
        effect_size=effect_size,
        n_queries=n,
        ci_low=ci_low,
        ci_high=ci_high,
        confidence=confidence,
        iterations=iterations,
    )
