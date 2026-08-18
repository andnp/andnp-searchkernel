"""Isotonic lane calibration: fit P(relevant | raw_score) from labelled data.

``lane_confidence`` maps each lane's raw score onto ``[0, 1]`` with a fixed
parametric transfer function chosen by inspection -- cosine rescaled, lexical
saturated, graph clamped. That is a large improvement over comparing raw
scores from incompatible lanes against the same absolute threshold, but the
curve shape is still a guess. Given labelled ``(raw_score, relevant)`` pairs,
pool-adjacent-violators (PAVA) fits the actual monotone step function that
best maps raw score to ``P(relevant | score)``, in pure numpy -- scipy is not
a dependency of this project and must not become one.

This module is additive: it does not touch ``lane_confidence`` or any
existing Protocol, so a consumer without labels sees no change at all.
:func:`calibrated_lane_confidence` is the seam a consumer swaps in once they
have enough labelled data for a lane, with a ``lane_confidence`` fallback for
every lane that lacks one.

Plateau problem and the interpolation choice
----------------------------------------------
PAVA is non-decreasing by construction, so it can never invert two scores --
but it is piecewise constant, so it *ties* raw scores that were previously
distinct. Within a single lane a tie is harmless (it does not change which
lane wins). Across lanes it is not: a flat plateau in one lane hands every
tie-break inside that score range to whichever other lane is being fused
against it, quietly shifting the fusion balance in a way that has nothing to
do with relevance.

This module resolves the plateau by **linear interpolation between knot
midpoints** rather than leaving the curve piecewise constant. Each PAVA block
(a maximal run of pooled points) is collapsed to one knot placed at the
midpoint of the block's raw-score range, holding the block's average label
rate. Adjacent knot values are non-decreasing (by the PAVA invariant), so
linear interpolation between them is itself non-decreasing -- monotonicity is
preserved -- while distinct raw scores inside a former plateau now map to
distinct (if close) confidences instead of an exact tie. Below the first knot
or above the last, the curve is flat at the boundary value, since PAVA offers
no evidence outside the observed range.

Degenerate cases
-----------------
- **All-positive or all-negative labels**: the minority class count is zero,
  which is below ``minimum_samples`` for any realistic gate, so
  :func:`fit_isotonic` returns ``None`` before a curve is ever built -- there
  is no discrimination to fit. (Passing ``minimum_samples=0`` explicitly opts
  out of the gate; in that case the fit proceeds and correctly collapses to a
  single knot at the shared label value.)
- **Duplicate raw scores with conflicting labels**: pre-aggregated by
  (weighted) mean before PAVA runs, so a repeated raw score contributes its
  observed relevance rate rather than being pooled arbitrarily by sort order.
- **Non-finite scores**: rejected with ``ValueError``, matching
  ``lane_confidence``'s contract for raw scores.
- **Length mismatch between scores and labels**: rejected with
  ``ValueError``, matching the pairing contract used elsewhere in this
  codebase (see :func:`searchkernel.eval.significance.compare_paired`).

Determinism: no RNG anywhere. The same ``(scores, labels)`` always produce
the same knots.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from searchkernel.search.lane_confidence import lane_confidence

__all__ = [
    "IsotonicCurve",
    "calibrated_lane_confidence",
    "fit_isotonic",
]


@dataclass(frozen=True, slots=True)
class IsotonicCurve:
    """A fitted, monotone raw-score-to-confidence curve.

    ``knot_x``/``knot_y`` are the PAVA block midpoints and their pooled label
    rates, strictly increasing in ``knot_x`` and non-decreasing in
    ``knot_y``. :meth:`confidence` linearly interpolates between them (see
    the module docstring for why interpolation rather than a step function).
    """

    knot_x: tuple[float, ...]
    knot_y: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.knot_x) == 0 or len(self.knot_x) != len(self.knot_y):
            raise ValueError("knot_x and knot_y must be equal-length and non-empty")
        if any(a >= b for a, b in zip(self.knot_x, self.knot_x[1:], strict=False)):
            raise ValueError("knot_x must be strictly increasing")
        if any(a > b for a, b in zip(self.knot_y, self.knot_y[1:], strict=False)):
            raise ValueError("knot_y must be non-decreasing")

    def confidence(self, raw_score: float) -> float:
        """Evaluate the fitted curve at ``raw_score`` via ``np.searchsorted``."""
        if not math.isfinite(raw_score):
            raise ValueError("raw_score must be finite")
        xs = np.asarray(self.knot_x, dtype=np.float64)
        ys = np.asarray(self.knot_y, dtype=np.float64)
        idx = int(np.searchsorted(xs, raw_score))
        if idx == 0:
            return float(ys[0])
        if idx == len(xs):
            return float(ys[-1])
        x0, x1 = xs[idx - 1], xs[idx]
        y0, y1 = ys[idx - 1], ys[idx]
        t = (raw_score - x0) / (x1 - x0)
        return float(y0 + t * (y1 - y0))

    def to_dict(self) -> dict[str, Any]:
        return {"knot_x": list(self.knot_x), "knot_y": list(self.knot_y)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IsotonicCurve:
        return cls(
            knot_x=tuple(float(x) for x in data["knot_x"]),
            knot_y=tuple(float(y) for y in data["knot_y"]),
        )


def fit_isotonic(
    scores: Sequence[float],
    labels: Sequence[bool | int],
    *,
    minimum_samples: int = 50,
) -> IsotonicCurve | None:
    """Fit a monotone raw-score-to-P(relevant) curve via PAVA.

    Returns ``None`` when there is not enough data to trust a fitted curve --
    specifically when the count of positives or negatives falls below
    ``minimum_samples``. A curve fitted on a handful of examples is worse
    than the parametric fallback it would replace, so refusing to fit is the
    correct behaviour, not a missing feature.

    Raises:
        ValueError: if ``scores`` and ``labels`` differ in length, any score
            is non-finite, any label is not boolean-valued (0/1 or
            True/False), or ``minimum_samples`` is negative.
    """
    if len(scores) != len(labels):
        raise ValueError(
            "scores and labels must be paired one-to-one: "
            f"got {len(scores)} vs {len(labels)}"
        )
    if isinstance(minimum_samples, bool) or minimum_samples < 0:
        raise ValueError("minimum_samples must be a non-negative integer")
    if len(scores) == 0:
        return None

    scores_arr = np.asarray(scores, dtype=np.float64)
    if not np.all(np.isfinite(scores_arr)):
        raise ValueError("scores must be finite")

    labels_arr = np.asarray(labels, dtype=np.float64)
    if not np.all(np.isin(labels_arr, (0.0, 1.0))):
        raise ValueError("labels must be boolean-valued (0/1 or True/False)")

    positives = int(labels_arr.sum())
    negatives = len(labels_arr) - positives
    if positives < minimum_samples or negatives < minimum_samples:
        return None

    # Pre-aggregate duplicate raw scores by their (weighted) mean label rate
    # so a repeated score contributes its observed relevance rate to PAVA
    # rather than being pooled arbitrarily by sort order.
    order = np.argsort(scores_arr, kind="stable")
    unique_x, inverse, counts = np.unique(
        scores_arr[order], return_inverse=True, return_counts=True
    )
    sums = np.zeros(len(unique_x), dtype=np.float64)
    np.add.at(sums, inverse, labels_arr[order])
    means = sums / counts

    # Pool-adjacent-violators: sweep points left to right, maintaining a
    # stack of blocks whose (weighted) mean values are non-decreasing.
    # Each block tracks its total weighted value, its total weight, and the
    # [x_min, x_max] range of raw scores it has absorbed.
    stack: list[list[float]] = []
    for x, value, weight in zip(unique_x.tolist(), means.tolist(), counts.tolist(), strict=True):
        stack.append([value * weight, float(weight), x, x])
        while len(stack) >= 2 and (
            stack[-2][0] / stack[-2][1] > stack[-1][0] / stack[-1][1]
        ):
            top = stack.pop()
            below = stack.pop()
            stack.append(
                [below[0] + top[0], below[1] + top[1], below[2], top[3]]
            )

    knot_x = tuple((x_min + x_max) / 2.0 for _, _, x_min, x_max in stack)
    knot_y = tuple(sum_wy / weight for sum_wy, weight, _, _ in stack)
    return IsotonicCurve(knot_x=knot_x, knot_y=knot_y)


def calibrated_lane_confidence(
    curves: Mapping[str, IsotonicCurve],
    *,
    saturation_k: float,
    fallback: Callable[..., float] = lane_confidence,
) -> Callable[[str, float], float]:
    """Build a ``lane_confidence``-shaped callable backed by fitted curves.

    Uses a fitted :class:`IsotonicCurve` for any lane present in ``curves``;
    falls back to ``fallback`` (the existing parametric ``lane_confidence``
    by default) for every other lane. This is the seam a consumer swaps in
    once they have enough labelled data for a lane -- lanes without a curve
    keep behaving exactly as they do today.
    """

    def _confidence(lane: str, raw_score: float) -> float:
        curve = curves.get(lane)
        if curve is not None:
            return curve.confidence(raw_score)
        return fallback(lane, raw_score, saturation_k=saturation_k)

    return _confidence
