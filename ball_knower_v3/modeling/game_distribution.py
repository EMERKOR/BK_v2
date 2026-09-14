"""Betting-relevant discrete predictive distributions for Ball Knower v3.

Continuous game-model draws are never exposed directly to the betting layer.
This module converts posterior predictive Student-t components to integer mass,
tracks residual tail probability explicitly, and prices threshold events with
an exact push state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import t as student_t


@dataclass(frozen=True)
class StudentTMixture:
    """Equal-weight posterior predictive Student-t components."""

    location: np.ndarray
    scale: np.ndarray
    df: np.ndarray

    def __post_init__(self) -> None:
        location = np.asarray(self.location, dtype=float)
        scale = np.asarray(self.scale, dtype=float)
        df = np.asarray(self.df, dtype=float)
        if location.ndim != 1 or scale.shape != location.shape or df.shape != location.shape:
            raise ValueError("location, scale, and df must be equal one-dimensional arrays")
        if location.size == 0:
            raise ValueError("at least one predictive component is required")
        if not np.isfinite(location).all() or not np.isfinite(scale).all() or not np.isfinite(df).all():
            raise ValueError("predictive component parameters must be finite")
        if (scale <= 0.0).any():
            raise ValueError("scale must be positive")
        if (df <= 2.0).any():
            raise ValueError("df must exceed 2")
        object.__setattr__(self, "location", location)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "df", df)

    def cdf(self, x: float | np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        z = (x_arr[..., None] - self.location) / self.scale
        values = student_t.cdf(z, df=self.df)
        return values.mean(axis=-1)


@dataclass(frozen=True)
class DiscretePredictivePMF:
    """Integer PMF with explicit probability outside the represented support.

    ``lower_tail`` and ``upper_tail`` are aggregate probability bins outside the
    represented integer support. Exact CDF values are therefore defined only on
    represented support points; calibration utilities may treat each tail as one
    coarsened category when an observation falls outside support.
    """

    support: np.ndarray
    probabilities: np.ndarray
    lower_tail: float
    upper_tail: float

    def __post_init__(self) -> None:
        support = np.asarray(self.support, dtype=int)
        probabilities = np.asarray(self.probabilities, dtype=float)
        if support.ndim != 1 or probabilities.shape != support.shape or support.size == 0:
            raise ValueError("support and probabilities must be equal non-empty 1D arrays")
        if not np.array_equal(support, np.arange(support[0], support[-1] + 1)):
            raise ValueError("support must be consecutive integers")
        if not np.isfinite(probabilities).all() or (probabilities < -1e-12).any():
            raise ValueError("probabilities must be finite and nonnegative")
        tails = np.array([self.lower_tail, self.upper_tail], dtype=float)
        if not np.isfinite(tails).all() or (tails < -1e-12).any():
            raise ValueError("tail probabilities must be finite and nonnegative")
        total = float(probabilities.sum() + tails.sum())
        if not np.isclose(total, 1.0, atol=1e-9):
            raise ValueError(f"distribution mass must sum to 1; got {total}")
        object.__setattr__(self, "support", support)
        object.__setattr__(self, "probabilities", np.clip(probabilities, 0.0, 1.0))
        object.__setattr__(self, "lower_tail", max(float(self.lower_tail), 0.0))
        object.__setattr__(self, "upper_tail", max(float(self.upper_tail), 0.0))

    def mass_at(self, outcome: int) -> float:
        if outcome < self.support[0] or outcome > self.support[-1]:
            return 0.0
        return float(self.probabilities[outcome - self.support[0]])

    def cdf_at(self, outcome: int) -> float:
        """Return ``P(Y <= outcome)`` for a represented support point.

        Exact CDF values outside support are unknowable from aggregate tail mass,
        so callers must not interpret a tail bin as though its internal integer
        allocation were known.
        """

        if outcome < self.support[0] or outcome > self.support[-1]:
            raise ValueError("exact cdf is unavailable outside represented support")
        index = outcome - self.support[0]
        return float(self.lower_tail + self.probabilities[: index + 1].sum())


@dataclass(frozen=True)
class ThresholdProbabilities:
    below: float
    push: float
    above: float

    def __post_init__(self) -> None:
        values = np.array([self.below, self.push, self.above], dtype=float)
        if not np.isfinite(values).all() or (values < -1e-12).any():
            raise ValueError("threshold probabilities must be finite and nonnegative")
        if not np.isclose(values.sum(), 1.0, atol=1e-9):
            raise ValueError("threshold probabilities must sum to 1")


def discretize_student_t_mixture(
    mixture: StudentTMixture,
    *,
    support_min: int,
    support_max: int,
) -> DiscretePredictivePMF:
    """Bin a continuous posterior mixture into integer outcome mass.

    For integer ``k`` the represented mass is
    ``F(k + 0.5) - F(k - 0.5)``. Probability beyond the finite requested
    support is retained explicitly rather than renormalized away or silently
    assigned to a key number.
    """

    if support_min >= support_max:
        raise ValueError("support_min must be less than support_max")
    support = np.arange(int(support_min), int(support_max) + 1, dtype=int)
    lower_edges = support.astype(float) - 0.5
    upper_edges = support.astype(float) + 0.5
    probabilities = mixture.cdf(upper_edges) - mixture.cdf(lower_edges)
    lower_tail = float(mixture.cdf(float(support_min) - 0.5))
    upper_tail = float(1.0 - mixture.cdf(float(support_max) + 0.5))
    return DiscretePredictivePMF(support, probabilities, lower_tail, upper_tail)


def threshold_probabilities(distribution: DiscretePredictivePMF, line: float) -> ThresholdProbabilities:
    """Return probability below / push / above an actual sportsbook line.

    Half-point lines have zero push probability. Whole-number lines retain the
    exact PMF atom. The line must lie inside represented support so lower and
    upper tail mass can be assigned unambiguously to opposite sides.
    """

    if not np.isfinite(line):
        raise ValueError("line must be finite")
    if line <= distribution.support[0] - 0.5 or line >= distribution.support[-1] + 0.5:
        raise ValueError("line must lie inside represented support")

    integer_line = float(line).is_integer()
    if integer_line:
        k = int(line)
        push = distribution.mass_at(k)
        below_mask = distribution.support < k
        above_mask = distribution.support > k
    else:
        push = 0.0
        below_mask = distribution.support < line
        above_mask = distribution.support > line

    below = float(distribution.lower_tail + distribution.probabilities[below_mask].sum())
    above = float(distribution.upper_tail + distribution.probabilities[above_mask].sum())
    total = below + push + above
    if not np.isclose(total, 1.0, atol=1e-9):
        raise RuntimeError(f"threshold partition does not conserve mass; got {total}")
    return ThresholdProbabilities(below=below, push=push, above=above)


def randomized_pit(
    distribution: DiscretePredictivePMF,
    observed: int,
    *,
    uniform: float,
) -> float:
    """Randomized PIT for an observed discrete outcome.

    ``uniform`` is supplied by the caller so evaluation can use a reproducibly
    seeded random stream. For represented outcome y,
    ``PIT = F(y-) + U * P(Y=y)``.

    If an observation lies outside finite represented support, the corresponding
    aggregate tail is treated as one coarsened category. This preserves a valid
    calibration diagnostic for the stored finite-support forecast without
    pretending the unknown integer allocation inside that tail is available.
    Tail-hit rates should still be reported separately so support can be widened
    if coarsening is material.
    """

    if not 0.0 <= uniform <= 1.0:
        raise ValueError("uniform must be in [0, 1]")
    if observed < distribution.support[0]:
        return float(uniform * distribution.lower_tail)
    if observed > distribution.support[-1]:
        return float(1.0 - distribution.upper_tail + uniform * distribution.upper_tail)
    index = observed - distribution.support[0]
    below = distribution.lower_tail + float(distribution.probabilities[:index].sum())
    return float(below + uniform * distribution.probabilities[index])
