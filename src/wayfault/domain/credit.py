"""Credit term-structure value objects.

Provides :class:`CreditCurve` with interchangeable hazard / survival / marginal
PD representations, and a :class:`RecoveryRate` value object. Two concrete hazard
shapes are supported: flat and piecewise-constant.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wayfault.domain.errors import ValidationError
from wayfault.domain.tenors import TenorGrid


@dataclass(frozen=True)
class RecoveryRate:
    """A recovery rate ``R`` in ``[0, 1)``."""

    value: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.value < 1.0):
            raise ValidationError("RecoveryRate must be in [0, 1).")

    @property
    def loss_given_default(self) -> float:
        """Loss-given-default ``1 - R``."""
        return 1.0 - self.value


@dataclass(frozen=True)
class CreditCurve:
    """A piecewise-constant hazard credit curve.

    A flat curve is the degenerate single-segment case. Internally the curve is
    represented by piecewise-constant hazard rates over the segments defined by
    ``knots``.

    Parameters
    ----------
    knots:
        Strictly increasing positive segment end-times. Hazard ``hazards[k]``
        applies on ``(knots[k-1], knots[k]]`` (with ``knots[-1] = 0``).
        Beyond the last knot the final hazard is held flat.
    hazards:
        Non-negative piecewise-constant hazard rates, same length as ``knots``.
    recovery:
        The :class:`RecoveryRate`.
    """

    knots: np.ndarray
    hazards: np.ndarray
    recovery: RecoveryRate

    def __post_init__(self) -> None:
        knots = np.asarray(self.knots, dtype=float)
        hazards = np.asarray(self.hazards, dtype=float)
        if knots.ndim != 1 or hazards.ndim != 1:
            raise ValidationError("CreditCurve knots and hazards must be one-dimensional.")
        if knots.size == 0 or knots.size != hazards.size:
            raise ValidationError("CreditCurve knots and hazards must be non-empty, equal length.")
        if np.any(knots <= 0.0):
            raise ValidationError("CreditCurve knots must be strictly positive.")
        if np.any(np.diff(knots) <= 0.0):
            raise ValidationError("CreditCurve knots must be strictly increasing.")
        if np.any(hazards < 0.0):
            raise ValidationError("CreditCurve hazards must be non-negative.")
        object.__setattr__(self, "knots", knots)
        object.__setattr__(self, "hazards", hazards)

    @classmethod
    def flat(cls, hazard: float, recovery: float = 0.4) -> CreditCurve:
        """Construct a flat-hazard curve."""
        return cls(
            knots=np.array([100.0]),
            hazards=np.array([float(hazard)]),
            recovery=RecoveryRate(recovery),
        )

    @classmethod
    def piecewise(
        cls, knots: list[float], hazards: list[float], recovery: float = 0.4
    ) -> CreditCurve:
        """Construct a piecewise-constant hazard curve."""
        return cls(
            knots=np.asarray(knots, dtype=float),
            hazards=np.asarray(hazards, dtype=float),
            recovery=RecoveryRate(recovery),
        )

    def hazard(self, t: np.ndarray) -> np.ndarray:
        """Instantaneous hazard ``lambda(t)`` at each time in ``t``."""
        t = np.asarray(t, dtype=float)
        idx = np.searchsorted(self.knots, t, side="left")
        idx = np.clip(idx, 0, self.hazards.size - 1)
        return self.hazards[idx]

    def cumulative_hazard(self, t: np.ndarray) -> np.ndarray:
        r"""Integrated hazard :math:`\int_0^t \lambda(u)\,du` at each time."""
        t = np.asarray(t, dtype=float)
        seg_start = np.concatenate(([0.0], self.knots[:-1]))
        seg_len = self.knots - seg_start
        cum_at_knot = np.concatenate(([0.0], np.cumsum(self.hazards * seg_len)))

        out = np.empty_like(t)
        for j, tj in np.ndenumerate(t):
            k = int(np.searchsorted(self.knots, tj, side="left"))
            k = min(k, self.hazards.size - 1)
            base = cum_at_knot[k]
            out[j] = base + self.hazards[k] * (tj - seg_start[k])
        return out

    def survival(self, grid: TenorGrid) -> np.ndarray:
        """Survival probabilities ``S(t_i)`` on the grid."""
        surv: np.ndarray = np.exp(-self.cumulative_hazard(grid.times))
        return surv

    def marginal_pd(self, grid: TenorGrid) -> np.ndarray:
        """Marginal default probabilities ``PD(t_{i-1}, t_i)`` on the grid.

        Returns the probability of default in each interval, length ``grid.n``,
        with the first interval measured from time ``0``.
        """
        s = self.survival(grid)
        s_prev = np.concatenate(([1.0], s[:-1]))
        pd: np.ndarray = s_prev - s
        return pd
