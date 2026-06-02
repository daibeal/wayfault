"""Advanced Archimedean-copula dependence models (numpy-only).

These couple a credit margin to the portfolio/market margin through an
Archimedean copula, using the copula's closed-form *h-function* (conditional
CDF) as the per-scenario default re-weight. Unlike the Gaussian copula they
capture **asymmetric tail dependence**, which is the realistic shape of
wrong-way risk: defaults cluster precisely in the high-exposure tail.

- :class:`ClaytonCopulaModel` — lower-tail dependence; ``theta > 0`` is
  wrong-way (and ``theta -> 0`` recovers independence). Pure powers, no special
  functions.
- :class:`FrankCopulaModel` — symmetric, no tail dependence; the sign of
  ``theta`` flips the direction (``theta > 0`` wrong-way, ``theta < 0``
  right-way). Uses only ``exp``.

Both are fully vectorised across tenors via the shared re-weighting core. The
market margin is oriented so that **high exposure maps to the lower tail**
(``v = 1 - empirical_grade(V)``), making a positive ``theta`` wrong-way.
"""

from __future__ import annotations

import numpy as np

from wayfault.adapters.outbound import _reweight
from wayfault.domain.credit import CreditCurve
from wayfault.domain.exposure import EEProfile, ExposureCube
from wayfault.domain.tenors import TenorGrid

_EPS = 1e-12


def _market_grade(values: np.ndarray) -> np.ndarray:
    """Empirical copula grade oriented so high exposure is the lower tail."""
    return np.clip(1.0 - _reweight.rank_uniform_columns(values), _EPS, 1.0 - _EPS)


class ClaytonCopulaModel:
    """Clayton-copula WWR model with lower-tail dependence.

    Parameters
    ----------
    theta:
        Dependence strength ``theta > 0``. Larger ``theta`` means stronger
        clustering of defaults with high exposure (more wrong-way). As
        ``theta -> 0`` the model converges to independence.
    """

    def __init__(self, theta: float = 1.0) -> None:
        if theta <= 0.0:
            raise ValueError("Clayton theta must be > 0.")
        self.theta = float(theta)

    def calibrate_to_curve(self, curve: CreditCurve, grid: TenorGrid) -> None:
        """No eager state; marginal PDs are read on demand."""
        _ = curve.marginal_pd(grid)

    def conditional_ee(
        self, cube: ExposureCube, curve: CreditCurve, grid: TenorGrid
    ) -> EEProfile:
        """Conditional EE via the Clayton h-function re-weighting (vectorised)."""
        th = self.theta
        u = np.clip(curve.marginal_pd(grid), _EPS, 1.0 - _EPS)[None, :]
        v = _market_grade(cube.values)
        # h(u|v) = v^{-(theta+1)} (u^{-theta} + v^{-theta} - 1)^{-(theta+1)/theta}
        inner = np.maximum(u ** (-th) + v ** (-th) - 1.0, _EPS)
        weights = v ** (-(th + 1.0)) * inner ** (-(th + 1.0) / th)
        out = _reweight.conditional_ee_columns(cube.values, weights)
        return EEProfile(grid, out)

    def dependence_param(self) -> float:
        """Positive parameter ⇒ wrong-way; magnitude is the Clayton ``theta``."""
        return self.theta

    def params(self) -> dict[str, float]:
        """Model metadata."""
        return {"theta": self.theta}


class FrankCopulaModel:
    """Frank-copula WWR model (symmetric, sign-controlled direction).

    Parameters
    ----------
    theta:
        Non-zero dependence parameter. ``theta > 0`` is wrong-way, ``theta < 0``
        right-way; ``theta -> 0`` is independence.
    """

    def __init__(self, theta: float = 2.0) -> None:
        if abs(theta) < _EPS:
            raise ValueError("Frank theta must be non-zero (use IndependentModel for 0).")
        self.theta = float(theta)

    def calibrate_to_curve(self, curve: CreditCurve, grid: TenorGrid) -> None:
        """No eager state; marginal PDs are read on demand."""
        _ = curve.marginal_pd(grid)

    def conditional_ee(
        self, cube: ExposureCube, curve: CreditCurve, grid: TenorGrid
    ) -> EEProfile:
        """Conditional EE via the Frank h-function re-weighting (vectorised)."""
        th = self.theta
        u = np.clip(curve.marginal_pd(grid), _EPS, 1.0 - _EPS)[None, :]
        v = _market_grade(cube.values)
        eu = np.expm1(-th * u)  # e^{-theta u} - 1
        ev = np.expm1(-th * v)
        e1 = np.expm1(-th)  # e^{-theta} - 1
        # h(u|v) = e^{-th v}(e^{-th u}-1) / [(e^{-th}-1) + (e^{-th u}-1)(e^{-th v}-1)]
        num = np.exp(-th * v) * eu
        den = e1 + eu * ev
        weights = np.abs(num / np.where(np.abs(den) < _EPS, _EPS, den))
        out = _reweight.conditional_ee_columns(cube.values, weights)
        return EEProfile(grid, out)

    def dependence_param(self) -> float:
        """Signed parameter: ``> 0`` wrong-way, ``< 0`` right-way."""
        return self.theta

    def params(self) -> dict[str, float]:
        """Model metadata."""
        return {"theta": self.theta}
