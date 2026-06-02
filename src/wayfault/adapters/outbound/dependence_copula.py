"""Gaussian-copula dependence model (FR-WWR-023).

Couples a credit latent factor to a portfolio/market factor with correlation
``rho`` via a one-factor Gaussian copula. The conditional default probability
given the (rank-normalised) portfolio value re-weights scenarios to produce the
conditional EE. ``rho > 0`` is wrong-way, ``rho < 0`` right-way.
"""

from __future__ import annotations

import math

import numpy as np

from wayfault.adapters.outbound import _reweight
from wayfault.domain.credit import CreditCurve
from wayfault.domain.exposure import EEProfile, ExposureCube
from wayfault.domain.tenors import TenorGrid


def _norm_ppf(p: np.ndarray) -> np.ndarray:
    """Vectorised standard-normal inverse CDF via the error function."""
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1.0 - 1e-12)
    vfn = np.vectorize(lambda x: math.sqrt(2.0) * _erfinv(2.0 * x - 1.0))
    out: np.ndarray = vfn(p)
    return out


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    """Vectorised standard-normal CDF."""
    x = np.asarray(x, dtype=float)
    cdf: np.ndarray = 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))
    return cdf


def _erfinv(y: float) -> float:
    """Inverse error function (Newton refinement on a rational seed)."""
    if y <= -1.0:
        return -math.inf
    if y >= 1.0:
        return math.inf
    # Winitzki approximation as a seed.
    a = 0.147
    ln = math.log(1.0 - y * y)
    term = 2.0 / (math.pi * a) + ln / 2.0
    seed = math.copysign(math.sqrt(math.sqrt(term * term - ln / a) - term), y)
    x = seed
    for _ in range(3):  # Newton steps on erf(x) - y = 0
        err = math.erf(x) - y
        x -= err / (2.0 / math.sqrt(math.pi) * math.exp(-x * x))
    return x


def _rank_normal_scores(v: np.ndarray) -> np.ndarray:
    """Rank-based standard-normal scores of a sample (robust market factor)."""
    n = v.size
    order = np.argsort(np.argsort(v))
    uniform = (order + 0.5) / n
    return _norm_ppf(uniform)


class GaussianCopulaModel:
    """One-factor Gaussian-copula WWR model with correlation ``rho``."""

    def __init__(self, rho: float = 0.0) -> None:
        if not (-1.0 < rho < 1.0):
            raise ValueError("rho must be in (-1, 1).")
        self.rho = float(rho)

    def calibrate_to_curve(self, curve: CreditCurve, grid: TenorGrid) -> None:
        """No eager state: marginal PDs are read from the curve on demand."""
        _ = curve.marginal_pd(grid)

    def conditional_ee(
        self, cube: ExposureCube, curve: CreditCurve, grid: TenorGrid
    ) -> EEProfile:
        """Conditional EE via Gaussian-copula default re-weighting."""
        positive = np.maximum(cube.values, 0.0)
        pd = curve.marginal_pd(grid)
        out = np.empty(grid.n)
        denom = math.sqrt(max(1.0 - self.rho * self.rho, 1e-12))
        for i in range(grid.n):
            v = cube.values[:, i]
            y = _rank_normal_scores(v)
            thresh = _norm_ppf(np.full(v.shape, pd[i]))
            cond_pd = _norm_cdf((thresh + self.rho * y) / denom)
            weights = _reweight.normalize_weights(cond_pd)
            out[i] = float(np.sum(positive[:, i] * weights))
        return EEProfile(grid, out)

    def dependence_param(self) -> float:
        """Return the copula correlation ``rho``."""
        return self.rho

    def params(self) -> dict[str, float]:
        """Model metadata."""
        return {"rho": self.rho}
