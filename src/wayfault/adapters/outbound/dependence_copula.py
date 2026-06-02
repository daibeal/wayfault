"""Gaussian-copula dependence model (FR-WWR-023).

Couples a credit latent factor to a portfolio/market factor with correlation
``rho`` via a one-factor Gaussian copula. The conditional default probability
given the (rank-normalised) portfolio value re-weights scenarios to produce the
conditional EE. ``rho > 0`` is wrong-way, ``rho < 0`` right-way.

Fully vectorised across tenors via the shared column-wise re-weighting core.
"""

from __future__ import annotations

import math

from wayfault.adapters.outbound import _normals, _reweight
from wayfault.domain.credit import CreditCurve
from wayfault.domain.exposure import EEProfile, ExposureCube
from wayfault.domain.tenors import TenorGrid


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
        """Conditional EE via Gaussian-copula default re-weighting (vectorised)."""
        pd = curve.marginal_pd(grid)
        denom = math.sqrt(max(1.0 - self.rho * self.rho, 1e-12))
        y = _normals.norm_ppf(_reweight.rank_uniform_columns(cube.values))
        thresh = _normals.norm_ppf(pd)[None, :]
        cond_pd = _normals.norm_cdf((thresh + self.rho * y) / denom)
        out = _reweight.conditional_ee_columns(cube.values, cond_pd)
        return EEProfile(grid, out)

    def dependence_param(self) -> float:
        """Return the copula correlation ``rho``."""
        return self.rho

    def params(self) -> dict[str, float]:
        """Model metadata."""
        return {"rho": self.rho}
