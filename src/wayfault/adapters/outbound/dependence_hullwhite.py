"""Hull-White stochastic-hazard dependence model (FR-WWR-022).

Canonical WWR formulation following Hull & White (2012):

.. math::

    \\lambda(t) = \\exp(a(t) + b\\, V(t))

The intensity offset ``a(t)`` is solved per tenor so that, integrated over the
exposure distribution, the model reproduces the input curve's marginal PDs
(arbitrage consistency). ``b`` is the wrong-way-risk knob: ``b > 0`` is
wrong-way, ``b < 0`` right-way, ``b = 0`` independence.
"""

from __future__ import annotations

import numpy as np

from wayfault.adapters.outbound import _reweight
from wayfault.domain.credit import CreditCurve
from wayfault.domain.exposure import EEProfile, ExposureCube
from wayfault.domain.tenors import TenorGrid


class HullWhiteHazardModel:
    """Stochastic-hazard WWR model with intensity ``exp(a(t) + b*V(t))``.

    Parameters
    ----------
    b:
        The wrong-way-risk coupling. ``b > 0`` increases default likelihood in
        high-exposure scenarios (wrong-way); ``b < 0`` is right-way.
    """

    def __init__(self, b: float = 0.0) -> None:
        self.b = float(b)
        self._a: np.ndarray | None = None

    def calibrate_to_curve(self, curve: CreditCurve, grid: TenorGrid) -> None:
        """Record the target marginal PDs; ``a(t)`` is solved at re-weighting.

        The per-tenor offset cancels in the normalised conditional expectation,
        so calibration only needs the target marginal PDs, which are recovered
        from the curve on demand.
        """
        # Stored implicitly via the curve at conditional_ee time; nothing to do
        # eagerly besides validating that a curve/grid were provided.
        _ = curve.marginal_pd(grid)

    def conditional_ee(
        self, cube: ExposureCube, curve: CreditCurve, grid: TenorGrid
    ) -> EEProfile:
        """Conditional EE by re-weighting scenarios toward their default risk.

        Fully vectorised: every tenor column is re-weighted in a single numpy
        expression via the shared column-wise softmax core.
        """
        v = cube.values
        weights = _reweight.softmax_columns(self.b * v)
        out = _reweight.conditional_ee_columns(v, weights)

        # Offsets a(t) implied by reproducing the marginal PD in expectation
        # (kept for diagnostics; they cancel in the normalised expectation).
        pd_target = curve.marginal_pd(grid)
        shifted = self.b * (v - v.max(axis=0, keepdims=True))
        mean_exp = np.mean(np.exp(shifted), axis=0)
        self._a = np.log(np.maximum(pd_target, 1e-300)) - (self.b * v).max(axis=0) - np.log(
            mean_exp
        )
        return EEProfile(grid, out)

    def implied_marginal_pd(
        self, cube: ExposureCube, curve: CreditCurve, grid: TenorGrid
    ) -> np.ndarray:
        """Model-implied marginal PDs integrated over the exposure distribution.

        By construction this reproduces ``curve.marginal_pd(grid)`` (criterion
        4): the per-scenario conditional PD is the target PD scaled by the
        normalised exposure weight, whose scenario mean is the target.
        """
        pd_target = curve.marginal_pd(grid)
        w = _reweight.softmax_columns(self.b * cube.values)
        # q_i(s) = pd_target_i * w_{s,i} * S has column mean exactly pd_target_i.
        implied: np.ndarray = np.mean(pd_target[None, :] * w * cube.n_scenarios, axis=0)
        return implied

    def dependence_param(self) -> float:
        """Return the WWR coupling ``b``."""
        return self.b

    def params(self) -> dict[str, float]:
        """Model metadata."""
        return {"b": self.b}
