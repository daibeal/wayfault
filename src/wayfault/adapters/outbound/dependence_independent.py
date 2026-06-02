"""Independent dependence model (FR-WWR-021)."""

from __future__ import annotations

import numpy as np

from wayfault.domain.credit import CreditCurve
from wayfault.domain.exposure import EEProfile, ExposureCube
from wayfault.domain.tenors import TenorGrid


class IndependentModel:
    """Conditional EE equals unconditional EE (no dependence)."""

    def calibrate_to_curve(self, curve: CreditCurve, grid: TenorGrid) -> None:
        """No-op: the independent model has nothing to calibrate."""

    def conditional_ee(
        self, cube: ExposureCube, curve: CreditCurve, grid: TenorGrid
    ) -> EEProfile:
        """Return the unconditional EPE profile unchanged."""
        values = np.maximum(cube.values, 0.0).mean(axis=0)
        return EEProfile(grid, values)

    def dependence_param(self) -> float:
        """The independent model has a zero dependence parameter."""
        return 0.0

    def params(self) -> dict[str, float]:
        """Model metadata."""
        return {}
