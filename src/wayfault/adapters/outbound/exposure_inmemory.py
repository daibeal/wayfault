"""In-memory exposure source (no extras)."""

from __future__ import annotations

import numpy as np

from wayfault.domain.exposure import EEProfile, ExposureCube
from wayfault.domain.tenors import TenorGrid


class InMemoryExposureSource:
    """Wraps an in-memory array (or EE profile) as an :class:`ExposureSource`."""

    def __init__(self, values: np.ndarray, tenors: list[float] | np.ndarray) -> None:
        self._grid = TenorGrid(np.asarray(tenors, dtype=float))
        self._cube = ExposureCube(self._grid, np.asarray(values, dtype=float))

    @classmethod
    def from_ee_profile(cls, profile: EEProfile) -> InMemoryExposureSource:
        """Build a degenerate source from a precomputed EE profile."""
        cube = ExposureCube.from_ee_profile(profile)
        return cls(cube.values, profile.grid.times)

    def load(self) -> ExposureCube:
        """Return the wrapped exposure cube."""
        return self._cube
