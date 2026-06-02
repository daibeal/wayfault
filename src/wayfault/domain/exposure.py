"""Exposure value objects: :class:`ExposureCube` and :class:`EEProfile`."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wayfault.domain.errors import ValidationError
from wayfault.domain.tenors import TenorGrid


@dataclass(frozen=True)
class EEProfile:
    """An expected-exposure curve over a tenor grid.

    Parameters
    ----------
    grid:
        The tenor grid the profile is defined on.
    values:
        Expected positive exposure ``E[V(t)+]`` at each tenor. Length must
        equal ``grid.n``. Values must be non-negative.
    """

    grid: TenorGrid
    values: np.ndarray

    def __post_init__(self) -> None:
        arr = np.asarray(self.values, dtype=float)
        if arr.ndim != 1:
            raise ValidationError("EEProfile values must be one-dimensional.")
        if arr.size != self.grid.n:
            raise ValidationError("EEProfile values must match the grid length.")
        if np.any(arr < 0.0):
            raise ValidationError("EEProfile values must be non-negative.")
        object.__setattr__(self, "values", arr)


@dataclass(frozen=True)
class ExposureCube:
    """A Monte-Carlo exposure cube of netting-set mark-to-market values.

    Parameters
    ----------
    grid:
        The tenor grid the columns are defined on.
    values:
        Array of shape ``(n_scenarios, n_tenors)`` of mark-to-market values
        (can be positive or negative).
    """

    grid: TenorGrid
    values: np.ndarray

    def __post_init__(self) -> None:
        arr = np.asarray(self.values, dtype=float)
        if arr.ndim != 2:
            raise ValidationError("ExposureCube values must be two-dimensional.")
        if arr.shape[1] != self.grid.n:
            raise ValidationError("ExposureCube column count must match the grid length.")
        if arr.shape[0] == 0:
            raise ValidationError("ExposureCube must contain at least one scenario.")
        object.__setattr__(self, "values", arr)

    @property
    def n_scenarios(self) -> int:
        """Number of Monte-Carlo scenarios (rows)."""
        return int(self.values.shape[0])

    @property
    def n_tenors(self) -> int:
        """Number of tenors (columns)."""
        return int(self.values.shape[1])

    def tenor_slice(self, i: int) -> np.ndarray:
        """Return the scenario vector at tenor index ``i``."""
        return self.values[:, i]

    @classmethod
    def from_ee_profile(cls, profile: EEProfile) -> ExposureCube:
        """Build a degenerate single-scenario cube from an :class:`EEProfile`.

        This lets users with only a precomputed EE profile still run the
        independent metrics. The single row reproduces the profile exactly for
        the positive-exposure expectation.
        """
        return cls(profile.grid, profile.values.reshape(1, -1))
