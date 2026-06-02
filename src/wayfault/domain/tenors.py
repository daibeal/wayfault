"""The :class:`TenorGrid` value object."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wayfault.domain.errors import ValidationError


@dataclass(frozen=True)
class TenorGrid:
    """A strictly increasing grid of positive year-fractions ``t_1 < ... < t_n``.

    Parameters
    ----------
    times:
        One-dimensional array of strictly increasing, strictly positive
        year-fractions.

    Raises
    ------
    ValidationError
        If the grid is empty, not one-dimensional, contains non-positive
        values, or is not strictly increasing.
    """

    times: np.ndarray

    def __post_init__(self) -> None:
        arr = np.asarray(self.times, dtype=float)
        if arr.ndim != 1:
            raise ValidationError("TenorGrid times must be one-dimensional.")
        if arr.size == 0:
            raise ValidationError("TenorGrid must contain at least one tenor.")
        if np.any(arr <= 0.0):
            raise ValidationError("TenorGrid times must be strictly positive.")
        if np.any(np.diff(arr) <= 0.0):
            raise ValidationError("TenorGrid times must be strictly increasing.")
        object.__setattr__(self, "times", arr)

    @classmethod
    def from_list(cls, times: list[float]) -> TenorGrid:
        """Build a grid from a plain Python list of year-fractions."""
        return cls(np.asarray(times, dtype=float))

    def __len__(self) -> int:
        return int(self.times.size)

    @property
    def n(self) -> int:
        """Number of tenors on the grid."""
        return int(self.times.size)

    @property
    def intervals(self) -> np.ndarray:
        r"""Interval lengths :math:`\Delta t_i`.

        The first interval is measured from ``0`` to ``t_1``; subsequent
        intervals are consecutive differences. Length ``n``.
        """
        prev = np.concatenate(([0.0], self.times[:-1]))
        return self.times - prev

    @property
    def midpoints(self) -> np.ndarray:
        """Midpoints of each interval (from the previous node). Length ``n``."""
        prev = np.concatenate(([0.0], self.times[:-1]))
        mids: np.ndarray = 0.5 * (prev + self.times)
        return mids

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TenorGrid):
            return NotImplemented
        return np.array_equal(self.times, other.times)

    def __hash__(self) -> int:
        return hash(self.times.tobytes())
