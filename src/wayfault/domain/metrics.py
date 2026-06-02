"""Baseline exposure metrics: EPE, ENE, PFE, EEPE."""

from __future__ import annotations

import numpy as np

from wayfault.domain.errors import ValidationError
from wayfault.domain.exposure import EEProfile, ExposureCube


def epe(cube: ExposureCube) -> EEProfile:
    """Expected positive exposure ``EPE(t) = E[V(t)+]`` per tenor."""
    values = np.maximum(cube.values, 0.0).mean(axis=0)
    return EEProfile(cube.grid, values)


def ene(cube: ExposureCube) -> EEProfile:
    """Expected negative exposure ``ENE(t) = E[(-V(t))+]`` per tenor."""
    values = np.maximum(-cube.values, 0.0).mean(axis=0)
    return EEProfile(cube.grid, values)


def pfe(cube: ExposureCube, q: float = 0.95) -> np.ndarray:
    """Potential future exposure: the ``q``-quantile of ``V(t)+`` per tenor."""
    if not (0.0 < q < 1.0):
        raise ValidationError("PFE quantile q must be in (0, 1).")
    positive = np.maximum(cube.values, 0.0)
    quantiles: np.ndarray = np.quantile(positive, q, axis=0)
    return quantiles


def eepe(cube: ExposureCube) -> float:
    """Effective expected positive exposure (time-weighted running-max EE).

    EEE(t) is the running maximum of EPE up to ``t``; EEPE is the
    time-weighted average of EEE over the grid (Basel definition), using the
    grid interval lengths as weights.
    """
    ee = epe(cube).values
    eee = np.maximum.accumulate(ee)
    dt = cube.grid.intervals
    horizon = float(cube.grid.times[-1])
    return float(np.sum(eee * dt) / horizon)
