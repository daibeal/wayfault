"""Wrong-Way / Right-Way Risk: alpha multiplier, classification, diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from wayfault.domain.exposure import EEProfile


class WWRClass(StrEnum):
    """Classification of the dependence direction."""

    WRONG_WAY = "WRONG_WAY"
    RIGHT_WAY = "RIGHT_WAY"
    NEUTRAL = "NEUTRAL"


def alpha_multiplier(wwr_cva: float, baseline_cva: float) -> float:
    """Empirical alpha multiplier ``alpha = WWR-CVA / independent-CVA``.

    Returns ``1.0`` if the baseline CVA is (near) zero, since there is no
    exposure to amplify.
    """
    if abs(baseline_cva) < 1e-300:
        return 1.0
    return wwr_cva / baseline_cva


def ead(alpha: float, eepe: float) -> float:
    """Exposure-at-default view ``EAD = alpha * EEPE``."""
    return alpha * eepe


def classify(dependence_param: float, alpha: float, tol: float = 1e-9) -> WWRClass:
    """Label the dependence from the parameter sign and CVA uplift.

    A positive parameter together with ``alpha > 1`` is wrong-way; a negative
    parameter with ``alpha < 1`` is right-way; otherwise neutral.
    """
    if dependence_param > tol and alpha > 1.0 + tol:
        return WWRClass.WRONG_WAY
    if dependence_param < -tol and alpha < 1.0 - tol:
        return WWRClass.RIGHT_WAY
    return WWRClass.NEUTRAL


@dataclass(frozen=True)
class Diagnostics:
    """Diagnostic summary of the WWR adjustment."""

    uplift_pct: float
    ee_ratio: np.ndarray
    ee_hazard_corr: float


def diagnostics(
    unconditional: EEProfile,
    conditional: EEProfile,
    hazard: np.ndarray,
    baseline_cva: float,
    wwr_cva: float,
) -> Diagnostics:
    """Compute per-tenor diagnostics for the WWR adjustment.

    Parameters
    ----------
    unconditional:
        Unconditional EPE profile.
    conditional:
        Conditional-on-default EE profile.
    hazard:
        Per-tenor hazard rates (length ``grid.n``).
    baseline_cva, wwr_cva:
        The two CVA figures, for the uplift percentage.
    """
    uplift = 0.0 if abs(baseline_cva) < 1e-300 else (wwr_cva - baseline_cva) / baseline_cva * 100.0
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(unconditional.values > 0.0, conditional.values / unconditional.values, 1.0)
    ee = unconditional.values
    if ee.size > 1 and np.std(ee) > 0 and np.std(hazard) > 0:
        corr = float(np.corrcoef(ee, hazard)[0, 1])
    else:
        corr = 0.0
    return Diagnostics(uplift_pct=uplift, ee_ratio=ratio, ee_hazard_corr=corr)
