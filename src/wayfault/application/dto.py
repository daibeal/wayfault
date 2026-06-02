"""Data-transfer objects for the WWR use case."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from wayfault.domain.wwr import WWRClass

if TYPE_CHECKING:
    from wayfault.ports.outbound import (
        CreditCurveSource,
        DependenceModel,
        ExposureSource,
    )


@dataclass
class WWRRequest:
    """Inputs for a WWR estimation.

    Parameters
    ----------
    exposure:
        Port supplying the exposure cube.
    credit:
        Port supplying the credit curve.
    model:
        The dependence model to apply.
    discount:
        Optional discount factors on the grid (defaults to 1.0).
    pfe_quantile:
        Quantile used for the PFE profile.
    """

    exposure: ExposureSource
    credit: CreditCurveSource
    model: DependenceModel
    discount: np.ndarray | None = None
    pfe_quantile: float = 0.95


@dataclass
class WWRResult:
    """Outputs of a WWR estimation.

    Attributes
    ----------
    baseline_cva, wwr_cva, alpha:
        The independent CVA, the WWR-adjusted CVA, and their ratio.
    classification:
        WWR / RWR / NEUTRAL label.
    tenors:
        The tenor grid year-fractions (x-axis for the per-tenor profiles).
    epe, conditional_ee, pfe:
        Per-tenor profiles (numpy arrays).
    eepe:
        Effective EPE scalar.
    ead:
        ``alpha * eepe``.
    uplift_pct, ee_ratio, ee_hazard_corr:
        Diagnostics.
    model, params:
        Metadata about the dependence model used.
    """

    baseline_cva: float
    wwr_cva: float
    alpha: float
    classification: WWRClass
    tenors: np.ndarray
    epe: np.ndarray
    conditional_ee: np.ndarray
    pfe: np.ndarray
    eepe: float
    ead: float
    uplift_pct: float
    ee_ratio: np.ndarray
    ee_hazard_corr: float
    model: str
    params: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary representation."""
        return {
            "baseline_cva": self.baseline_cva,
            "wwr_cva": self.wwr_cva,
            "alpha": self.alpha,
            "classification": self.classification.value,
            "tenors": self.tenors.tolist(),
            "epe": self.epe.tolist(),
            "conditional_ee": self.conditional_ee.tolist(),
            "pfe": self.pfe.tolist(),
            "eepe": self.eepe,
            "ead": self.ead,
            "uplift_pct": self.uplift_pct,
            "ee_ratio": self.ee_ratio.tolist(),
            "ee_hazard_corr": self.ee_hazard_corr,
            "model": self.model,
            "params": self.params,
        }
