"""Numpy-only regression calibrator (FR-WWR-041)."""

from __future__ import annotations

import numpy as np

from wayfault.domain.errors import ValidationError


class RegressionCalibrator:
    """Estimate the Hull-White ``b`` by least-squares regression.

    The Hull-White intensity is ``lambda = exp(a + b*V)``, so ``log lambda`` is
    linear in the portfolio value ``V`` with slope ``b``. The ``credit_factor``
    samples are interpreted as realised hazard rates; their logs are regressed
    on the portfolio value to recover ``b`` (and the intercept ``a``).
    """

    def fit(
        self, portfolio_value: np.ndarray, credit_factor: np.ndarray
    ) -> dict[str, float]:
        """Return ``{"a": intercept, "b": slope}`` from an OLS fit."""
        v = np.asarray(portfolio_value, dtype=float).ravel()
        c = np.asarray(credit_factor, dtype=float).ravel()
        if v.size != c.size:
            raise ValidationError("portfolio_value and credit_factor must be equal length.")
        if v.size < 2:
            raise ValidationError("Need at least two samples to calibrate.")
        if np.any(c <= 0.0):
            raise ValidationError("credit_factor (hazard) samples must be positive.")
        log_hazard = np.log(c)
        design = np.column_stack([np.ones_like(v), v])
        coef, *_ = np.linalg.lstsq(design, log_hazard, rcond=None)
        return {"a": float(coef[0]), "b": float(coef[1])}
