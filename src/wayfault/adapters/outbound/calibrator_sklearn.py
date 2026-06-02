"""scikit-learn covariate-hazard calibrator ([ml] extra, lazy import)."""

from __future__ import annotations

import numpy as np

from wayfault.adapters.outbound._optional import require
from wayfault.domain.errors import ValidationError


class SklearnSurvivalCalibrator:
    """Fit a covariate-driven (log-)hazard model and reduce it to ``b``.

    Uses a gradient-boosting regressor on ``log(credit_factor)`` against the
    portfolio value, then linearises the fitted response to a single
    Hull-White ``b`` slope that a :class:`DependenceModel` can consume. Raises
    :class:`MissingDependencyError` if scikit-learn is absent.
    """

    def __init__(self, **estimator_kwargs: object) -> None:
        self._kwargs = estimator_kwargs

    def fit(
        self, portfolio_value: np.ndarray, credit_factor: np.ndarray
    ) -> dict[str, float]:
        """Fit the survival surrogate and return ``{"b": slope}``."""
        ensemble = require("sklearn.ensemble", "ml")
        v = np.asarray(portfolio_value, dtype=float).ravel()
        c = np.asarray(credit_factor, dtype=float).ravel()
        if v.size != c.size:
            raise ValidationError("portfolio_value and credit_factor must be equal length.")
        if np.any(c <= 0.0):
            raise ValidationError("credit_factor (hazard) samples must be positive.")
        log_hazard = np.log(c)
        model = ensemble.GradientBoostingRegressor(**self._kwargs)
        model.fit(v.reshape(-1, 1), log_hazard)
        # Linearise: finite-difference slope of the fitted response over the
        # observed support, reduced to a single Hull-White coupling.
        lo, hi = float(np.min(v)), float(np.max(v))
        if hi - lo < 1e-12:
            return {"b": 0.0}
        grid = np.linspace(lo, hi, 50).reshape(-1, 1)
        pred = model.predict(grid)
        slope = float(np.polyfit(grid.ravel(), pred, 1)[0])
        return {"b": slope}
