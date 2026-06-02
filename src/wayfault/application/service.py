"""The :class:`WrongWayRiskService` orchestration."""

from __future__ import annotations

from wayfault.application.dto import WWRRequest, WWRResult
from wayfault.domain import cva, metrics, wwr


class WrongWayRiskService:
    """Orchestrates the WWR estimation through the outbound ports.

    The service contains no I/O and no numpy-free leakage of external concerns:
    everything outside the domain is reached through the request's ports.
    """

    def estimate(self, request: WWRRequest) -> WWRResult:
        """Run the full WWR pipeline and return a :class:`WWRResult`."""
        cube = request.exposure.load()
        curve = request.credit.load()
        grid = cube.grid

        model = request.model
        model.calibrate_to_curve(curve, grid)

        # Baseline (independent) metrics.
        epe_profile = metrics.epe(cube)
        pfe_profile = metrics.pfe(cube, request.pfe_quantile)
        eepe_value = metrics.eepe(cube)
        baseline_cva = cva.discounted_cva(epe_profile, curve, grid, request.discount)

        # Conditional exposure and WWR-CVA.
        conditional = model.conditional_ee(cube, curve, grid)
        wwr_cva = cva.discounted_cva(conditional, curve, grid, request.discount)

        alpha = wwr.alpha_multiplier(wwr_cva, baseline_cva)
        params = _model_params(model)
        dep_param = _dependence_param(model)
        classification = wwr.classify(dep_param, alpha)
        ead_value = wwr.ead(alpha, eepe_value)

        diag = wwr.diagnostics(
            unconditional=epe_profile,
            conditional=conditional,
            hazard=curve.hazard(grid.times),
            baseline_cva=baseline_cva,
            wwr_cva=wwr_cva,
        )

        return WWRResult(
            baseline_cva=baseline_cva,
            wwr_cva=wwr_cva,
            alpha=alpha,
            classification=classification,
            epe=epe_profile.values,
            conditional_ee=conditional.values,
            pfe=pfe_profile,
            eepe=eepe_value,
            ead=ead_value,
            uplift_pct=diag.uplift_pct,
            ee_ratio=diag.ee_ratio,
            ee_hazard_corr=diag.ee_hazard_corr,
            model=type(model).__name__,
            params=params,
        )


def _model_params(model: object) -> dict[str, float]:
    getter = getattr(model, "params", None)
    if callable(getter):
        result = getter()
        return dict(result) if result else {}
    return {}


def _dependence_param(model: object) -> float:
    getter = getattr(model, "dependence_param", None)
    if callable(getter):
        return float(getter())
    return 0.0
