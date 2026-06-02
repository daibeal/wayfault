"""Inverse WWR solvers (prescriptive analytics).

Where the rest of the library answers *"given this dependence, what is the
WWR?"*, this module inverts the question:

- :func:`calibrate_to_alpha` / :func:`calibrate_to_cva` — find the dependence
  parameter that **reproduces a target** alpha multiplier or WWR-CVA (e.g. a
  desk/regulator target, or a historically observed CVA).
- :func:`find_breakpoint` — find the dependence level at which a chosen metric
  (alpha, WWR-CVA, EAD) **crosses a threshold** — a reverse-stress / capital
  breach diagnostic: *"how much wrong-way correlation until we breach?"*.

All solvers root-find on the monotone metric-vs-parameter curve with a robust
bracketed bisection (deterministic, numpy-only). The caller supplies a
``model_factory`` mapping a scalar parameter to a dependence model, so the same
solver works for the Hull-White ``b``, the Gaussian ``rho``, or a copula
``theta``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from wayfault.application.dto import WWRRequest, WWRResult
from wayfault.application.service import WrongWayRiskService
from wayfault.domain.errors import ValidationError
from wayfault.ports.outbound import (
    CreditCurveSource,
    DependenceModel,
    ExposureSource,
)

ModelFactory = Callable[[float], DependenceModel]
Metric = Callable[[WWRResult], float]


@dataclass(frozen=True)
class SolveResult:
    """Outcome of an inverse solve.

    Attributes
    ----------
    param:
        The solved dependence parameter.
    achieved:
        The metric value attained at ``param``.
    target:
        The requested target / threshold.
    iterations:
        Number of bisection steps taken.
    converged:
        Whether the solver reached ``tol`` within ``max_iter``.
    result:
        The full :class:`WWRResult` evaluated at ``param``.
    """

    param: float
    achieved: float
    target: float
    iterations: int
    converged: bool
    result: WWRResult


def alpha_metric(result: WWRResult) -> float:
    """Metric selector: the alpha multiplier."""
    return result.alpha


def wwr_cva_metric(result: WWRResult) -> float:
    """Metric selector: the WWR-adjusted CVA."""
    return result.wwr_cva


def ead_metric(result: WWRResult) -> float:
    """Metric selector: the exposure-at-default view."""
    return result.ead


def solve_for_target(
    exposure: ExposureSource,
    credit: CreditCurveSource,
    model_factory: ModelFactory,
    target: float,
    *,
    metric: Metric = alpha_metric,
    lo: float,
    hi: float,
    tol: float = 1e-4,
    max_iter: int = 60,
    discount: np.ndarray | None = None,
    pfe_quantile: float = 0.95,
) -> SolveResult:
    """Find the parameter in ``[lo, hi]`` whose ``metric`` equals ``target``.

    The metric must be monotone in the parameter over the bracket (true for the
    built-in models). Raises :class:`ValidationError` if ``[lo, hi]`` does not
    bracket the target.
    """
    if hi <= lo:
        raise ValidationError("Require lo < hi for the solver bracket.")
    service = WrongWayRiskService()

    def evaluate(param: float) -> WWRResult:
        request = WWRRequest(
            exposure=exposure,
            credit=credit,
            model=model_factory(param),
            discount=discount,
            pfe_quantile=pfe_quantile,
        )
        return service.estimate(request)

    res_lo = evaluate(lo)
    res_hi = evaluate(hi)
    f_lo = metric(res_lo) - target
    f_hi = metric(res_hi) - target

    if f_lo * f_hi > 0.0:
        raise ValidationError(
            f"Target {target!r} not bracketed: metric spans "
            f"[{metric(res_lo):.6g}, {metric(res_hi):.6g}] over [{lo}, {hi}]."
        )

    a, b = lo, hi
    mid, res_mid, f_mid = lo, res_lo, f_lo
    for i in range(1, max_iter + 1):
        mid = 0.5 * (a + b)
        res_mid = evaluate(mid)
        f_mid = metric(res_mid) - target
        if abs(f_mid) <= tol:
            return SolveResult(mid, metric(res_mid), target, i, True, res_mid)
        if (f_mid > 0.0) == (f_lo > 0.0):
            a, f_lo = mid, f_mid
        else:
            b = mid
    return SolveResult(mid, metric(res_mid), target, max_iter, False, res_mid)


def calibrate_to_alpha(
    exposure: ExposureSource,
    credit: CreditCurveSource,
    model_factory: ModelFactory,
    target_alpha: float,
    *,
    lo: float,
    hi: float,
    tol: float = 1e-4,
    max_iter: int = 60,
    discount: np.ndarray | None = None,
    pfe_quantile: float = 0.95,
) -> SolveResult:
    """Find the dependence parameter that reproduces ``target_alpha``."""
    return solve_for_target(
        exposure, credit, model_factory, target_alpha, metric=alpha_metric,
        lo=lo, hi=hi, tol=tol, max_iter=max_iter, discount=discount,
        pfe_quantile=pfe_quantile,
    )


def calibrate_to_cva(
    exposure: ExposureSource,
    credit: CreditCurveSource,
    model_factory: ModelFactory,
    target_cva: float,
    *,
    lo: float,
    hi: float,
    tol: float = 1e-8,
    max_iter: int = 80,
    discount: np.ndarray | None = None,
    pfe_quantile: float = 0.95,
) -> SolveResult:
    """Find the dependence parameter that reproduces ``target_cva``."""
    return solve_for_target(
        exposure, credit, model_factory, target_cva, metric=wwr_cva_metric,
        lo=lo, hi=hi, tol=tol, max_iter=max_iter, discount=discount,
        pfe_quantile=pfe_quantile,
    )


def find_breakpoint(
    exposure: ExposureSource,
    credit: CreditCurveSource,
    model_factory: ModelFactory,
    threshold: float,
    *,
    metric: Metric = alpha_metric,
    lo: float,
    hi: float,
    tol: float = 1e-4,
    max_iter: int = 60,
    discount: np.ndarray | None = None,
    pfe_quantile: float = 0.95,
) -> SolveResult:
    """Find the dependence level at which ``metric`` crosses ``threshold``.

    A reverse-stress diagnostic: the returned ``param`` is the breakpoint where
    the chosen metric (default alpha) equals ``threshold``.
    """
    return solve_for_target(
        exposure, credit, model_factory, threshold, metric=metric,
        lo=lo, hi=hi, tol=tol, max_iter=max_iter, discount=discount,
        pfe_quantile=pfe_quantile,
    )
