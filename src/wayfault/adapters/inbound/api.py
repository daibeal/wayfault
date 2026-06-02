"""Thin Python facade over the WrongWayRiskService."""

from __future__ import annotations

import numpy as np

from wayfault.application.dto import WWRRequest, WWRResult
from wayfault.application.service import WrongWayRiskService
from wayfault.ports.outbound import (
    CreditCurveSource,
    DependenceModel,
    ExposureSource,
    ResultSink,
)


def estimate_wwr(
    exposure: ExposureSource,
    credit: CreditCurveSource,
    model: DependenceModel,
    discount: np.ndarray | None = None,
    pfe_quantile: float = 0.95,
    sink: ResultSink | None = None,
) -> WWRResult:
    """Estimate Wrong-Way Risk for one counterparty.

    Parameters
    ----------
    exposure, credit, model:
        The outbound-port collaborators.
    discount:
        Optional discount factors on the grid.
    pfe_quantile:
        Quantile for the PFE profile.
    sink:
        Optional result sink; if given, the result is also written to it.

    Returns
    -------
    WWRResult
        The full result object.
    """
    request = WWRRequest(
        exposure=exposure,
        credit=credit,
        model=model,
        discount=discount,
        pfe_quantile=pfe_quantile,
    )
    result = WrongWayRiskService().estimate(request)
    if sink is not None:
        sink.write(result)
    return result
