"""Parallel orchestration helpers (acceleration).

Batch and parameter-sweep workloads — pricing many counterparties, or sweeping
the wrong-way knob to trace the alpha curve — are *embarrassingly parallel*:
each estimate is independent. These helpers fan the work out across workers
while preserving **deterministic, input-order** results (NFR-4): the output for
index ``i`` depends only on request ``i``, never on the worker count.

Only the standard library and numpy are used. The default backend is a thread
pool, which accelerates the numpy-heavy core because numpy releases the GIL
during array operations. For CPU-bound pure-Python paths, pass your own
``concurrent.futures.ProcessPoolExecutor`` via ``executor``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from concurrent.futures import Executor, ThreadPoolExecutor

import numpy as np

from wayfault.application.dto import WWRRequest, WWRResult
from wayfault.application.service import WrongWayRiskService
from wayfault.ports.outbound import (
    CreditCurveSource,
    DependenceModel,
    ExposureSource,
)


def batch_estimate(
    requests: Iterable[WWRRequest],
    *,
    max_workers: int | None = None,
    executor: Executor | None = None,
) -> list[WWRResult]:
    """Estimate a batch of requests in parallel, preserving input order.

    Parameters
    ----------
    requests:
        The requests to evaluate.
    max_workers:
        Worker count for the default thread pool (ignored if ``executor`` is
        given). ``None`` lets the pool choose.
    executor:
        An optional pre-built ``concurrent.futures.Executor`` (e.g. a
        ``ProcessPoolExecutor``). The caller owns its lifecycle.

    Returns
    -------
    list[WWRResult]
        Results aligned with the input order.
    """
    service = WrongWayRiskService()
    reqs = list(requests)
    if executor is not None:
        return list(executor.map(service.estimate, reqs))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(service.estimate, reqs))


def sweep_models(
    exposure: ExposureSource,
    credit: CreditCurveSource,
    models: Sequence[DependenceModel],
    *,
    discount: np.ndarray | None = None,
    pfe_quantile: float = 0.95,
    max_workers: int | None = None,
    executor: Executor | None = None,
) -> list[WWRResult]:
    """Sweep several dependence models over a shared exposure and credit curve.

    Builds one request per model (re-using the same exposure/credit ports) and
    evaluates them in parallel via :func:`batch_estimate`. Useful for tracing an
    alpha curve across a grid of ``b`` / ``rho`` / ``theta`` values, or for
    comparing model families side by side.
    """
    requests = [
        WWRRequest(
            exposure=exposure,
            credit=credit,
            model=model,
            discount=discount,
            pfe_quantile=pfe_quantile,
        )
        for model in models
    ]
    return batch_estimate(requests, max_workers=max_workers, executor=executor)
