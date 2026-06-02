"""Tests for the parallel orchestration helpers."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from wayfault import estimate_wwr
from wayfault.adapters.outbound.credit_flat import FlatHazardCreditCurveSource
from wayfault.adapters.outbound.dependence_hullwhite import HullWhiteHazardModel
from wayfault.adapters.outbound.exposure_inmemory import InMemoryExposureSource
from wayfault.application.dto import WWRRequest
from wayfault.application.parallel import batch_estimate, sweep_models


def _exposure() -> InMemoryExposureSource:
    rng = np.random.default_rng(1)
    tenors = [i / 4 for i in range(1, 9)]
    cube = np.asarray(tenors) + rng.normal(scale=0.5, size=(1500, len(tenors)))
    return InMemoryExposureSource(cube, tenors)


def _credit() -> FlatHazardCreditCurveSource:
    return FlatHazardCreditCurveSource(hazard=0.02, recovery=0.4)


def test_sweep_models_matches_sequential() -> None:
    exposure, credit = _exposure(), _credit()
    bs = [-0.5, 0.0, 0.5, 1.0]
    models = [HullWhiteHazardModel(b=b) for b in bs]

    parallel = sweep_models(exposure, credit, models, max_workers=4)
    sequential = [
        estimate_wwr(exposure, credit, HullWhiteHazardModel(b=b)) for b in bs
    ]

    assert [r.to_dict() for r in parallel] == [r.to_dict() for r in sequential]
    # order preserved + monotone alpha
    alphas = [r.alpha for r in parallel]
    assert alphas == sorted(alphas)


def test_batch_estimate_with_explicit_executor() -> None:
    exposure, credit = _exposure(), _credit()
    requests = [
        WWRRequest(exposure, credit, HullWhiteHazardModel(b=b)) for b in (0.0, 0.7)
    ]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = batch_estimate(requests, executor=pool)
    assert len(results) == 2
    assert results[0].alpha == 1.0 or abs(results[0].alpha - 1.0) < 1e-6
    assert results[1].alpha >= 1.0


def test_batch_estimate_deterministic_across_worker_counts() -> None:
    exposure, credit = _exposure(), _credit()
    requests = [
        WWRRequest(exposure, credit, HullWhiteHazardModel(b=b))
        for b in (-0.3, 0.2, 0.9)
    ]
    one = batch_estimate(requests, max_workers=1)
    many = batch_estimate(requests, max_workers=8)
    assert [r.to_dict() for r in one] == [r.to_dict() for r in many]
