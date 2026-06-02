"""Application + acceptance-criteria tests."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from wayfault import estimate_wwr
from wayfault.adapters.outbound.credit_flat import FlatHazardCreditCurveSource
from wayfault.adapters.outbound.dependence_copula import GaussianCopulaModel
from wayfault.adapters.outbound.dependence_hullwhite import HullWhiteHazardModel
from wayfault.adapters.outbound.dependence_independent import IndependentModel
from wayfault.adapters.outbound.exposure_inmemory import InMemoryExposureSource
from wayfault.domain.tenors import TenorGrid
from wayfault.domain.wwr import WWRClass


def test_independence_reduces(exposure, credit) -> None:
    res = estimate_wwr(exposure, credit, IndependentModel())
    assert res.wwr_cva == pytest.approx(res.baseline_cva, rel=1e-6)
    assert res.alpha == pytest.approx(1.0, rel=1e-6)
    assert res.classification == WWRClass.NEUTRAL


def test_hullwhite_b_zero_equals_baseline(exposure, credit) -> None:
    res = estimate_wwr(exposure, credit, HullWhiteHazardModel(b=0.0))
    assert res.wwr_cva == pytest.approx(res.baseline_cva, rel=1e-6)
    assert res.alpha == pytest.approx(1.0, rel=1e-6)


def test_monotonic_wwr(exposure, credit) -> None:
    bs = [-1.0, -0.5, 0.0, 0.5, 1.0]
    cvas = [estimate_wwr(exposure, credit, HullWhiteHazardModel(b=b)).wwr_cva for b in bs]
    assert all(x <= y + 1e-12 for x, y in pairwise(cvas))
    pos = estimate_wwr(exposure, credit, HullWhiteHazardModel(b=0.8))
    neg = estimate_wwr(exposure, credit, HullWhiteHazardModel(b=-0.8))
    assert pos.alpha >= 1.0
    assert neg.alpha <= 1.0
    assert pos.classification == WWRClass.WRONG_WAY
    assert neg.classification == WWRClass.RIGHT_WAY


def test_conditional_dominance(exposure, credit) -> None:
    res = estimate_wwr(exposure, credit, HullWhiteHazardModel(b=0.8))
    assert np.all(res.conditional_ee >= res.epe - 1e-9)


def test_marginal_consistency(exposure, credit) -> None:
    model = HullWhiteHazardModel(b=0.7)
    cube = exposure.load()
    curve = credit.load()
    model.calibrate_to_curve(curve, cube.grid)
    implied = model.implied_marginal_pd(cube, curve, cube.grid)
    target = curve.marginal_pd(cube.grid)
    np.testing.assert_allclose(implied, target, rtol=1e-6)


def test_copula_wwr_direction(exposure, credit) -> None:
    pos = estimate_wwr(exposure, credit, GaussianCopulaModel(rho=0.6))
    neg = estimate_wwr(exposure, credit, GaussianCopulaModel(rho=-0.6))
    assert pos.wwr_cva >= pos.baseline_cva
    assert neg.wwr_cva <= neg.baseline_cva
    assert pos.classification == WWRClass.WRONG_WAY


def test_reproducibility(tenors) -> None:
    arr = np.random.default_rng(123).normal(size=(2000, len(tenors))) + 1.0
    src1 = InMemoryExposureSource(arr.copy(), tenors)
    src2 = InMemoryExposureSource(arr.copy(), tenors)
    credit = FlatHazardCreditCurveSource(hazard=0.03, recovery=0.4)
    r1 = estimate_wwr(src1, credit, HullWhiteHazardModel(b=0.5))
    r2 = estimate_wwr(src2, credit, HullWhiteHazardModel(b=0.5))
    assert r1.to_dict() == r2.to_dict()


def test_result_to_dict_keys(exposure, credit) -> None:
    res = estimate_wwr(exposure, credit, HullWhiteHazardModel(b=0.3))
    d = res.to_dict()
    for key in ("baseline_cva", "wwr_cva", "alpha", "classification", "epe",
                "conditional_ee", "pfe", "eepe", "ead", "model", "params"):
        assert key in d
    assert d["params"] == {"b": 0.3}


def test_eeprofile_grid_mismatch_guard() -> None:
    from wayfault.domain import cva
    from wayfault.domain.credit import CreditCurve
    from wayfault.domain.errors import ValidationError
    from wayfault.domain.exposure import EEProfile
    g1 = TenorGrid.from_list([1.0, 2.0])
    g2 = TenorGrid.from_list([1.0, 3.0])
    ee = EEProfile(g1, np.array([1.0, 1.0]))
    with pytest.raises(ValidationError):
        cva.discounted_cva(ee, CreditCurve.flat(0.02), g2)
