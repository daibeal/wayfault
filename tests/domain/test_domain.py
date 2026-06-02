"""Domain-layer unit tests."""

from __future__ import annotations

import numpy as np
import pytest

from wayfault.domain import cva, metrics, wwr
from wayfault.domain.credit import CreditCurve, RecoveryRate
from wayfault.domain.errors import ValidationError
from wayfault.domain.exposure import EEProfile, ExposureCube
from wayfault.domain.portfolio import Counterparty, NettingSet
from wayfault.domain.tenors import TenorGrid


def test_tenor_grid_validation() -> None:
    g = TenorGrid.from_list([0.25, 0.5, 1.0])
    assert len(g) == 3
    assert g.n == 3
    np.testing.assert_allclose(g.intervals, [0.25, 0.25, 0.5])
    np.testing.assert_allclose(g.midpoints, [0.125, 0.375, 0.75])
    with pytest.raises(ValidationError):
        TenorGrid.from_list([0.5, 0.25])
    with pytest.raises(ValidationError):
        TenorGrid.from_list([-1.0])
    with pytest.raises(ValidationError):
        TenorGrid.from_list([])


def test_tenor_grid_equality_hash() -> None:
    a = TenorGrid.from_list([1.0, 2.0])
    b = TenorGrid.from_list([1.0, 2.0])
    assert a == b
    assert hash(a) == hash(b)
    assert (a == 5) is False or a != 5


def test_exposure_cube_and_profile() -> None:
    g = TenorGrid.from_list([1.0, 2.0])
    cube = ExposureCube(g, np.array([[1.0, -2.0], [3.0, 4.0]]))
    assert cube.n_scenarios == 2
    assert cube.n_tenors == 2
    np.testing.assert_allclose(cube.tenor_slice(0), [1.0, 3.0])
    with pytest.raises(ValidationError):
        ExposureCube(g, np.array([1.0, 2.0]))
    with pytest.raises(ValidationError):
        EEProfile(g, np.array([-1.0, 2.0]))


def test_cube_from_ee_profile() -> None:
    g = TenorGrid.from_list([1.0, 2.0])
    prof = EEProfile(g, np.array([1.0, 2.0]))
    cube = ExposureCube.from_ee_profile(prof)
    np.testing.assert_allclose(metrics.epe(cube).values, prof.values)


def test_recovery_rate() -> None:
    assert RecoveryRate(0.4).loss_given_default == pytest.approx(0.6)
    with pytest.raises(ValidationError):
        RecoveryRate(1.0)
    with pytest.raises(ValidationError):
        RecoveryRate(-0.1)


def test_credit_curve_flat_survival_pd() -> None:
    g = TenorGrid.from_list([1.0, 2.0, 3.0])
    curve = CreditCurve.flat(hazard=0.05, recovery=0.4)
    np.testing.assert_allclose(curve.survival(g), np.exp(-0.05 * g.times))
    pd = curve.marginal_pd(g)
    assert np.all(pd > 0)
    # survival + cumulative PD consistency
    assert curve.survival(g)[-1] + pd.sum() == pytest.approx(1.0)


def test_credit_curve_piecewise() -> None:
    curve = CreditCurve.piecewise(knots=[1.0, 3.0], hazards=[0.02, 0.05], recovery=0.4)
    cum = curve.cumulative_hazard(np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(cum, [0.02, 0.02 + 0.05, 0.02 + 0.10])
    with pytest.raises(ValidationError):
        CreditCurve.piecewise(knots=[2.0, 1.0], hazards=[0.1, 0.1])
    with pytest.raises(ValidationError):
        CreditCurve.piecewise(knots=[1.0], hazards=[-0.1])


def test_metrics() -> None:
    g = TenorGrid.from_list([1.0, 2.0])
    cube = ExposureCube(g, np.array([[2.0, -1.0], [-2.0, 3.0]]))
    np.testing.assert_allclose(metrics.epe(cube).values, [1.0, 1.5])
    np.testing.assert_allclose(metrics.ene(cube).values, [1.0, 0.5])
    np.testing.assert_allclose(metrics.pfe(cube, 0.5), [1.0, 1.5], atol=1.0)
    assert metrics.eepe(cube) >= 0.0
    with pytest.raises(ValidationError):
        metrics.pfe(cube, 1.5)


def test_cva_integral() -> None:
    g = TenorGrid.from_list([1.0, 2.0])
    curve = CreditCurve.flat(hazard=0.05, recovery=0.4)
    ee = EEProfile(g, np.array([10.0, 10.0]))
    val = cva.discounted_cva(ee, curve, g)
    pd = curve.marginal_pd(g)
    assert val == pytest.approx(0.6 * 10.0 * pd.sum())
    # discount applied
    val_disc = cva.discounted_cva(ee, curve, g, discount=np.array([0.5, 0.5]))
    assert val_disc == pytest.approx(0.5 * val)
    with pytest.raises(ValidationError):
        cva.discounted_cva(ee, curve, g, discount=np.array([1.0]))


def test_wwr_classification_alpha() -> None:
    assert wwr.alpha_multiplier(2.0, 1.0) == pytest.approx(2.0)
    assert wwr.alpha_multiplier(1.0, 0.0) == 1.0
    assert wwr.classify(0.5, 1.2) == wwr.WWRClass.WRONG_WAY
    assert wwr.classify(-0.5, 0.8) == wwr.WWRClass.RIGHT_WAY
    assert wwr.classify(0.0, 1.0) == wwr.WWRClass.NEUTRAL
    assert wwr.ead(2.0, 5.0) == pytest.approx(10.0)


def test_diagnostics() -> None:
    g = TenorGrid.from_list([1.0, 2.0])
    uncond = EEProfile(g, np.array([1.0, 2.0]))
    cond = EEProfile(g, np.array([2.0, 3.0]))
    diag = wwr.diagnostics(uncond, cond, np.array([0.02, 0.03]), 1.0, 1.5)
    assert diag.uplift_pct == pytest.approx(50.0)
    np.testing.assert_allclose(diag.ee_ratio, [2.0, 1.5])


def test_counterparty_aggregation() -> None:
    g = TenorGrid.from_list([1.0, 2.0])
    c1 = ExposureCube(g, np.array([[1.0, 1.0], [1.0, 1.0]]))
    c2 = ExposureCube(g, np.array([[2.0, 2.0], [2.0, 2.0]]))
    cp = Counterparty("CP", (NettingSet("a", c1), NettingSet("b", c2)))
    agg = cp.aggregate()
    np.testing.assert_allclose(agg.values, 3.0)
    assert cp.grid == g
    with pytest.raises(ValidationError):
        Counterparty("X", ())
