"""Tests for the matplotlib visualization adapter."""

from __future__ import annotations

import numpy as np
import pytest

from wayfault import estimate_wwr
from wayfault.adapters.outbound.credit_flat import FlatHazardCreditCurveSource
from wayfault.adapters.outbound.dependence_hullwhite import HullWhiteHazardModel
from wayfault.adapters.outbound.exposure_inmemory import InMemoryExposureSource

pytest.importorskip("matplotlib")
import matplotlib

matplotlib.use("Agg")

from wayfault.adapters.outbound import viz


@pytest.fixture
def result():
    rng = np.random.default_rng(0)
    tenors = [i / 4 for i in range(1, 9)]
    cube = np.asarray(tenors) + rng.normal(scale=0.5, size=(2000, len(tenors)))
    src = InMemoryExposureSource(cube, tenors)
    credit = FlatHazardCreditCurveSource(hazard=0.02, recovery=0.4)
    return estimate_wwr(src, credit, HullWhiteHazardModel(b=0.7))


def test_exposure_profiles_returns_figure(result) -> None:
    from matplotlib.figure import Figure
    fig = viz.plot_exposure_profiles(result)
    assert isinstance(fig, Figure)
    assert len(fig.axes[0].lines) >= 3


def test_ee_ratio_plot(result) -> None:
    fig = viz.plot_ee_ratio(result)
    assert fig.axes[0].patches  # bars drawn


def test_alpha_sweep_plot() -> None:
    bs = [-0.5, 0.0, 0.5]
    alphas = [0.9, 1.0, 1.1]
    fig = viz.plot_alpha_sweep(bs, alphas, wwr_cvas=[0.04, 0.05, 0.06], baseline_cva=0.05)
    assert fig.axes  # primary + twin axis


def test_dashboard_and_save(result, tmp_path) -> None:
    fig = viz.plot_dashboard(result, bs=[-0.5, 0.0, 0.5], alphas=[0.9, 1.0, 1.1])
    assert len(fig.axes) >= 4
    out = tmp_path / "dash.png"
    viz.save(fig, str(out))
    assert out.exists() and out.stat().st_size > 0


def test_result_carries_tenors(result) -> None:
    np.testing.assert_allclose(result.tenors, [i / 4 for i in range(1, 9)])
    assert "tenors" in result.to_dict()
