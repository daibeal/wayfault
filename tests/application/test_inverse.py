"""Tests for the inverse WWR solvers."""

from __future__ import annotations

import numpy as np
import pytest

from wayfault import estimate_wwr
from wayfault.adapters.outbound.credit_flat import FlatHazardCreditCurveSource
from wayfault.adapters.outbound.dependence_hullwhite import HullWhiteHazardModel
from wayfault.adapters.outbound.exposure_inmemory import InMemoryExposureSource
from wayfault.application.inverse import (
    calibrate_to_alpha,
    calibrate_to_cva,
    ead_metric,
    find_breakpoint,
)
from wayfault.domain.errors import ValidationError


@pytest.fixture
def exposure() -> InMemoryExposureSource:
    rng = np.random.default_rng(0)
    tenors = [i / 4 for i in range(1, 13)]
    cube = np.asarray(tenors) + rng.normal(scale=0.6, size=(6000, len(tenors)))
    return InMemoryExposureSource(cube, tenors)


@pytest.fixture
def credit() -> FlatHazardCreditCurveSource:
    return FlatHazardCreditCurveSource(hazard=0.02, recovery=0.4)


def _hw(b: float) -> HullWhiteHazardModel:
    return HullWhiteHazardModel(b=b)


def test_calibrate_to_alpha_recovers_known_b(exposure, credit) -> None:
    true_b = 0.6
    target = estimate_wwr(exposure, credit, _hw(true_b)).alpha
    sol = calibrate_to_alpha(exposure, credit, _hw, target, lo=-1.5, hi=1.5)
    assert sol.converged
    assert sol.param == pytest.approx(true_b, abs=1e-2)
    assert sol.achieved == pytest.approx(target, abs=1e-4)
    assert sol.result.alpha == pytest.approx(target, abs=1e-4)


def test_calibrate_to_cva_recovers_known_b(exposure, credit) -> None:
    true_b = -0.4
    target = estimate_wwr(exposure, credit, _hw(true_b)).wwr_cva
    sol = calibrate_to_cva(exposure, credit, _hw, target, lo=-1.5, hi=1.5)
    assert sol.converged
    assert sol.param == pytest.approx(true_b, abs=1e-2)


def test_find_breakpoint_alpha_threshold(exposure, credit) -> None:
    # the parameter at which alpha hits 1.05
    sol = find_breakpoint(exposure, credit, _hw, 1.05, lo=-1.5, hi=1.5)
    assert sol.converged
    assert estimate_wwr(exposure, credit, _hw(sol.param)).alpha == pytest.approx(1.05, abs=1e-3)


def test_find_breakpoint_ead_metric(exposure, credit) -> None:
    base = estimate_wwr(exposure, credit, _hw(0.0)).ead
    sol = find_breakpoint(exposure, credit, _hw, base * 1.1, metric=ead_metric, lo=0.0, hi=1.5)
    assert sol.converged
    assert sol.result.ead == pytest.approx(base * 1.1, rel=1e-3)


def test_bracket_failure_raises(exposure, credit) -> None:
    # alpha can never reach 5.0 within this small bracket -> not bracketed
    with pytest.raises(ValidationError, match="not bracketed"):
        calibrate_to_alpha(exposure, credit, _hw, 5.0, lo=-0.1, hi=0.1)


def test_invalid_bracket_order_raises(exposure, credit) -> None:
    with pytest.raises(ValidationError, match="lo < hi"):
        calibrate_to_alpha(exposure, credit, _hw, 1.1, lo=1.0, hi=1.0)


def test_non_convergence_reports_false(exposure, credit) -> None:
    target = estimate_wwr(exposure, credit, _hw(0.5)).alpha
    sol = calibrate_to_alpha(exposure, credit, _hw, target, lo=-1.5, hi=1.5,
                             tol=1e-15, max_iter=3)
    assert not sol.converged
    assert sol.iterations == 3
