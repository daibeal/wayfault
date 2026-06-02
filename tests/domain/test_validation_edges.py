"""Edge-case tests closing the remaining validation branches (100% core)."""

from __future__ import annotations

import numpy as np
import pytest

from wayfault.adapters.outbound.credit_flat import FlatHazardCreditCurveSource
from wayfault.adapters.outbound.exposure_inmemory import InMemoryExposureSource
from wayfault.application.dto import WWRRequest
from wayfault.application.service import WrongWayRiskService
from wayfault.domain.credit import CreditCurve, RecoveryRate
from wayfault.domain.errors import ValidationError
from wayfault.domain.exposure import EEProfile, ExposureCube
from wayfault.domain.portfolio import Counterparty, NettingSet
from wayfault.domain.tenors import TenorGrid


def _grid() -> TenorGrid:
    return TenorGrid.from_list([1.0, 2.0])


# --- tenors.py:34 (ndim != 1) ---------------------------------------------
def test_tenor_grid_rejects_non_1d() -> None:
    with pytest.raises(ValidationError, match="one-dimensional"):
        TenorGrid(np.array([[1.0, 2.0], [3.0, 4.0]]))


# --- exposure.py EEProfile 32/34 ------------------------------------------
def test_ee_profile_rejects_non_1d() -> None:
    with pytest.raises(ValidationError, match="one-dimensional"):
        EEProfile(_grid(), np.array([[1.0, 2.0]]))


def test_ee_profile_rejects_wrong_length() -> None:
    with pytest.raises(ValidationError, match="match the grid length"):
        EEProfile(_grid(), np.array([1.0, 2.0, 3.0]))


# --- exposure.py ExposureCube 61/63 ---------------------------------------
def test_exposure_cube_rejects_wrong_columns() -> None:
    with pytest.raises(ValidationError, match="column count"):
        ExposureCube(_grid(), np.array([[1.0, 2.0, 3.0]]))


def test_exposure_cube_rejects_empty() -> None:
    with pytest.raises(ValidationError, match="at least one scenario"):
        ExposureCube(_grid(), np.empty((0, 2)))


# --- credit.py 62/64/66 ----------------------------------------------------
def test_credit_curve_rejects_non_1d() -> None:
    with pytest.raises(ValidationError, match="one-dimensional"):
        CreditCurve(np.array([[1.0]]), np.array([0.02]), RecoveryRate(0.4))


def test_credit_curve_rejects_length_mismatch() -> None:
    with pytest.raises(ValidationError, match="equal length"):
        CreditCurve(np.array([1.0, 2.0]), np.array([0.02]), RecoveryRate(0.4))


def test_credit_curve_rejects_non_positive_knot() -> None:
    with pytest.raises(ValidationError, match="strictly positive"):
        CreditCurve(np.array([0.0]), np.array([0.02]), RecoveryRate(0.4))


# --- portfolio.py 36/38 ----------------------------------------------------
def test_counterparty_rejects_grid_mismatch() -> None:
    c1 = ExposureCube(TenorGrid.from_list([1.0, 2.0]), np.ones((2, 2)))
    c2 = ExposureCube(TenorGrid.from_list([1.0, 3.0]), np.ones((2, 2)))
    with pytest.raises(ValidationError, match="same tenor grid"):
        Counterparty("CP", (NettingSet("a", c1), NettingSet("b", c2)))


def test_counterparty_rejects_scenario_mismatch() -> None:
    grid = TenorGrid.from_list([1.0, 2.0])
    c1 = ExposureCube(grid, np.ones((2, 2)))
    c2 = ExposureCube(grid, np.ones((3, 2)))
    with pytest.raises(ValidationError, match="same scenario count"):
        Counterparty("CP", (NettingSet("a", c1), NettingSet("b", c2)))


# --- service.py 73/80 (model without params/dependence_param) -------------
class _BareModel:
    """A minimal DependenceModel exposing neither params nor dependence_param."""

    def calibrate_to_curve(self, curve: CreditCurve, grid: TenorGrid) -> None:
        return None

    def conditional_ee(
        self, cube: ExposureCube, curve: CreditCurve, grid: TenorGrid
    ) -> EEProfile:
        values = np.maximum(cube.values, 0.0).mean(axis=0)
        return EEProfile(grid, values)


def test_service_handles_model_without_metadata() -> None:
    tenors = [0.5, 1.0]
    src = InMemoryExposureSource(np.array([[1.0, 2.0], [1.0, 2.0]]), tenors)
    credit = FlatHazardCreditCurveSource(hazard=0.02, recovery=0.4)
    request = WWRRequest(exposure=src, credit=credit, model=_BareModel())
    result = WrongWayRiskService().estimate(request)
    assert result.params == {}
    assert result.model == "_BareModel"
