"""Tests for the advanced Archimedean copula models and vectorised core."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from wayfault import estimate_wwr
from wayfault.adapters.outbound import _reweight
from wayfault.adapters.outbound.credit_flat import FlatHazardCreditCurveSource
from wayfault.adapters.outbound.dependence_archimedean import (
    ClaytonCopulaModel,
    FrankCopulaModel,
)
from wayfault.adapters.outbound.exposure_inmemory import InMemoryExposureSource
from wayfault.domain.wwr import WWRClass


@pytest.fixture
def exposure() -> InMemoryExposureSource:
    rng = np.random.default_rng(0)
    tenors = [i / 4 for i in range(1, 13)]
    cube = np.asarray(tenors) + rng.normal(scale=0.6, size=(6000, len(tenors)))
    return InMemoryExposureSource(cube, tenors)


@pytest.fixture
def credit() -> FlatHazardCreditCurveSource:
    return FlatHazardCreditCurveSource(hazard=0.02, recovery=0.4)


# --- vectorised re-weighting core -----------------------------------------
def test_softmax_columns_independence() -> None:
    v = np.random.default_rng(0).normal(size=(100, 5))
    w = _reweight.softmax_columns(0.0 * v)
    np.testing.assert_allclose(w, np.full_like(w, 1.0 / 100))


def test_conditional_ee_columns_matches_mean_when_uniform() -> None:
    v = np.random.default_rng(0).normal(size=(200, 4))
    uniform = np.ones_like(v)
    ee = _reweight.conditional_ee_columns(v, uniform)
    np.testing.assert_allclose(ee, np.maximum(v, 0.0).mean(axis=0))


def test_normalize_columns_zero_mass_falls_back_uniform() -> None:
    raw = np.zeros((10, 3))
    w = _reweight.normalize_columns(raw)
    np.testing.assert_allclose(w, np.full_like(w, 0.1))


def test_rank_uniform_columns_in_unit_interval() -> None:
    v = np.random.default_rng(0).normal(size=(50, 3))
    u = _reweight.rank_uniform_columns(v)
    assert np.all((u > 0.0) & (u < 1.0))


# --- Clayton ---------------------------------------------------------------
def test_clayton_is_wrong_way(exposure, credit) -> None:
    res = estimate_wwr(exposure, credit, ClaytonCopulaModel(theta=2.0))
    assert res.alpha >= 1.0
    assert res.classification == WWRClass.WRONG_WAY
    assert np.all(res.conditional_ee >= res.epe - 1e-9)


def test_clayton_monotonic_in_theta(exposure, credit) -> None:
    thetas = [0.2, 0.8, 2.0, 5.0]
    alphas = [estimate_wwr(exposure, credit, ClaytonCopulaModel(t)).alpha for t in thetas]
    assert all(x <= y + 1e-6 for x, y in pairwise(alphas))


def test_clayton_rejects_non_positive() -> None:
    with pytest.raises(ValueError, match="theta must be > 0"):
        ClaytonCopulaModel(theta=0.0)


# --- Frank -----------------------------------------------------------------
def test_frank_direction(exposure, credit) -> None:
    pos = estimate_wwr(exposure, credit, FrankCopulaModel(theta=4.0))
    neg = estimate_wwr(exposure, credit, FrankCopulaModel(theta=-4.0))
    assert pos.alpha >= 1.0 and pos.classification == WWRClass.WRONG_WAY
    assert neg.alpha <= 1.0 and neg.classification == WWRClass.RIGHT_WAY


def test_frank_rejects_zero() -> None:
    with pytest.raises(ValueError, match="non-zero"):
        FrankCopulaModel(theta=0.0)


def test_advanced_models_carry_metadata(exposure, credit) -> None:
    res = estimate_wwr(exposure, credit, ClaytonCopulaModel(theta=1.5))
    assert res.params == {"theta": 1.5}
    assert res.model == "ClaytonCopulaModel"
