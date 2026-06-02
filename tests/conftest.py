"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from wayfault.adapters.outbound.credit_flat import FlatHazardCreditCurveSource
from wayfault.adapters.outbound.exposure_inmemory import InMemoryExposureSource


@pytest.fixture
def tenors() -> list[float]:
    return [i / 4 for i in range(1, 13)]  # quarterly to 3y


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def cube_array(rng: np.random.Generator, tenors: list[float]) -> np.ndarray:
    # Drifting positive exposure with noise so EPE is non-trivial.
    n_scen = 4000
    base = np.asarray(tenors)
    return base + rng.normal(scale=0.5, size=(n_scen, len(tenors)))


@pytest.fixture
def exposure(cube_array: np.ndarray, tenors: list[float]) -> InMemoryExposureSource:
    return InMemoryExposureSource(cube_array, tenors)


@pytest.fixture
def credit() -> FlatHazardCreditCurveSource:
    return FlatHazardCreditCurveSource(hazard=0.02, recovery=0.4)
