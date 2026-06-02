"""Outbound ports — Protocols the application depends on."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np

from wayfault.domain.credit import CreditCurve
from wayfault.domain.exposure import EEProfile, ExposureCube
from wayfault.domain.tenors import TenorGrid

if TYPE_CHECKING:
    from wayfault.application.dto import WWRResult


class ExposureSource(Protocol):
    """Supplies an :class:`ExposureCube`."""

    def load(self) -> ExposureCube: ...


class CreditCurveSource(Protocol):
    """Supplies a :class:`CreditCurve`."""

    def load(self) -> CreditCurve: ...


class DependenceModel(Protocol):
    """Couples default timing to portfolio value to produce conditional EE."""

    def calibrate_to_curve(self, curve: CreditCurve, grid: TenorGrid) -> None: ...

    def conditional_ee(
        self, cube: ExposureCube, curve: CreditCurve, grid: TenorGrid
    ) -> EEProfile: ...


class Calibrator(Protocol):
    """Estimates dependence parameters from historical data."""

    def fit(
        self, portfolio_value: np.ndarray, credit_factor: np.ndarray
    ) -> dict[str, float]: ...


class ResultSink(Protocol):
    """Writes a :class:`WWRResult` to some destination."""

    def write(self, result: WWRResult) -> None: ...
