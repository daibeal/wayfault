"""Inbound port — the primary use case Protocol."""

from __future__ import annotations

from typing import Protocol

from wayfault.application.dto import WWRRequest, WWRResult


class WrongWayRiskUseCase(Protocol):
    """The single inbound use case of the library."""

    def estimate(self, request: WWRRequest) -> WWRResult: ...
