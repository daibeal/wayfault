"""Piecewise-constant-hazard credit-curve source (no extras)."""

from __future__ import annotations

from wayfault.domain.credit import CreditCurve


class PiecewiseHazardCreditCurveSource:
    """Supplies a piecewise-constant-hazard :class:`CreditCurve`."""

    def __init__(
        self, knots: list[float], hazards: list[float], recovery: float = 0.4
    ) -> None:
        self._curve = CreditCurve.piecewise(knots=knots, hazards=hazards, recovery=recovery)

    def load(self) -> CreditCurve:
        """Return the piecewise credit curve."""
        return self._curve
