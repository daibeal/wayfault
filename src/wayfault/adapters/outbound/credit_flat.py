"""Flat-hazard credit-curve source (no extras)."""

from __future__ import annotations

from wayfault.domain.credit import CreditCurve


class FlatHazardCreditCurveSource:
    """Supplies a flat-hazard :class:`CreditCurve`."""

    def __init__(self, hazard: float, recovery: float = 0.4) -> None:
        self._curve = CreditCurve.flat(hazard=hazard, recovery=recovery)

    def load(self) -> CreditCurve:
        """Return the flat credit curve."""
        return self._curve
