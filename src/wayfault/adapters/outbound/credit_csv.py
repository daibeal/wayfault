"""CSV credit-curve source ([io] extra, lazy import).

Expects a CSV with columns ``knot`` and ``hazard`` (one row per segment). The
recovery rate is supplied at construction.
"""

from __future__ import annotations

from wayfault.adapters.outbound._optional import require
from wayfault.domain.credit import CreditCurve


class CsvCreditCurveSource:
    """Loads a piecewise-constant-hazard curve from CSV via pandas."""

    def __init__(self, path: str, recovery: float = 0.4) -> None:
        self._path = path
        self._recovery = recovery

    def load(self) -> CreditCurve:
        """Read the curve lazily and build a :class:`CreditCurve`."""
        pd = require("pandas", "io")
        frame = pd.read_csv(self._path)
        knots = [float(x) for x in frame["knot"]]
        hazards = [float(x) for x in frame["hazard"]]
        return CreditCurve.piecewise(knots=knots, hazards=hazards, recovery=self._recovery)
