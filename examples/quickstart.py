"""Runnable quickstart — uses only in-memory adapters (zero extras).

Run with::

    python examples/quickstart.py
"""

from __future__ import annotations

import numpy as np

from wayfault import estimate_wwr
from wayfault.adapters.outbound.credit_flat import FlatHazardCreditCurveSource
from wayfault.adapters.outbound.dependence_hullwhite import HullWhiteHazardModel
from wayfault.adapters.outbound.exposure_inmemory import InMemoryExposureSource


def toy_cube(seed: int = 0) -> tuple[np.ndarray, list[float]]:
    """A toy positive-drift exposure cube (allowed in examples only)."""
    rng = np.random.default_rng(seed)
    tenors = [i / 4 for i in range(1, 13)]  # quarterly to 3y
    drift = np.asarray(tenors)
    cube = drift + rng.normal(scale=0.5, size=(10_000, len(tenors)))
    return cube, tenors


def main() -> None:
    cube, tenors = toy_cube()
    exposure = InMemoryExposureSource(cube, tenors)
    credit = FlatHazardCreditCurveSource(hazard=0.02, recovery=0.4)

    for b in (0.0, 0.5, -0.5):
        result = estimate_wwr(exposure, credit, HullWhiteHazardModel(b=b))
        print(
            f"b={b:+.1f}  baseline_cva={result.baseline_cva:.5f}  "
            f"wwr_cva={result.wwr_cva:.5f}  alpha={result.alpha:.4f}  "
            f"{result.classification.value}"
        )


if __name__ == "__main__":
    main()
