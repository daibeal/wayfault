"""Generate the wayfault plot gallery (requires the ``[viz]`` extra).

Writes PNGs into ``docs/assets/img/`` so they can be embedded in both the
README and the MkDocs site.

Run with::

    pip install 'wayfault[viz]'
    python examples/gallery.py
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # headless / reproducible rendering

import numpy as np

from wayfault import estimate_wwr
from wayfault.adapters.outbound import viz
from wayfault.adapters.outbound.credit_flat import FlatHazardCreditCurveSource
from wayfault.adapters.outbound.dependence_hullwhite import HullWhiteHazardModel
from wayfault.adapters.outbound.exposure_inmemory import InMemoryExposureSource

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "assets", "img")


def toy_exposure(seed: int = 0) -> InMemoryExposureSource:
    """A toy positive-drift exposure cube (examples only)."""
    rng = np.random.default_rng(seed)
    tenors = [i / 4 for i in range(1, 13)]  # quarterly to 3y
    drift = np.asarray(tenors)
    cube = drift + rng.normal(scale=0.6, size=(20_000, len(tenors)))
    return InMemoryExposureSource(cube, tenors)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    exposure = toy_exposure()
    credit = FlatHazardCreditCurveSource(hazard=0.02, recovery=0.4)

    wwr = estimate_wwr(exposure, credit, HullWhiteHazardModel(b=0.8))

    # Sweep the WWR knob for the alpha curve.
    bs = [round(x, 2) for x in np.linspace(-1.2, 1.2, 13)]
    sweep = [estimate_wwr(exposure, credit, HullWhiteHazardModel(b=b)) for b in bs]
    alphas = [r.alpha for r in sweep]
    wwr_cvas = [r.wwr_cva for r in sweep]
    baseline = sweep[0].baseline_cva

    figs = {
        "exposure_profiles.png": viz.plot_exposure_profiles(wwr),
        "ee_ratio.png": viz.plot_ee_ratio(wwr),
        "alpha_sweep.png": viz.plot_alpha_sweep(bs, alphas, wwr_cvas, baseline),
        "dashboard.png": viz.plot_dashboard(wwr, bs, alphas, wwr_cvas),
    }
    for name, fig in figs.items():
        path = os.path.join(OUT, name)
        viz.save(fig, path)
        print(f"wrote {os.path.normpath(path)}")


if __name__ == "__main__":
    main()
