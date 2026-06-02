"""Generate the homepage case-study dataset (real library, realistic inputs).

Models a 5-year receiver interest-rate swap (humped exposure) against a BB-rated
counterparty (upward-sloping hazard), then computes the full WWR analysis and
dumps every series the homepage Plotly charts need to
``docs/assets/data/case_study.json``.

Run with::

    python examples/case_study.py
"""

from __future__ import annotations

import json
import os

import numpy as np

from wayfault import estimate_wwr
from wayfault.adapters.outbound.credit_piecewise import PiecewiseHazardCreditCurveSource
from wayfault.adapters.outbound.dependence_archimedean import (
    ClaytonCopulaModel,
    FrankCopulaModel,
)
from wayfault.adapters.outbound.dependence_copula import GaussianCopulaModel
from wayfault.adapters.outbound.dependence_hullwhite import HullWhiteHazardModel
from wayfault.adapters.outbound.exposure_inmemory import InMemoryExposureSource
from wayfault.domain import metrics

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "assets", "data")
SEED = 20260603
N_SCEN = 20_000
HORIZON = 5.0
RATE = 0.03  # flat discounting
WWR_B = 0.6  # the headline wrong-way coupling


def swap_exposure_cube() -> tuple[np.ndarray, list[float]]:
    """A humped receiver-swap exposure cube with a systematic market factor."""
    rng = np.random.default_rng(SEED)
    tenors = [round(0.25 * k, 2) for k in range(1, 21)]  # quarterly to 5y
    t = np.asarray(tenors)
    # Classic swap MtM hump: rises with accumulated rate moves, decays to maturity.
    hump = np.sqrt(t / HORIZON) * (1.0 - t / HORIZON)
    hump = hump / hump.max()
    market = rng.normal(size=(N_SCEN, 1))  # systematic rate factor
    idio = rng.normal(size=(N_SCEN, len(tenors)))
    drift, vol, rho_m = 0.45, 0.65, 0.7
    factor = drift + vol * (rho_m * market + np.sqrt(1.0 - rho_m**2) * idio)
    cube = hump[None, :] * factor  # O(1) normalised exposure units
    return cube, tenors


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    cube, tenors = swap_exposure_cube()
    t = np.asarray(tenors)
    exposure = InMemoryExposureSource(cube, tenors)
    credit = PiecewiseHazardCreditCurveSource(
        knots=[1.0, 3.0, 5.0, 7.0], hazards=[0.015, 0.022, 0.030, 0.035], recovery=0.4
    )
    discount = np.exp(-RATE * t)

    # Headline wrong-way result (Hull-White, b = 0.6).
    res = estimate_wwr(exposure, credit, HullWhiteHazardModel(b=WWR_B), discount=discount)

    cube_obj = exposure.load()
    curve = credit.load()
    grid = cube_obj.grid
    pfe95 = metrics.pfe(cube_obj, 0.95)
    pfe99 = metrics.pfe(cube_obj, 0.99)
    ene = metrics.ene(cube_obj).values

    # Alpha / CVA sweep over the WWR coupling.
    bs = [round(x, 3) for x in np.linspace(-1.0, 1.0, 21)]
    sweep = []
    for b in bs:
        r = estimate_wwr(exposure, credit, HullWhiteHazardModel(b=b), discount=discount)
        sweep.append({"b": b, "alpha": r.alpha, "wwr_cva": r.wwr_cva})

    # Conditional-EE surface and ratio heatmap over (b, tenor).
    b_grid = [round(x, 3) for x in np.linspace(-1.0, 1.0, 15)]
    surface = []
    ratio = []
    for b in b_grid:
        r = estimate_wwr(exposure, credit, HullWhiteHazardModel(b=b), discount=discount)
        surface.append([float(v) for v in r.conditional_ee])
        ratio.append([
            float(c / e) if e > 0 else 1.0
            for c, e in zip(r.conditional_ee, r.epe, strict=False)
        ])

    # Cross-model comparison (different families, comparable strength).
    comparison = []
    for name, model in [
        ("Hull-White (b=0.6)", HullWhiteHazardModel(b=0.6)),
        ("Gaussian (rho=0.5)", GaussianCopulaModel(rho=0.5)),
        ("Clayton (theta=2)", ClaytonCopulaModel(theta=2.0)),
        ("Frank (theta=5)", FrankCopulaModel(theta=5.0)),
    ]:
        r = estimate_wwr(exposure, credit, model, discount=discount)
        comparison.append(
            {"name": name, "alpha": r.alpha, "wwr_cva": r.wwr_cva, "uplift": r.uplift_pct}
        )

    # Marginal-PD consistency check (Hull-White reproduces the curve).
    hw = HullWhiteHazardModel(b=WWR_B)
    hw.calibrate_to_curve(curve, grid)
    implied = hw.implied_marginal_pd(cube_obj, curve, grid)
    target = curve.marginal_pd(grid)

    data = {
        "meta": {
            "n_scenarios": N_SCEN,
            "horizon": HORIZON,
            "rate": RATE,
            "wwr_b": WWR_B,
            "recovery": 0.4,
        },
        "tenors": tenors,
        "discount": [float(x) for x in discount],
        "epe": [float(x) for x in res.epe],
        "ene": [float(x) for x in ene],
        "pfe95": [float(x) for x in pfe95],
        "pfe99": [float(x) for x in pfe99],
        "conditional_ee": [float(x) for x in res.conditional_ee],
        "ee_ratio": [float(x) for x in res.ee_ratio],
        "survival": [float(x) for x in curve.survival(grid)],
        "hazard": [float(x) for x in curve.hazard(grid.times)],
        "baseline_cva": res.baseline_cva,
        "wwr_cva": res.wwr_cva,
        "alpha": res.alpha,
        "eepe": res.eepe,
        "ead": res.ead,
        "uplift_pct": res.uplift_pct,
        "classification": res.classification.value,
        "sweep": sweep,
        "surface": {"b": b_grid, "tenors": tenors, "z": surface, "ratio": ratio},
        "comparison": comparison,
        "marginal": {
            "tenors": tenors,
            "target": [float(x) for x in target],
            "implied": [float(x) for x in implied],
        },
    }
    path = os.path.join(OUT, "case_study.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    print(f"wrote {os.path.normpath(path)}")
    print(f"baseline_cva={res.baseline_cva:.5f}  wwr_cva={res.wwr_cva:.5f}  "
          f"alpha={res.alpha:.4f}  {res.classification.value}")


if __name__ == "__main__":
    main()
