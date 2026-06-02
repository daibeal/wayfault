# wayfault

**Wrong-Way Risk (WWR) estimation for counterparty credit risk.**

`wayfault` quantifies the adverse dependence between exposure and counterparty
credit quality — the risk that exposure rises precisely when the counterparty
deteriorates (WWR), and its favourable mirror, Right-Way Risk (RWR). It takes a
Monte-Carlo **exposure cube** and a **credit curve** as inputs and produces:

- baseline (independence-assumption) exposure metrics and CVA,
- a **conditional expected exposure given default** under a pluggable
  dependence model,
- a **WWR-adjusted CVA** and the empirical **alpha multiplier**
  `α = WWR-CVA / independent-CVA`,
- ML-based **calibration** of the dependence parameter,
- WWR/RWR **classification and diagnostics**.

The library does **not** generate exposures or bootstrap curves — those are
inputs.

## Architecture

`wayfault` follows a strict **hexagonal (ports & adapters)** design. The
dependency rule points inward: `adapters → application → ports → domain`. The
`domain`, `application`, and `ports` layers import **only the standard library
and numpy**. Optional adapters import their heavy dependencies *lazily* and
raise a clear `MissingDependencyError` if the extra is not installed.

## Install

```bash
pip install -e .                       # core (numpy only)
pip install -e '.[io,ml,viz,dev]'      # with all optional extras + tooling
```

Optional extras:

| Extra   | Enables                                   | Pulls in              |
|---------|-------------------------------------------|-----------------------|
| `[io]`  | CSV/Parquet source adapters               | `pandas`, `pyarrow`   |
| `[ml]`  | scikit-learn survival calibrator          | `scikit-learn`        |
| `[viz]` | diagnostic plots                          | `matplotlib`          |
| `[dev]` | tests, type-checking, linting             | `pytest`, `mypy`, …   |

## Quickstart

```python
import numpy as np
from wayfault import estimate_wwr
from wayfault.adapters.outbound.exposure_inmemory import InMemoryExposureSource
from wayfault.adapters.outbound.credit_flat import FlatHazardCreditCurveSource
from wayfault.adapters.outbound.dependence_hullwhite import HullWhiteHazardModel

cube = np.random.default_rng(0).normal(size=(10_000, 12)) + 1.0  # scenarios x tenors
tenors = [i / 4 for i in range(1, 13)]                           # quarterly to 3y

result = estimate_wwr(
    exposure=InMemoryExposureSource(cube, tenors),
    credit=FlatHazardCreditCurveSource(hazard=0.02, recovery=0.4),
    model=HullWhiteHazardModel(b=0.5),   # b > 0  ->  wrong-way
)

print(result.baseline_cva, result.wwr_cva, result.alpha, result.classification)
```

A full runnable example lives in [`examples/quickstart.py`](examples/quickstart.py)
and uses only the in-memory adapters (zero extras).

## CLI

```bash
python -m wayfault estimate \
    --exposure cube.csv --credit curve.csv \
    --model hullwhite --b 0.5 --out result.json
```

(`--exposure`/`--credit` CSV ingestion requires the `[io]` extra.)

## Dependence models

| Model                   | Knob | WWR when | Notes                                   |
|-------------------------|------|----------|-----------------------------------------|
| `IndependentModel`      | —    | —        | conditional EE ≡ unconditional EE       |
| `HullWhiteHazardModel`  | `b`  | `b > 0`  | `λ(t) = exp(a(t) + b·V(t))` (Hull–White)|
| `GaussianCopulaModel`   | `ρ`  | `ρ > 0`  | one-factor Gaussian copula              |

## Calibration

- `RegressionCalibrator` — numpy-only OLS estimate of the Hull–White `b`.
- `SklearnSurvivalCalibrator` — `[ml]` covariate-hazard surrogate.

## Development

```bash
pytest                 # tests
mypy --strict src      # type checks
ruff check             # lint
```

Reference: Hull & White, *CVA and Wrong-Way Risk* (2012).

## License

MIT.
