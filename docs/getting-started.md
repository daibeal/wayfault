# Getting Started

## Installation

`wayfault` requires **Python ≥ 3.11**.

```bash
pip install wayfault
```

Optional extras pull in heavier dependencies, imported lazily only when the
relevant adapter is used:

| Extra   | Enables                            | Pulls in              |
|---------|------------------------------------|-----------------------|
| `[io]`  | CSV/Parquet source adapters        | `pandas`, `pyarrow`   |
| `[ml]`  | scikit-learn survival calibrator   | `scikit-learn`        |
| `[viz]` | diagnostic plots                   | `matplotlib`          |
| `[dev]` | tests, type-checking, linting      | `pytest`, `mypy`, …   |

```bash
pip install 'wayfault[io,ml,viz]'
```

!!! note "Lazy optional dependencies"
    Importing `wayfault` never imports `pandas`, `scikit-learn`, or
    `matplotlib`. If you use an adapter whose extra is not installed, you get a
    clear [`MissingDependencyError`][wayfault.MissingDependencyError] with the
    exact `pip install` hint — not an `ImportError` at import time.

## Inputs

`wayfault` consumes two things per counterparty:

1. **An exposure cube** — a `(n_scenarios × n_tenors)` array of netting-set
   mark-to-market values from your Monte-Carlo engine. If you only have a
   precomputed EE profile, you can still run the independent metrics via
   [`ExposureCube.from_ee_profile`][wayfault.domain.exposure.ExposureCube.from_ee_profile].
2. **A credit curve** — a flat or piecewise-constant hazard term structure with
   a recovery rate.

These are wired in through **source adapters**:

```python
import numpy as np
from wayfault.adapters.outbound.exposure_inmemory import InMemoryExposureSource
from wayfault.adapters.outbound.credit_flat import FlatHazardCreditCurveSource

cube = np.random.default_rng(0).normal(size=(10_000, 12)) + 1.0
tenors = [i / 4 for i in range(1, 13)]

exposure = InMemoryExposureSource(cube, tenors)
credit = FlatHazardCreditCurveSource(hazard=0.02, recovery=0.4)
```

## Estimating WWR

Pick a [dependence model](models.md) and call the facade:

```python
from wayfault import estimate_wwr
from wayfault.adapters.outbound.dependence_hullwhite import HullWhiteHazardModel

result = estimate_wwr(exposure, credit, HullWhiteHazardModel(b=0.5))
```

The returned [`WWRResult`][wayfault.WWRResult] carries everything:

```python
result.baseline_cva      # independent CVA
result.wwr_cva           # WWR-adjusted CVA
result.alpha             # wwr_cva / baseline_cva
result.classification    # WRONG_WAY / RIGHT_WAY / NEUTRAL
result.epe               # per-tenor expected positive exposure
result.conditional_ee    # per-tenor conditional EE given default
result.pfe               # per-tenor potential future exposure
result.eepe              # effective EPE scalar
result.ead               # alpha * eepe
result.to_dict()         # JSON-serialisable view
```

## Interpreting the result

| Knob       | Sign      | Meaning                              | Expected `alpha` |
|------------|-----------|--------------------------------------|------------------|
| `b` / `ρ`  | `> 0`     | wrong-way (exposure ↑ with default)  | `≥ 1`            |
| `b` / `ρ`  | `= 0`     | independence                         | `= 1`            |
| `b` / `ρ`  | `< 0`     | right-way (exposure ↓ with default)  | `≤ 1`            |

```text
b=+0.0  baseline_cva=0.05672  wwr_cva=0.05672  alpha=1.0000  NEUTRAL
b=+0.5  baseline_cva=0.05672  wwr_cva=0.06092  alpha=1.0741  WRONG_WAY
b=-0.5  baseline_cva=0.05672  wwr_cva=0.05259  alpha=0.9272  RIGHT_WAY
```

## Writing results

Pass a [`ResultSink`][wayfault.ports.outbound.ResultSink] to capture output:

```python
from wayfault.adapters.outbound.sinks import JsonReportWriter

estimate_wwr(exposure, credit, HullWhiteHazardModel(b=0.5),
             sink=JsonReportWriter("result.json"))
```

## Next steps

- [Architecture](architecture.md) — how the layers fit together.
- [Dependence Models](models.md) — Independent, Hull-White, Gaussian copula.
- [Calibration](calibration.md) — estimating the dependence parameter from data.
- [CLI](cli.md) — running from the command line.
