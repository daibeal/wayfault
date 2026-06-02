# wayfault

**Wrong-Way Risk (WWR) estimation for counterparty credit risk.**

[![PyPI](https://img.shields.io/pypi/v/wayfault.svg)](https://pypi.org/project/wayfault/)
[![Python](https://img.shields.io/pypi/pyversions/wayfault.svg)](https://pypi.org/project/wayfault/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/daibeal/wayfault/blob/main/LICENSE)

`wayfault` quantifies the adverse dependence between exposure and counterparty
credit quality — the risk that exposure rises precisely when the counterparty
deteriorates (**Wrong-Way Risk**), and its favourable mirror, **Right-Way
Risk**.

It takes a Monte-Carlo **exposure cube** and a **credit curve** as inputs and
produces:

- baseline (independence-assumption) exposure metrics and CVA,
- a **conditional expected exposure given default** under a pluggable
  dependence model,
- a **WWR-adjusted CVA** and the empirical **alpha multiplier**
  $\alpha = \text{WWR-CVA} / \text{independent-CVA}$,
- ML-based **calibration** of the dependence parameter,
- WWR/RWR **classification and diagnostics**.

The library does **not** generate exposures or bootstrap curves — those are
inputs.

## Why wayfault

!!! tip "Design principles"
    - **Minimal core.** The only hard runtime dependency is `numpy`. Everything
      else (`pandas`, `scikit-learn`, `matplotlib`) is an *optional extra*,
      imported lazily.
    - **Hexagonal architecture.** A pure domain, surrounded by ports
      (Protocols) and swappable adapters. The dependency rule points inward.
    - **Deterministic.** Same seed ⇒ identical results.
    - **Typed & tested.** `mypy --strict` clean, `ruff` clean, ≥ 90 % coverage
      on the core.

## Install

```bash
pip install wayfault                 # core (numpy only)
pip install 'wayfault[io,ml,viz]'    # with optional extras
```

## 30-second example

```python
import numpy as np
from wayfault import estimate_wwr
from wayfault.adapters.outbound.exposure_inmemory import InMemoryExposureSource
from wayfault.adapters.outbound.credit_flat import FlatHazardCreditCurveSource
from wayfault.adapters.outbound.dependence_hullwhite import HullWhiteHazardModel

cube = np.random.default_rng(0).normal(size=(10_000, 12)) + 1.0
tenors = [i / 4 for i in range(1, 13)]   # quarterly to 3y

result = estimate_wwr(
    exposure=InMemoryExposureSource(cube, tenors),
    credit=FlatHazardCreditCurveSource(hazard=0.02, recovery=0.4),
    model=HullWhiteHazardModel(b=0.5),    # b > 0  ->  wrong-way
)

print(result.baseline_cva, result.wwr_cva, result.alpha, result.classification)
```

Continue with the [Getting Started](getting-started.md) guide, or jump to the
[API Reference](api/index.md).
