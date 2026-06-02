# Calibration

A **calibrator** estimates dependence parameters from historical
`(portfolio_value, credit_factor)` samples. Each implements the
[`Calibrator`][wayfault.ports.outbound.Calibrator] port and returns a `dict` of
parameters that round-trips directly into a [dependence model](models.md).

| Calibrator                                                                              | Extra | Dependencies   |
|-----------------------------------------------------------------------------------------|-------|----------------|
| [`RegressionCalibrator`][wayfault.adapters.outbound.calibrator_regression.RegressionCalibrator] | none  | numpy          |
| [`SklearnSurvivalCalibrator`][wayfault.adapters.outbound.calibrator_sklearn.SklearnSurvivalCalibrator] | `[ml]` | scikit-learn |

## Regression calibrator (numpy only)

Since the Hull-White intensity is $\lambda = \exp(a + b\,V)$, $\log\lambda$ is
linear in the portfolio value with slope $b$. The `credit_factor` samples are
interpreted as realised hazard rates; their logs are regressed on the portfolio
value (ordinary least squares) to recover $b$ and the intercept $a$.

```python
import numpy as np
from wayfault.adapters.outbound.calibrator_regression import RegressionCalibrator
from wayfault.adapters.outbound.dependence_hullwhite import HullWhiteHazardModel

rng = np.random.default_rng(7)
true_b, true_a = 0.6, np.log(0.02)
v = rng.normal(size=5000)
hazard = np.exp(true_a + true_b * v)         # simulated credit-factor history

params = RegressionCalibrator().fit(v, hazard)
# {'a': -3.912..., 'b': 0.600...}

model = HullWhiteHazardModel(b=params["b"])  # round-trip into a model
```

!!! success "Round-trip guarantee"
    The calibrator recovers a known `b` from data simulated with that `b`
    (within tolerance) — this is an acceptance test.

## Sklearn survival calibrator (`[ml]`)

Fits a covariate-driven (log-)hazard surrogate (gradient-boosted regression)
and linearises the fitted response over the observed support to a single
Hull-White $b$ that a dependence model can consume.

```python
from wayfault.adapters.outbound.calibrator_sklearn import SklearnSurvivalCalibrator

params = SklearnSurvivalCalibrator(n_estimators=200).fit(v, hazard)
model = HullWhiteHazardModel(b=params["b"])
```

If `scikit-learn` is not installed, calling `.fit(...)` raises a clear
[`MissingDependencyError`][wayfault.MissingDependencyError] — never an
`ImportError` at import time.

```bash
pip install 'wayfault[ml]'
```
