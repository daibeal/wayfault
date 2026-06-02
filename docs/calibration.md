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

## Inverse solvers (prescriptive)

Calibrators estimate parameters from *historical data*. The **inverse solvers**
go the other way — they answer **target-driven** and **reverse-stress**
questions by root-finding on the monotone metric-vs-parameter curve (robust
bisection, numpy-only):

- [`calibrate_to_alpha`][wayfault.calibrate_to_alpha] — find the dependence
  parameter that reproduces a **target alpha** (e.g. a desk/regulator number).
- [`calibrate_to_cva`][wayfault.calibrate_to_cva] — find the parameter that
  reproduces a **target WWR-CVA** (e.g. a historically observed CVA).
- [`find_breakpoint`][wayfault.find_breakpoint] — find the dependence level at
  which a metric (alpha, WWR-CVA, or EAD) **crosses a threshold**:
  *"how much wrong-way correlation until we breach?"*.

You supply a `model_factory` mapping a scalar to a dependence model, so the same
solver works for Hull-White `b`, Gaussian `ρ`, or a copula `θ`.

```python
from wayfault import calibrate_to_alpha, find_breakpoint
from wayfault.adapters.outbound.dependence_hullwhite import HullWhiteHazardModel

hw = lambda b: HullWhiteHazardModel(b=b)

# 1) Calibrate: which b reproduces a desk target alpha of 1.25?
sol = calibrate_to_alpha(exposure, credit, hw, target_alpha=1.25, lo=-1.5, hi=1.5)
print(sol.param, sol.converged)        # -> ~the b giving alpha = 1.25

# 2) Reverse stress: at what b does alpha breach 1.40?
brk = find_breakpoint(exposure, credit, hw, threshold=1.40, lo=0.0, hi=3.0)
print(brk.param, brk.result.classification)
```

Each returns a [`SolveResult`][wayfault.SolveResult] carrying the solved
`param`, the `achieved` metric, `iterations`, a `converged` flag, and the full
`result` ([`WWRResult`][wayfault.WWRResult]) at the solution. The solve is
deterministic. If `[lo, hi]` does not bracket the target, a clear
[`ValidationError`][wayfault.ValidationError] is raised.
