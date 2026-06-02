# Dependence Models

A **dependence model** couples default timing to portfolio value to produce a
**conditional expected exposure given default**. Each model implements the
[`DependenceModel`][wayfault.ports.outbound.DependenceModel] port and is
calibrated to the input credit curve so that, integrated over the exposure
distribution, it reproduces the marginal PDs (arbitrage consistency).

| Model                                                                       | Knob | WWR when | Tail dep. | Dependencies |
|-----------------------------------------------------------------------------|------|----------|-----------|--------------|
| [`IndependentModel`][wayfault.adapters.outbound.dependence_independent.IndependentModel]   | —    | —        | —         | numpy        |
| [`HullWhiteHazardModel`][wayfault.adapters.outbound.dependence_hullwhite.HullWhiteHazardModel] | `b`  | `b > 0`  | —         | numpy        |
| [`GaussianCopulaModel`][wayfault.adapters.outbound.dependence_copula.GaussianCopulaModel]   | `ρ`  | `ρ > 0`  | none      | numpy        |
| [`ClaytonCopulaModel`][wayfault.adapters.outbound.dependence_archimedean.ClaytonCopulaModel] | `θ`  | `θ > 0`  | lower     | numpy        |
| [`FrankCopulaModel`][wayfault.adapters.outbound.dependence_archimedean.FrankCopulaModel]     | `θ`  | `θ > 0`  | none      | numpy        |

## Independent

Conditional EE ≡ unconditional EE. Selecting it makes the WWR-CVA equal the
baseline CVA exactly, and `alpha == 1`. Useful as a control and a sanity check.

```python
from wayfault.adapters.outbound.dependence_independent import IndependentModel
model = IndependentModel()
```

## Hull-White stochastic hazard

The canonical WWR formulation (Hull & White, 2012):

$$
\lambda(t) = \exp\!\big(a(t) + b\,V(t)\big)
$$

- $a(t)$ is solved per tenor so the model reproduces the curve's marginal PDs.
- $b$ is the **wrong-way knob**: $b > 0$ makes default more likely in
  high-exposure scenarios.

Conditional EE is produced by **re-weighting scenarios** by their
model-implied default likelihood at each tenor (a numerically stable softmax of
$b\,V$). The shared re-weighting core normalises weights and degrades
gracefully to the unconditional expectation when $b = 0$.

```python
from wayfault.adapters.outbound.dependence_hullwhite import HullWhiteHazardModel
model = HullWhiteHazardModel(b=0.5)    # b>0 wrong-way, b<0 right-way
```

!!! info "Properties (each is an acceptance test)"
    - **Monotonic:** `wwr_cva` is non-decreasing in `b`.
    - **Dominance:** under WWR, `conditional_ee(t) ≥ epe(t)` at every tenor.
    - **Marginal consistency:** the integrated implied PDs reproduce the curve.

## Gaussian copula

A one-factor Gaussian copula couples a credit latent factor to a
(rank-normalised) portfolio/market factor with correlation $\rho$. The
conditional default probability given the market factor re-weights scenarios:

$$
\Pr(\text{default}_i \mid y) =
\Phi\!\left(\frac{\Phi^{-1}(p_i) + \rho\, y}{\sqrt{1-\rho^2}}\right)
$$

where $p_i$ is the marginal PD over interval $i$ and $y$ is the standard-normal
score of the portfolio value. The sign of $\rho$ mirrors WWR/RWR.

```python
from wayfault.adapters.outbound.dependence_copula import GaussianCopulaModel
model = GaussianCopulaModel(rho=0.6)   # rho in (-1, 1)
```

## Archimedean copulas (advanced)

Archimedean copulas couple the credit and market margins through a closed-form
generator, capturing **asymmetric tail dependence** — the realistic shape of
wrong-way risk, where defaults cluster precisely in the high-exposure tail. Both
are numpy-only (no SciPy) and use the copula *h-function* (conditional CDF) as
the per-scenario default re-weight. The market margin is oriented so high
exposure maps to the lower tail, making a positive parameter wrong-way.

### Clayton

Lower-tail dependence via the Clayton generator. `θ > 0`; larger `θ` means
stronger clustering of defaults with high exposure. `θ → 0` is independence.

$$
h(u \mid v) = v^{-(\theta+1)}\big(u^{-\theta} + v^{-\theta} - 1\big)^{-(\theta+1)/\theta}
$$

```python
from wayfault.adapters.outbound.dependence_archimedean import ClaytonCopulaModel
model = ClaytonCopulaModel(theta=2.0)   # theta > 0, wrong-way, lower-tail
```

### Frank

Symmetric (no tail dependence); the **sign** of `θ` flips the direction.

```python
from wayfault.adapters.outbound.dependence_archimedean import FrankCopulaModel
wrong_way = FrankCopulaModel(theta=4.0)    # theta > 0
right_way = FrankCopulaModel(theta=-4.0)   # theta < 0
```

!!! info "Performance"
    All built-in models are **fully vectorised** across tenors — the entire
    exposure cube is re-weighted in a single numpy expression, with no per-tenor
    Python loop. See [Performance](performance.md) for batching and parallelism.

## Writing your own

Any object implementing the [`DependenceModel`][wayfault.ports.outbound.DependenceModel]
Protocol works. Optionally expose `dependence_param()` and `params()` so the
service can classify and report metadata:

```python
import numpy as np
from wayfault.domain.exposure import EEProfile

class MyModel:
    def calibrate_to_curve(self, curve, grid) -> None: ...
    def conditional_ee(self, cube, curve, grid) -> EEProfile:
        values = np.maximum(cube.values, 0.0).mean(axis=0)
        return EEProfile(grid, values)
    def dependence_param(self) -> float: return 0.0
    def params(self) -> dict[str, float]: return {}
```
