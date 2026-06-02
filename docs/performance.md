# Performance

`wayfault` is built for production CCR/XVA workloads. Two complementary
acceleration techniques keep it fast while staying numpy-only.

## 1. Vectorised re-weighting

Every built-in dependence model re-weights the **entire exposure cube at once**.
Instead of looping over tenors in Python, the shared re-weighting core computes
column-wise softmax / copula weights and the conditional EE in single numpy
expressions:

```python
# conceptually, for the whole (n_scenarios x n_tenors) cube:
W   = softmax_columns(b * V)            # column-wise, numerically stable
ee  = conditional_ee_columns(V, W)      # per-tenor conditional EE
```

This is both faster and numerically identical to a per-tenor loop, and it lets
numpy push the work down to vectorised BLAS-backed kernels.

## 2. Parallel batch & sweep

Pricing many counterparties, or sweeping the wrong-way knob to trace an alpha
curve, is *embarrassingly parallel* — each estimate is independent. The
[`wayfault.application.parallel`][wayfault.application.parallel] module fans the
work out across workers while preserving **deterministic, input-order results**
(same output regardless of worker count).

### Sweep a parameter grid

```python
import numpy as np
from wayfault.application.parallel import sweep_models
from wayfault.adapters.outbound.exposure_inmemory import InMemoryExposureSource
from wayfault.adapters.outbound.credit_flat import FlatHazardCreditCurveSource
from wayfault.adapters.outbound.dependence_hullwhite import HullWhiteHazardModel

tenors = [i / 4 for i in range(1, 13)]
cube = np.asarray(tenors) + np.random.default_rng(0).normal(scale=0.6, size=(20_000, 12))
exposure = InMemoryExposureSource(cube, tenors)
credit = FlatHazardCreditCurveSource(hazard=0.02, recovery=0.4)

bs = np.linspace(-1.2, 1.2, 25)
results = sweep_models(
    exposure, credit,
    [HullWhiteHazardModel(b=b) for b in bs],
    max_workers=8,                       # default thread pool
)
alphas = [r.alpha for r in results]      # aligned with bs
```

### Batch many counterparties

```python
from wayfault.application.parallel import batch_estimate
from wayfault.application.dto import WWRRequest

requests = [WWRRequest(exp_i, credit_i, model_i) for exp_i, credit_i, model_i in book]
results = batch_estimate(requests, max_workers=16)
```

### Threads vs processes

The default backend is a **thread pool**. Because numpy releases the GIL during
array operations, threads already overlap the heavy numeric work with low
overhead. For CPU-bound, pure-Python-heavy paths, pass your own
`ProcessPoolExecutor`:

```python
from concurrent.futures import ProcessPoolExecutor

with ProcessPoolExecutor(max_workers=8) as pool:
    results = batch_estimate(requests, executor=pool)
```

!!! success "Determinism guarantee"
    For a fixed seed and inputs, `batch_estimate` / `sweep_models` return
    **identical** results to the sequential path, independent of `max_workers`
    or backend (NFR-4). This is enforced by tests.

## API

::: wayfault.application.parallel
