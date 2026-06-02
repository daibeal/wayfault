"""Standard-normal CDF / inverse-CDF on numpy arrays (no SciPy dependency).

Implemented from :func:`math.erf` / a Newton-refined inverse error function, so
the dependence models stay numpy-only. All functions are array-shape-agnostic.
"""

from __future__ import annotations

import math

import numpy as np

_SQRT2 = math.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _erfinv_scalar(y: float) -> float:
    """Inverse error function (Winitzki seed + Newton refinement)."""
    if y <= -1.0:
        return -math.inf
    if y >= 1.0:
        return math.inf
    a = 0.147
    ln = math.log(1.0 - y * y)
    term = 2.0 / (math.pi * a) + ln / 2.0
    seed = math.copysign(math.sqrt(math.sqrt(term * term - ln / a) - term), y)
    x = seed
    two_over_sqrt_pi = 2.0 / math.sqrt(math.pi)
    for _ in range(3):  # Newton on erf(x) - y = 0; erf'(x) = 2/sqrt(pi) e^{-x^2}
        err = math.erf(x) - y
        x -= err / (two_over_sqrt_pi * math.exp(-x * x))
    return x


_erfinv_vec = np.vectorize(_erfinv_scalar)
_erf_vec = np.vectorize(math.erf)


def norm_ppf(p: np.ndarray) -> np.ndarray:
    """Inverse standard-normal CDF (probit), clipped away from 0/1."""
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1.0 - 1e-12)
    out: np.ndarray = _SQRT2 * _erfinv_vec(2.0 * p - 1.0)
    return out


def norm_cdf(x: np.ndarray) -> np.ndarray:
    """Standard-normal CDF."""
    x = np.asarray(x, dtype=float)
    cdf: np.ndarray = 0.5 * (1.0 + _erf_vec(x / _SQRT2))
    return cdf
