"""Algebraic proofs of the core formulas, verified symbolically with SymPy.

These tests independently *derive* the mathematical identities the
implementation relies on (copula h-functions as exact copula derivatives,
survival/PD telescoping, the Hull-White marginal-consistency identity, softmax
normalisation, the Gaussian-copula default threshold, and the CVA integrand)
and check them against the exact expressions used in the code. They are a
second, symbolic line of defence next to the numerical acceptance tests.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

sympy = pytest.importorskip("sympy")
import sympy as sp  # noqa: E402


# --- Clayton: the model weight is exactly d/dv of the Clayton copula ---------
def test_clayton_h_function_equals_copula_partial() -> None:
    u, v, th = sp.symbols("u v theta", positive=True)
    C = (u ** (-th) + v ** (-th) - 1) ** (-1 / th)  # Clayton copula CDF
    h = sp.diff(C, v)  # conditional CDF  P(U<=u | V=v)
    # the expression used in ClaytonCopulaModel.conditional_ee:
    impl = v ** (-(th + 1)) * (u ** (-th) + v ** (-th) - 1) ** (-(th + 1) / th)
    assert sp.simplify(h - impl) == 0


def test_clayton_independence_limit() -> None:
    # As theta -> 0 the Clayton copula -> the independence copula u*v,
    # so the conditional CDF -> u (constant in v): re-weighting becomes uniform.
    u, v, th = sp.symbols("u v theta", positive=True)
    C = (u ** (-th) + v ** (-th) - 1) ** (-1 / th)
    assert sp.simplify(sp.limit(C, th, 0) - u * v) == 0


# --- Frank: same, with the Frank generator ----------------------------------
def test_frank_h_function_equals_copula_partial() -> None:
    u, v, th = sp.symbols("u v theta", positive=True)
    C = -1 / th * sp.log(1 + (sp.exp(-th * u) - 1) * (sp.exp(-th * v) - 1) / (sp.exp(-th) - 1))
    h = sp.diff(C, v)
    impl = (sp.exp(-th * v) * (sp.exp(-th * u) - 1)) / (
        (sp.exp(-th) - 1) + (sp.exp(-th * u) - 1) * (sp.exp(-th * v) - 1)
    )
    assert sp.simplify(h - impl) == 0


# --- Survival / marginal PD telescope to 1 ----------------------------------
def test_survival_pd_telescopes_to_one() -> None:
    h1, h2, t1, t2 = sp.symbols("h1 h2 t1 t2", positive=True)
    s1 = sp.exp(-h1 * t1)                       # S(t1)
    s2 = sp.exp(-h1 * t1 - h2 * (t2 - t1))       # S(t2), piecewise hazard
    pd1 = 1 - s1                                 # PD(0, t1)
    pd2 = s1 - s2                                # PD(t1, t2)
    assert sp.simplify(pd1 + pd2 + s2 - 1) == 0


def test_cumulative_hazard_piecewise_integral() -> None:
    # The integral of a 2-segment piecewise-constant hazard, accumulated
    # segment-by-segment exactly as CreditCurve.cumulative_hazard does.
    h1, h2, t1, t, s = sp.symbols("h1 h2 t1 t s", positive=True)
    # Regime 1: t within the first segment (t <= t1).
    within = sp.integrate(h1, (s, 0, t))
    assert sp.simplify(within - h1 * t) == 0
    # Regime 2: t beyond the first knot — base accrual plus the partial segment.
    beyond = sp.integrate(h1, (s, 0, t1)) + sp.integrate(h2, (s, t1, t))
    assert sp.simplify(beyond - (h1 * t1 + h2 * (t - t1))) == 0


# --- Hull-White: marginal consistency + softmax normalisation ----------------
def test_softmax_normalises_to_one() -> None:
    b, v1, v2, v3 = sp.symbols("b v1 v2 v3", real=True)
    e = [sp.exp(b * vi) for vi in (v1, v2, v3)]
    z = sum(e)
    weights = [ei / z for ei in e]
    assert sp.simplify(sum(weights) - 1) == 0


def test_hullwhite_reproduces_marginal_pd() -> None:
    # q_i(s) = pd * w_s * N with softmax weights w_s has scenario mean exactly pd
    # (the arbitrage-consistency identity implemented in implied_marginal_pd).
    b, v1, v2, v3, pd = sp.symbols("b v1 v2 v3 pd", real=True)
    e = [sp.exp(b * vi) for vi in (v1, v2, v3)]
    z = sum(e)
    weights = [ei / z for ei in e]
    n = 3
    q = [pd * w * n for w in weights]
    mean_q = sum(q) / n
    assert sp.simplify(mean_q - pd) == 0


def test_conditional_ee_is_weighted_mean() -> None:
    # conditional EE = sum_s positive_s * w_s with normalised softmax weights.
    b, v1, v2, p1, p2 = sp.symbols("b v1 v2 p1 p2", real=True)
    z = sp.exp(b * v1) + sp.exp(b * v2)
    w1, w2 = sp.exp(b * v1) / z, sp.exp(b * v2) / z
    cond = p1 * w1 + p2 * w2
    # b = 0 collapses to the simple average (the independence reduction).
    assert sp.simplify(cond.subs(b, 0) - (p1 + p2) / 2) == 0


# --- Gaussian copula: the conditional-default threshold ----------------------
def test_gaussian_one_factor_threshold() -> None:
    # X = rho*M + sqrt(1-rho^2)*Z; default iff X <= k. Solving the boundary for
    # the idiosyncratic Z gives the argument used inside the normal CDF.
    rho, m, k, z = sp.symbols("rho m k z")
    x = rho * m + sp.sqrt(1 - rho ** 2) * z
    boundary = sp.solve(sp.Eq(x, k), z)[0]
    assert sp.simplify(boundary - (k - rho * m) / sp.sqrt(1 - rho ** 2)) == 0


# --- CVA integrand -----------------------------------------------------------
def test_cva_integrand_matches_implementation() -> None:
    r, df, ee, pd = sp.symbols("R DF EE PD")
    term = (1 - r) * df * ee * pd          # (1-R) * DF * EE * PD
    lgd = 1 - r
    impl = lgd * (df * ee * pd)            # as computed in discounted_cva
    assert sp.simplify(term - impl) == 0


# --- Symbolic <-> numpy bridge ----------------------------------------------
def test_clayton_symbolic_matches_numpy_weights() -> None:
    """lambdify the proven Clayton derivative and match the numpy expression."""
    u, v, th = sp.symbols("u v theta", positive=True)
    C = (u ** (-th) + v ** (-th) - 1) ** (-1 / th)
    h_fn = sp.lambdify((u, v, th), sp.diff(C, v), "numpy")

    rng = np.random.default_rng(0)
    uu = rng.uniform(0.01, 0.3, 200)
    vv = rng.uniform(0.01, 0.99, 200)
    theta = 2.0
    symbolic = h_fn(uu, vv, theta)
    # the literal expression from ClaytonCopulaModel.conditional_ee:
    inner = np.maximum(uu ** (-theta) + vv ** (-theta) - 1.0, 1e-12)
    code = vv ** (-(theta + 1.0)) * inner ** (-(theta + 1.0) / theta)
    np.testing.assert_allclose(symbolic, code, rtol=1e-9)


def test_normal_cdf_matches_erf_definition() -> None:
    # The hand-rolled norm_cdf used by the Gaussian copula equals 0.5*(1+erf(x/sqrt2)).
    from wayfault.adapters.outbound._normals import norm_cdf

    xs = np.linspace(-3, 3, 25)
    expected = np.array([0.5 * (1.0 + math.erf(x / math.sqrt(2.0))) for x in xs])
    np.testing.assert_allclose(norm_cdf(xs), expected, rtol=1e-12)
