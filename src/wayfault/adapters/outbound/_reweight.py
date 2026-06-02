"""Shared scenario re-weighting numerical core (FR-WWR-024).

Given per-scenario importance weights at a tenor (proportional to the
model-implied default likelihood conditional on the portfolio value), compute
the conditional expected positive exposure as a weighted average. Weights are
normalised and numerically stabilised.
"""

from __future__ import annotations

import numpy as np


def normalize_weights(raw: np.ndarray) -> np.ndarray:
    """Normalise a non-negative weight vector to sum to one.

    Falls back to uniform weights if the total mass is (near) zero, so the
    conditional exposure degenerates gracefully to the unconditional one.
    """
    raw = np.asarray(raw, dtype=float)
    raw = np.where(np.isfinite(raw) & (raw > 0.0), raw, 0.0)
    total = raw.sum()
    if total <= 0.0:
        return np.full(raw.shape, 1.0 / raw.size)
    weights: np.ndarray = raw / total
    return weights


def conditional_expectation(positive_exposure: np.ndarray, weights: np.ndarray) -> float:
    """Weighted mean of the positive-exposure column under ``weights``."""
    w = normalize_weights(weights)
    return float(np.sum(positive_exposure * w))


def softmax_weights(scores: np.ndarray) -> np.ndarray:
    """Numerically stable exponential weights ``exp(score) / sum``."""
    scores = np.asarray(scores, dtype=float)
    shifted = scores - np.max(scores)
    exp = np.exp(shifted)
    return normalize_weights(exp)


# --- Vectorised core (acceleration) ---------------------------------------
#
# The functions below operate on the **whole cube at once** (shape
# ``(n_scenarios, n_tenors)``), re-weighting every tenor column in a single
# numpy expression instead of a per-tenor Python loop. This is both faster and
# numerically identical to the scalar helpers above.


def normalize_columns(raw: np.ndarray) -> np.ndarray:
    """Column-wise normalisation of a ``(S, T)`` weight matrix to unit columns.

    Columns whose mass is (near) zero fall back to uniform weights, mirroring
    :func:`normalize_weights` so the conditional exposure degrades gracefully.
    """
    raw = np.asarray(raw, dtype=float)
    raw = np.where(np.isfinite(raw) & (raw > 0.0), raw, 0.0)
    totals = raw.sum(axis=0, keepdims=True)
    n = raw.shape[0]
    safe = totals > 0.0
    weights = np.where(safe, raw / np.where(safe, totals, 1.0), 1.0 / n)
    return weights


def softmax_columns(scores: np.ndarray) -> np.ndarray:
    """Column-wise numerically stable softmax of a ``(S, T)`` score matrix."""
    scores = np.asarray(scores, dtype=float)
    shifted = scores - scores.max(axis=0, keepdims=True)
    return normalize_columns(np.exp(shifted))


def conditional_ee_columns(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Per-tenor conditional EE from a ``(S, T)`` cube and column weights.

    ``weights`` need not be pre-normalised; they are normalised column-wise
    here. Returns a length-``T`` vector of conditional expected positive
    exposures.
    """
    positive = np.maximum(np.asarray(values, dtype=float), 0.0)
    w = normalize_columns(weights)
    ee: np.ndarray = np.sum(positive * w, axis=0)
    return ee


def rank_uniform_columns(values: np.ndarray) -> np.ndarray:
    """Column-wise rank-based uniforms in ``(0, 1)`` (the empirical copula grade).

    For each tenor column, returns ``(rank + 0.5) / n_scenarios`` — a robust,
    distribution-free mapping of the portfolio value to a uniform margin.
    """
    values = np.asarray(values, dtype=float)
    n = values.shape[0]
    ranks = np.argsort(np.argsort(values, axis=0), axis=0)
    grades: np.ndarray = (ranks + 0.5) / n
    return grades
