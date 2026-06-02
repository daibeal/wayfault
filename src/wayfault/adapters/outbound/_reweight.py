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
