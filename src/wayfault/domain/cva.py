"""Unilateral CVA integral (independent and conditional)."""

from __future__ import annotations

import numpy as np

from wayfault.domain.credit import CreditCurve
from wayfault.domain.errors import ValidationError
from wayfault.domain.exposure import EEProfile
from wayfault.domain.tenors import TenorGrid


def discounted_cva(
    ee: EEProfile,
    curve: CreditCurve,
    grid: TenorGrid,
    discount: np.ndarray | None = None,
) -> float:
    r"""Unilateral CVA over the grid.

    Computes

    .. math::

        CVA = (1 - R) \sum_i DF(t_i)\, EE(t_i)\, PD(t_{i-1}, t_i)

    Parameters
    ----------
    ee:
        The (independent or conditional) expected-exposure profile.
    curve:
        The counterparty credit curve (supplies marginal PDs and recovery).
    grid:
        The tenor grid.
    discount:
        Optional discount factors on the grid (length ``grid.n``). Defaults to
        ``1.0`` at every tenor.

    Returns
    -------
    float
        The CVA value.
    """
    if ee.grid != grid:
        raise ValidationError("EEProfile grid must match the supplied grid.")
    if discount is None:
        df = np.ones(grid.n)
    else:
        df = np.asarray(discount, dtype=float)
        if df.shape != (grid.n,):
            raise ValidationError("Discount factors must have length grid.n.")
    pd = curve.marginal_pd(grid)
    lgd = curve.recovery.loss_given_default
    return float(lgd * np.sum(df * ee.values * pd))
