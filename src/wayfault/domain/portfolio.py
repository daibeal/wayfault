"""Counterparty / netting-set aggregation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wayfault.domain.errors import ValidationError
from wayfault.domain.exposure import ExposureCube
from wayfault.domain.tenors import TenorGrid


@dataclass(frozen=True)
class NettingSet:
    """A named netting set holding an exposure cube."""

    name: str
    cube: ExposureCube


@dataclass(frozen=True)
class Counterparty:
    """A counterparty aggregating one or more netting sets."""

    name: str
    netting_sets: tuple[NettingSet, ...]

    def __post_init__(self) -> None:
        if not self.netting_sets:
            raise ValidationError("Counterparty must have at least one netting set.")
        grid = self.netting_sets[0].cube.grid
        n_scen = self.netting_sets[0].cube.n_scenarios
        for ns in self.netting_sets[1:]:
            if ns.cube.grid != grid:
                raise ValidationError("All netting sets must share the same tenor grid.")
            if ns.cube.n_scenarios != n_scen:
                raise ValidationError("All netting sets must share the same scenario count.")

    @property
    def grid(self) -> TenorGrid:
        """The shared tenor grid."""
        return self.netting_sets[0].cube.grid

    def aggregate(self) -> ExposureCube:
        """Aggregate netting-set cubes to a single counterparty-level cube.

        Netting applies *within* a set; across sets values are summed
        scenario-by-scenario (no further netting benefit across sets).
        """
        total = np.zeros_like(self.netting_sets[0].cube.values)
        for ns in self.netting_sets:
            total = total + ns.cube.values
        return ExposureCube(self.grid, total)
