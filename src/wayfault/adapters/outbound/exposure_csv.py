"""CSV/Parquet exposure source ([io] extra, lazy import)."""

from __future__ import annotations

import numpy as np

from wayfault.adapters.outbound._optional import require
from wayfault.domain.exposure import ExposureCube
from wayfault.domain.tenors import TenorGrid


class CsvExposureSource:
    """Loads an exposure cube from CSV or Parquet via pandas/pyarrow.

    The file's column headers are interpreted as tenor year-fractions and each
    row as one Monte-Carlo scenario.
    """

    def __init__(self, path: str, fmt: str = "csv") -> None:
        self._path = path
        self._fmt = fmt

    def load(self) -> ExposureCube:
        """Read the file lazily and build an :class:`ExposureCube`."""
        pd = require("pandas", "io")
        if self._fmt == "parquet":
            require("pyarrow", "io")
            frame = pd.read_parquet(self._path)
        else:
            frame = pd.read_csv(self._path)
        tenors = np.asarray([float(c) for c in frame.columns], dtype=float)
        grid = TenorGrid(tenors)
        return ExposureCube(grid, frame.to_numpy(dtype=float))
