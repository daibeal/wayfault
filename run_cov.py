"""Coverage runner that imports numpy before starting coverage.

Works around a local toolchain quirk where coverage + pytest re-execute
numpy's C extension (numpy 2.x raises "cannot load module more than once").
By fully importing numpy first, pytest's later ``import numpy`` is a cache hit.
"""

from __future__ import annotations

import sys

import coverage
import numpy  # noqa: F401  (imported first, on purpose)

cov = coverage.Coverage(source=["wayfault.domain", "wayfault.application"])
cov.start()

import pytest  # noqa: E402

rc = pytest.main(["-q", "tests"])

cov.stop()
cov.save()
percent = cov.report(show_missing=True)
print(f"\nTOTAL coverage: {percent:.2f}%")
sys.exit(rc)
