"""wayfault — Wrong-Way Risk (WWR) estimation for counterparty credit risk.

The public facade re-exports the convenience entry point and the result types.
Importing this package pulls in only the standard library and numpy; optional
adapters import their heavy dependencies lazily.
"""

from __future__ import annotations

from wayfault.adapters.inbound.api import estimate_wwr
from wayfault.application.dto import WWRRequest, WWRResult
from wayfault.domain.errors import (
    MissingDependencyError,
    ValidationError,
    WayfaultError,
)
from wayfault.domain.wwr import WWRClass

__version__ = "0.2.1"

__all__ = [
    "MissingDependencyError",
    "ValidationError",
    "WWRClass",
    "WWRRequest",
    "WWRResult",
    "WayfaultError",
    "__version__",
    "estimate_wwr",
]
