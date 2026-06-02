"""Domain exceptions for :mod:`wayfault`.

These exceptions are part of the pure domain layer and depend only on the
standard library.
"""

from __future__ import annotations


class WayfaultError(Exception):
    """Base class for all :mod:`wayfault` errors."""


class ValidationError(WayfaultError):
    """Raised when a value object fails its construction-time invariants."""


class MissingDependencyError(WayfaultError):
    """Raised when an optional adapter is used without its extra installed.

    Parameters
    ----------
    package:
        The importable package that was missing (e.g. ``"pandas"``).
    extra:
        The :mod:`wayfault` extras group that provides it (e.g. ``"io"``).
    """

    def __init__(self, package: str, extra: str) -> None:
        self.package = package
        self.extra = extra
        super().__init__(
            f"Optional dependency {package!r} is required for this feature. "
            f"Install it with: pip install 'wayfault[{extra}]'"
        )
