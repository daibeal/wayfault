"""Lazy optional-dependency import helper."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType

from wayfault.domain.errors import MissingDependencyError


def require(package: str, extra: str) -> ModuleType:
    """Import ``package`` lazily or raise :class:`MissingDependencyError`.

    Parameters
    ----------
    package:
        Importable module name (e.g. ``"pandas"``).
    extra:
        The :mod:`wayfault` extras group that ships it (e.g. ``"io"``).
    """
    try:
        return import_module(package)
    except ImportError as exc:  # pragma: no cover - exercised via adapters
        raise MissingDependencyError(package, extra) from exc
