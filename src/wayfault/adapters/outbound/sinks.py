"""Result sinks: in-memory dict and JSON report writer (stdlib json)."""

from __future__ import annotations

import json

from wayfault.application.dto import WWRResult


class DictResultSink:
    """Captures the latest result as a plain dictionary in memory."""

    def __init__(self) -> None:
        self.result: dict[str, object] | None = None

    def write(self, result: WWRResult) -> None:
        """Store ``result.to_dict()`` on the sink."""
        self.result = result.to_dict()


class JsonReportWriter:
    """Writes the result to a JSON file using the stdlib ``json`` module."""

    def __init__(self, path: str, indent: int = 2) -> None:
        self._path = path
        self._indent = indent

    def write(self, result: WWRResult) -> None:
        """Serialise ``result`` to the configured path."""
        with open(self._path, "w", encoding="utf-8") as fh:
            json.dump(result.to_dict(), fh, indent=self._indent)
