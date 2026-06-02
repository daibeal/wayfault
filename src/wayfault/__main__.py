"""Enable ``python -m wayfault``."""

from __future__ import annotations

from wayfault.adapters.inbound.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
