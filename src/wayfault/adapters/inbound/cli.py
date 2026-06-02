"""argparse CLI: ``python -m wayfault`` (stdlib only)."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from wayfault.adapters.inbound.api import estimate_wwr
from wayfault.adapters.outbound.credit_csv import CsvCreditCurveSource
from wayfault.adapters.outbound.dependence_copula import GaussianCopulaModel
from wayfault.adapters.outbound.dependence_hullwhite import HullWhiteHazardModel
from wayfault.adapters.outbound.dependence_independent import IndependentModel
from wayfault.adapters.outbound.exposure_csv import CsvExposureSource
from wayfault.adapters.outbound.sinks import JsonReportWriter
from wayfault.ports.outbound import DependenceModel


def _build_model(name: str, b: float, rho: float) -> DependenceModel:
    if name == "independent":
        return IndependentModel()
    if name == "hullwhite":
        return HullWhiteHazardModel(b=b)
    if name == "copula":
        return GaussianCopulaModel(rho=rho)
    raise SystemExit(f"Unknown model: {name!r}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wayfault", description="Wrong-Way Risk estimation.")
    sub = parser.add_subparsers(dest="command", required=True)

    est = sub.add_parser("estimate", help="Estimate WWR from CSV inputs.")
    est.add_argument("--exposure", required=True, help="Path to exposure cube CSV.")
    est.add_argument("--credit", required=True, help="Path to credit curve CSV.")
    est.add_argument(
        "--model",
        default="hullwhite",
        choices=["independent", "hullwhite", "copula"],
    )
    est.add_argument("--b", type=float, default=0.0, help="Hull-White WWR coupling.")
    est.add_argument("--rho", type=float, default=0.0, help="Copula correlation.")
    est.add_argument("--recovery", type=float, default=0.4, help="Recovery rate.")
    est.add_argument("--pfe-quantile", type=float, default=0.95)
    est.add_argument("--out", default=None, help="Optional JSON output path.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "estimate":
        model = _build_model(args.model, args.b, args.rho)
        sink = JsonReportWriter(args.out) if args.out else None
        result = estimate_wwr(
            exposure=CsvExposureSource(args.exposure),
            credit=CsvCreditCurveSource(args.credit, recovery=args.recovery),
            model=model,
            pfe_quantile=args.pfe_quantile,
            sink=sink,
        )
        json.dump(result.to_dict(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
