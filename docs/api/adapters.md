# Adapters

Concrete implementations of the ports. Optional dependencies are imported
lazily inside the methods that need them.

## Inbound

### Facade

::: wayfault.adapters.inbound.api

### CLI

::: wayfault.adapters.inbound.cli

## Outbound — Exposure sources

::: wayfault.adapters.outbound.exposure_inmemory

::: wayfault.adapters.outbound.exposure_csv

## Outbound — Credit sources

::: wayfault.adapters.outbound.credit_flat

::: wayfault.adapters.outbound.credit_piecewise

::: wayfault.adapters.outbound.credit_csv

## Outbound — Dependence models

::: wayfault.adapters.outbound.dependence_independent

::: wayfault.adapters.outbound.dependence_hullwhite

::: wayfault.adapters.outbound.dependence_copula

::: wayfault.adapters.outbound.dependence_archimedean

## Outbound — Calibrators

::: wayfault.adapters.outbound.calibrator_regression

::: wayfault.adapters.outbound.calibrator_sklearn

## Outbound — Sinks

::: wayfault.adapters.outbound.sinks

## Outbound — Visualization

The plotting API is documented on the dedicated
[Visualization](../visualization.md) page.
