# Command-Line Interface

`wayfault` ships a small CLI built on the standard-library `argparse` (no
`click`/`typer`). CSV ingestion requires the `[io]` extra.

```bash
pip install 'wayfault[io]'
```

## Usage

```bash
python -m wayfault estimate \
    --exposure cube.csv \
    --credit curve.csv \
    --model hullwhite --b 0.5 \
    --out result.json
```

A console entry point named `wayfault` is also installed:

```bash
wayfault estimate --exposure cube.csv --credit curve.csv --model copula --rho 0.6
```

## Options

| Flag             | Default     | Description                                   |
|------------------|-------------|-----------------------------------------------|
| `--exposure`     | *required*  | Path to the exposure cube CSV.                |
| `--credit`       | *required*  | Path to the credit curve CSV.                 |
| `--model`        | `hullwhite` | `independent`, `hullwhite`, or `copula`.      |
| `--b`            | `0.0`       | Hull-White WWR coupling.                       |
| `--rho`          | `0.0`       | Gaussian-copula correlation.                   |
| `--recovery`     | `0.4`       | Recovery rate.                                 |
| `--pfe-quantile` | `0.95`      | Quantile for the PFE profile.                  |
| `--out`          | *(stdout)*  | Optional JSON output path.                      |

The result is always printed to stdout as JSON; `--out` additionally writes it
to a file.

## Input file formats

**Exposure cube CSV** — column headers are tenor year-fractions, each row a
Monte-Carlo scenario:

```csv
0.5,1.0,1.5,2.0
1.21,1.55,1.80,2.10
0.98,1.33,1.71,1.95
...
```

**Credit curve CSV** — `knot` / `hazard` columns, one row per piecewise segment:

```csv
knot,hazard
1.0,0.02
3.0,0.03
```

The recovery rate is passed via `--recovery` (not in the file).
