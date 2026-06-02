# Paper — `wayfault`

A two-column working-draft paper on the library and its prescriptive
inverse-calibration contribution.

- `wayfault.tex` — the paper (standard `article` twocolumn; only common
  packages: amsmath, amssymb, amsthm, mathtools, graphicx, booktabs, hyperref).
- `make_figs.py` — regenerates the vector figures from the real case-study data
  (`docs/assets/data/case_study.json`).
- `figs/` — generated PDF figures.

## Build

```bash
python paper/make_figs.py          # (re)generate figs/*.pdf  (needs wayfault[viz])
cd paper
pdflatex -interaction=nonstopmode wayfault.tex
pdflatex -interaction=nonstopmode wayfault.tex   # 2nd pass for references
```

### Easiest: Overleaf (no local LaTeX needed)
Upload the `paper/` folder (the `.tex` and the `figs/` directory) to a new
[Overleaf](https://overleaf.com) project and compile — zero setup.

> Note: on machines with a Device Guard / WDAC allowlisting policy, local
> MiKTeX/TeX Live executables may be blocked from running. Overleaf or an
> unmanaged machine avoids this.

## Math verification

Every structural identity in the paper is machine-checked symbolically with
SymPy in [`tests/test_symbolic_proofs.py`](../tests/test_symbolic_proofs.py)
(13 proofs): the Clayton/Frank h-functions as exact copula derivatives,
survival/PD telescoping, the piecewise cumulative-hazard integral, the
marginal-consistency identity, softmax normalisation, the Gaussian threshold,
the CVA integrand, and the score (covariance) identity behind the monotonicity
theorem.

```bash
pip install -e '.[dev]'
pytest tests/test_symbolic_proofs.py -v
```
