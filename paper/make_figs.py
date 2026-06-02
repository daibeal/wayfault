"""Render vector (PDF) figures for the paper from the case-study dataset."""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "docs", "assets", "data", "case_study.json")
FIGS = os.path.join(HERE, "figs")

INK, BASE, WWR, PFE, MUT = "#1b1f3b", "#4c72b0", "#c44e52", "#8172b3", "#9aa0b5"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "legend.fontsize": 7.5, "axes.grid": True, "grid.color": "#dddddd",
    "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.dpi": 150,
    "axes.spines.top": False, "axes.spines.right": False,
})


def main() -> None:
    os.makedirs(FIGS, exist_ok=True)
    d = json.load(open(DATA, encoding="utf-8"))
    t = d["tenors"]

    # Fig 1: exposure envelope + WWR conditional EE
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.plot(t, d["pfe99"], color=PFE, lw=1, ls=":", label="PFE 99%")
    ax.plot(t, d["pfe95"], color=PFE, lw=1, label="PFE 95%")
    ax.plot(t, d["epe"], color=BASE, lw=1.8, label="EPE")
    ax.plot(t, d["conditional_ee"], color=WWR, lw=1.8, label=r"cond. EE $\mid$ default")
    ax.fill_between(t, d["epe"], d["conditional_ee"], color=WWR, alpha=0.15)
    ax.set_xlabel("tenor (years)"); ax.set_ylabel("exposure")
    ax.legend(loc="upper left", ncol=1)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "exposure.pdf")); plt.close(fig)

    # Fig 2: alpha + WWR-CVA sweep over b
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    bs = [p["b"] for p in d["sweep"]]
    ax.axhline(1.0, color=MUT, lw=0.8, ls="--"); ax.axvline(0.0, color=MUT, lw=0.8, ls="--")
    ax.plot(bs, [p["alpha"] for p in d["sweep"]], color=INK, lw=1.8, marker="o", ms=2.5)
    ax.set_xlabel("wrong-way coupling $b$"); ax.set_ylabel(r"$\alpha$")
    ax2 = ax.twinx()
    ax2.plot(bs, [p["wwr_cva"] for p in d["sweep"]], color=PFE, lw=1.3, ls="--")
    ax2.set_ylabel("WWR-CVA", color=PFE); ax2.tick_params(axis="y", colors=PFE)
    ax2.grid(False)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "alpha_sweep.pdf")); plt.close(fig)

    # Fig 3: marginal-PD reproduction
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    m = d["marginal"]
    ax.plot(m["tenors"], m["target"], color=BASE, lw=2, label="curve target")
    ax.plot(m["tenors"], m["implied"], color=WWR, lw=0, marker="o", ms=4,
            markerfacecolor="none", label="model-implied")
    ax.set_xlabel("tenor (years)"); ax.set_ylabel("marginal PD")
    ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "marginal.pdf")); plt.close(fig)

    print("wrote figures to", os.path.normpath(FIGS))


if __name__ == "__main__":
    main()
