"""Diagnostic plotting adapter ([viz] extra, lazy import).

Beautiful, self-contained matplotlib visualisations of WWR results. Every
function imports :mod:`matplotlib` lazily and raises
:class:`~wayfault.domain.errors.MissingDependencyError` if the ``[viz]`` extra
is not installed, so importing :mod:`wayfault` stays numpy-only.

All functions return a ``matplotlib.figure.Figure`` and never call ``show``;
the caller decides whether to display or save.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from wayfault.adapters.outbound._optional import require
from wayfault.domain.wwr import WWRClass

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from wayfault.application.dto import WWRResult

# A cohesive, colour-blind-friendly palette.
_INK = "#1b1f3b"
_BASE = "#4c72b0"
_WWR = "#c44e52"
_RWR = "#55a868"
_PFE = "#8172b3"
_GRID = "#d9dce3"
_FILL_ALPHA = 0.16

_CLASS_COLOR = {
    WWRClass.WRONG_WAY: _WWR,
    WWRClass.RIGHT_WAY: _RWR,
    WWRClass.NEUTRAL: _BASE,
}


def _pyplot() -> Any:
    return require("matplotlib.pyplot", "viz")


def _style(plt: Any) -> dict[str, Any]:
    """A clean, modern rcParams overlay applied via ``plt.rc_context``."""
    return {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": _GRID,
        "axes.linewidth": 1.1,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": _GRID,
        "grid.linewidth": 0.8,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlecolor": _INK,
        "axes.labelcolor": _INK,
        "axes.labelsize": 11,
        "text.color": _INK,
        "xtick.color": _INK,
        "ytick.color": _INK,
        "font.size": 10,
        "legend.frameon": False,
        "figure.dpi": 130,
    }


def _despine(ax: Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_exposure_profiles(result: WWRResult, ax: Axes | None = None) -> Figure:
    """Plot EPE, conditional-EE, and PFE across the tenor grid.

    The gap between the conditional EE and the EPE is shaded to highlight the
    wrong-/right-way adjustment.

    Parameters
    ----------
    result:
        The :class:`~wayfault.application.dto.WWRResult` to visualise.
    ax:
        Optional existing axes to draw on; a new figure is created otherwise.
    """
    plt = _pyplot()
    with plt.rc_context(_style(plt)):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4.6))
        else:
            fig = ax.figure
        t = result.tenors
        cls_color = _CLASS_COLOR.get(result.classification, _WWR)

        ax.plot(t, result.pfe, color=_PFE, lw=1.6, ls="--", label="PFE (95%)")
        ax.plot(t, result.epe, color=_BASE, lw=2.4, label="EPE (independent)")
        ax.plot(t, result.conditional_ee, color=cls_color, lw=2.4,
                label="Conditional EE | default")
        ax.fill_between(t, result.epe, result.conditional_ee, color=cls_color,
                        alpha=_FILL_ALPHA, label="WWR adjustment")

        ax.set_title(f"Exposure profiles  ·  {result.model}  ·  "
                     f"{result.classification.value}")
        ax.set_xlabel("Tenor (years)")
        ax.set_ylabel("Exposure")
        ax.set_xlim(float(t[0]), float(t[-1]))
        ax.margins(y=0.08)
        ax.legend(loc="upper left", ncol=2)
        _despine(ax)
        fig.tight_layout()
    return cast("Figure", fig)


def plot_ee_ratio(result: WWRResult, ax: Axes | None = None) -> Figure:
    """Bar chart of the per-tenor conditional/unconditional EE ratio."""
    plt = _pyplot()
    with plt.rc_context(_style(plt)):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4.2))
        else:
            fig = ax.figure
        t = result.tenors
        ratio = result.ee_ratio
        colors = [_WWR if r >= 1.0 else _RWR for r in ratio]
        width = float(np.min(np.diff(t))) * 0.8 if t.size > 1 else 0.2
        ax.bar(t, ratio, width=width, color=colors, alpha=0.85,
               edgecolor="white", linewidth=0.6)
        ax.axhline(1.0, color=_INK, lw=1.1, ls=":")
        ax.set_title("Conditional / unconditional EE ratio")
        ax.set_xlabel("Tenor (years)")
        ax.set_ylabel("ratio")
        ax.margins(y=0.12)
        _despine(ax)
        fig.tight_layout()
    return cast("Figure", fig)


def plot_alpha_sweep(
    bs: Sequence[float],
    alphas: Sequence[float],
    wwr_cvas: Sequence[float] | None = None,
    baseline_cva: float | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Plot the alpha multiplier (and optionally CVA) against the WWR knob.

    Parameters
    ----------
    bs:
        The swept dependence parameters (Hull-White ``b`` or copula ``rho``).
    alphas:
        The corresponding alpha multipliers.
    wwr_cvas:
        Optional WWR-CVA values, drawn on a secondary axis.
    baseline_cva:
        Optional baseline CVA, drawn as a reference line on the CVA axis.
    ax:
        Optional existing axes.
    """
    plt = _pyplot()
    bs_a = np.asarray(bs, dtype=float)
    alphas_a = np.asarray(alphas, dtype=float)
    with plt.rc_context(_style(plt)):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4.6))
        else:
            fig = ax.figure
        ax.axhline(1.0, color=_GRID, lw=1.2)
        ax.axvline(0.0, color=_GRID, lw=1.2)
        ax.plot(bs_a, alphas_a, color=_INK, lw=2.4, marker="o", ms=5,
                label="alpha")
        wrong = (bs_a > 0).tolist()
        right = (bs_a <= 0).tolist()
        ax.fill_between(bs_a, 1.0, alphas_a, where=wrong, color=_WWR,
                        alpha=_FILL_ALPHA)
        ax.fill_between(bs_a, 1.0, alphas_a, where=right, color=_RWR,
                        alpha=_FILL_ALPHA)
        ax.set_title("Alpha multiplier vs dependence knob")
        ax.set_xlabel("dependence parameter  (b / rho)")
        ax.set_ylabel("alpha = WWR-CVA / baseline-CVA")
        _despine(ax)

        if wwr_cvas is not None:
            ax2 = ax.twinx()
            ax2.plot(bs_a, np.asarray(wwr_cvas, dtype=float), color=_PFE,
                     lw=1.8, ls="--", marker="s", ms=4, label="WWR-CVA")
            if baseline_cva is not None:
                ax2.axhline(baseline_cva, color=_BASE, lw=1.2, ls=":",
                            label="baseline CVA")
            ax2.set_ylabel("CVA", color=_PFE)
            ax2.tick_params(axis="y", colors=_PFE)
            ax2.spines["top"].set_visible(False)
            ax2.grid(False)
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        else:
            ax.legend(loc="upper left")
        fig.tight_layout()
    return cast("Figure", fig)


def plot_dashboard(
    result: WWRResult,
    bs: Sequence[float] | None = None,
    alphas: Sequence[float] | None = None,
    wwr_cvas: Sequence[float] | None = None,
) -> Figure:
    """A 2x2 summary dashboard for a single WWR result.

    Combines the exposure profiles, the EE ratio, a CVA comparison bar, and —
    when a sweep is provided — the alpha curve.
    """
    plt = _pyplot()
    with plt.rc_context(_style(plt)):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        plot_exposure_profiles(result, ax=axes[0, 0])
        plot_ee_ratio(result, ax=axes[0, 1])

        # CVA comparison bar.
        ax = axes[1, 0]
        labels = ["baseline", "WWR"]
        vals = [result.baseline_cva, result.wwr_cva]
        cls_color = _CLASS_COLOR.get(result.classification, _WWR)
        bars = ax.bar(labels, vals, color=[_BASE, cls_color], alpha=0.9,
                      edgecolor="white", linewidth=0.8, width=0.6)
        for rect, v in zip(bars, vals, strict=False):
            ax.text(rect.get_x() + rect.get_width() / 2, v, f"{v:.4f}",
                    ha="center", va="bottom", fontsize=10, color=_INK)
        ax.set_title(f"CVA  ·  alpha = {result.alpha:.3f}  "
                     f"({result.uplift_pct:+.1f}%)")
        ax.set_ylabel("CVA")
        ax.margins(y=0.18)
        _despine(ax)

        # Alpha sweep (or a placeholder note).
        ax = axes[1, 1]
        if bs is not None and alphas is not None:
            plot_alpha_sweep(bs, alphas, wwr_cvas, result.baseline_cva, ax=ax)
        else:
            ax.axis("off")
            ax.text(0.5, 0.5, "pass bs / alphas\nfor the alpha sweep",
                    ha="center", va="center", color=_GRID, fontsize=12)

        fig.suptitle(f"wayfault — {result.model}  ·  "
                     f"{result.classification.value}", fontsize=15,
                     fontweight="bold", color=_INK)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
    return cast("Figure", fig)


def save(fig: Figure, path: str, dpi: int = 130) -> None:
    """Save a figure to ``path`` (PNG/SVG/PDF inferred from the extension)."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
