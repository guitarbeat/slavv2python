#!/usr/bin/env python3
"""Journal multipanel figure: quantitative MATLAB↔Python exact-parity results.

Three pure-data panels (methods / appendix figure):
  (a) Crop candidate-pair overlap trajectory
  (b) MATLAB edge-pair recovery waterfall (crop, post-fix)
  (c) Canonical-volume counts (MATLAB vs Python) + certified spatial/float bars

Evidence: docs/reference/core/EXACT_PROOF_FINDINGS.md
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.55,
        "xtick.major.width": 0.55,
        "ytick.major.width": 0.55,
        "xtick.major.size": 2.2,
        "ytick.major.size": 2.2,
        "lines.linewidth": 0.9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 600,
    }
)

INK = "#111111"
MUTED = "#555555"
MID = "#BDBDBD"
LIGHT = "#EDEDED"
LINE = "#222222"
ACCENT = "#1F4E79"
MATLAB = "#4A4A4A"
PYTHON = "#1F4E79"
LOSS = "#8A8A8A"


def _tag(ax, letter: str, x: float = -0.08, y: float = 1.04):
    ax.text(
        x,
        y,
        f"({letter})",
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
        ha="left",
        color=INK,
        clip_on=False,
    )


def draw_a(ax):
    _tag(ax, "a")
    ax.set_title(
        "Crop harness: MATLAB final pairs present in Python candidates",
        loc="left",
        pad=3,
        fontweight="bold",
        fontsize=8,
    )

    labels = ["Baseline\nfrontier", "List queue\n(reverted)", "Sorted\nfrontier", "LUT +\nsuppression"]
    y = np.array([57.89, 11.56, 57.89, 97.31])
    x = np.arange(len(y))

    ax.axhspan(80, 115, facecolor=LIGHT, edgecolor="none", zorder=0)
    ax.axhline(80.0, color=INK, linestyle=(0, (3, 2)), linewidth=0.65, zorder=2)
    ax.text(3.32, 82.0, "80% gate", ha="right", va="bottom", fontsize=6.3, color=INK)

    ax.plot(
        x,
        y,
        color=ACCENT,
        marker="o",
        markersize=5.5,
        markerfacecolor="white",
        markeredgecolor=ACCENT,
        markeredgewidth=1.15,
        linewidth=1.15,
        zorder=3,
    )

    for i, v in enumerate(y):
        if v < 20:
            ax.text(i, v + 4.5, f"{v:.1f}%", ha="center", va="bottom", fontsize=6.5, color=INK)
        elif i == 3:
            ax.text(i, v - 6.5, f"{v:.1f}%", ha="center", va="top", fontsize=6.5, color=INK)
        else:
            ax.text(i, v + 3.0, f"{v:.1f}%", ha="center", va="bottom", fontsize=6.5, color=INK)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel("Overlap (%)")
    ax.set_ylim(0, 115)
    ax.set_xlim(-0.35, 3.35)
    ax.yaxis.set_major_locator(plt.MultipleLocator(20))

    ax.text(
        0.0,
        -0.18,
        "n = 15,511 MATLAB final edge pairs (oracle 180709_E_crop_M_v2).",
        transform=ax.transAxes,
        fontsize=5.8,
        color=MUTED,
        ha="left",
        va="top",
        clip_on=False,
    )


def draw_b(ax):
    """Classic waterfall: MATLAB edge-pair recovery on crop (post-fix)."""
    _tag(ax, "b")
    ax.set_title(
        "Crop: recovery of MATLAB final edge pairs after generation fixes",
        loc="left",
        pad=3,
        fontweight="bold",
        fontsize=8,
    )

    # 15,511 → (−417 generation) → 15,094 → (−980 crop/residual) → 14,114
    heights = np.array([15511, 417, 15094, 980, 14114], dtype=float)
    bottoms = np.array([0, 15094, 0, 14114, 0], dtype=float)
    colors = [ACCENT, LOSS, ACCENT, LOSS, ACCENT]
    labels = [
        "MATLAB\nfinal edges",
        "Generation\ngap",
        "Present as\nPython candidate",
        "Crop +\nresidual selection",
        "Matched in\nPython final set",
    ]

    x = np.arange(5)
    ax.bar(
        x,
        heights,
        bottom=bottoms,
        width=0.58,
        color=colors,
        edgecolor=LINE,
        linewidth=0.45,
        zorder=3,
    )

    # Totals above solid bars; signed losses on floating bars
    for i, t in ((0, 15511), (2, 15094), (4, 14114)):
        ax.text(i, t + 160, f"{t:,}", ha="center", va="bottom", fontsize=6.5, color=INK)
    ax.text(1, 15094 + 417 / 2, "-417", ha="center", va="center", fontsize=6.4, color="white", fontweight="bold")
    ax.text(3, 14114 + 980 / 2, "-980", ha="center", va="center", fontsize=6.4, color="white", fontweight="bold")

    # Thin connectors at running totals
    for x0, y in ((0.29, 15511), (1.29, 15094), (2.29, 15094), (3.29, 14114)):
        ax.plot([x0, x0 + 0.42], [y, y], color=MUTED, lw=0.55, zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.0)
    ax.set_ylabel("MATLAB edge pairs retained")
    ax.set_ylim(10000, 17000)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))

    handles = [
        mpatches.Patch(facecolor=ACCENT, edgecolor=LINE, linewidth=0.35, label="Pairs retained"),
        mpatches.Patch(facecolor=LOSS, edgecolor=LINE, linewidth=0.35, label="Pairs lost at step"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper right", fontsize=6.0, handlelength=1.0)

    ax.text(
        0.0,
        -0.22,
        "Post-fix crop (2026-07-08). Python final edge count = 14,922 (includes pairs absent from MATLAB).",
        transform=ax.transAxes,
        fontsize=5.7,
        color=MUTED,
        ha="left",
        va="top",
        clip_on=False,
    )


def draw_c(ax_left, ax_right):
    """Split panel: absolute counts | certified metrics table."""
    _tag(ax_left, "c", x=-0.12, y=1.04)

    # --- left: grouped bars ---
    ax_left.set_title(
        "Canonical volume 180709_E: strict stage counts",
        loc="left",
        pad=3,
        fontweight="bold",
        fontsize=8,
    )

    categories = ["Edge\nconnections", "Network\nstrands"]
    matlab = np.array([69500, 48049])
    python = np.array([65436, 44595])
    x = np.arange(len(categories))
    w = 0.32

    ax_left.bar(
        x - w / 2,
        matlab,
        w,
        color=MATLAB,
        edgecolor=LINE,
        linewidth=0.45,
        label="MATLAB",
        zorder=3,
    )
    ax_left.bar(
        x + w / 2,
        python,
        w,
        color=PYTHON,
        edgecolor=LINE,
        linewidth=0.45,
        label="Python",
        zorder=3,
    )

    for i, (m, p) in enumerate(zip(matlab, python)):
        ax_left.text(i - w / 2, m + 900, f"{m:,}", ha="center", va="bottom", fontsize=5.9, color=INK)
        ax_left.text(i + w / 2, p + 900, f"{p:,}", ha="center", va="bottom", fontsize=5.9, color=INK)
        gap = m - p
        # Delta between the paired bars, mid-height
        ax_left.annotate(
            f"-{gap:,}\n({100 * p / m:.1f}%)",
            xy=(i, (m + p) / 2),
            xytext=(i + 0.42, (m + p) / 2),
            fontsize=5.8,
            color=MUTED,
            ha="left",
            va="center",
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5),
        )

    ax_left.set_xticks(x)
    ax_left.set_xticklabels(categories, fontsize=7)
    ax_left.set_ylabel("Count")
    ax_left.set_ylim(0, 82000)
    # room for delta annotations drawn below baseline
    ax_left.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{int(v / 1000)}k" if v >= 1000 else f"{int(v)}")
    )
    ax_left.legend(frameon=False, loc="upper right", fontsize=6.3, handlelength=1.0)

    ax_left.text(
        0.0,
        -0.18,
        "Stretch metrics. Network deficit tracks residual edge-generation gap (no independent network bug).",
        transform=ax_left.transAxes,
        fontsize=5.6,
        color=MUTED,
        ha="left",
        va="top",
        clip_on=False,
    )

    # --- right: certified metrics as clean table ---
    ax_right.set_xlim(0, 1)
    ax_right.set_ylim(0, 1)
    ax_right.axis("off")
    ax_right.set_title(
        "Certification metrics (same volume)",
        loc="left",
        pad=3,
        fontweight="bold",
        fontsize=8,
    )

    rows = [
        ("Stage", "Metric", "Result"),
        ("Energy", "scale-index mismatches", "0 / 16.8M"),
        ("Energy", "max |d| float fields", "~2e-11"),
        ("Vertices", "position / scale match", "exact"),
        ("Edges", "ownership-map agreement", "96.0%"),
        ("Edges", "ADR 0012 ownership bar", "pass (>=60%)"),
        ("Network", "topology given MATLAB edges", "exact*"),
    ]

    y0 = 0.90
    col_x = [0.02, 0.24, 0.70]
    for j, h in enumerate(rows[0]):
        ax_right.text(
            col_x[j],
            y0,
            h,
            fontsize=6.4,
            fontweight="bold",
            color=INK,
            ha="left",
            va="center",
            transform=ax_right.transAxes,
        )
    ax_right.plot([0.02, 0.98], [0.84, 0.84], color=LINE, lw=0.55, transform=ax_right.transAxes, clip_on=False)

    for i, (stage, metric, result) in enumerate(rows[1:]):
        y = 0.74 - i * 0.10
        ax_right.text(col_x[0], y, stage, fontsize=6.2, color=INK, ha="left", va="center", transform=ax_right.transAxes)
        ax_right.text(col_x[1], y, metric, fontsize=6.1, color=MUTED, ha="left", va="center", transform=ax_right.transAxes)
        ax_right.text(
            col_x[2],
            y,
            result,
            fontsize=6.2,
            color=INK,
            ha="left",
            va="center",
            fontweight="bold",
            transform=ax_right.transAxes,
        )
        if i < len(rows) - 2:
            ax_right.plot(
                [0.02, 0.98],
                [y - 0.05, y - 0.05],
                color=LIGHT,
                lw=0.5,
                transform=ax_right.transAxes,
                clip_on=False,
            )

    ax_right.text(
        0.02,
        0.03,
        "*Isolated network with MATLAB edges matches\n"
        "topology exactly. Residual network fail is\n"
        "downstream of the edge-count gap only.",
        fontsize=5.5,
        color=MUTED,
        ha="left",
        va="bottom",
        transform=ax_right.transAxes,
        linespacing=1.35,
    )


def main() -> Path:
    out_dir = Path(__file__).resolve().parent

    fig = plt.figure(figsize=(7.2, 8.4))
    gs = fig.add_gridspec(
        3,
        1,
        left=0.10,
        right=0.98,
        top=0.97,
        bottom=0.04,
        hspace=0.45,
        height_ratios=[1.0, 1.05, 1.15],
    )

    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])

    # Bottom row split 55/45
    gs_c = gs[2].subgridspec(1, 2, wspace=0.18, width_ratios=[1.15, 1.0])
    ax_c_left = fig.add_subplot(gs_c[0, 0])
    ax_c_right = fig.add_subplot(gs_c[0, 1])

    draw_a(ax_a)
    draw_b(ax_b)
    draw_c(ax_c_left, ax_c_right)

    png = out_dir / "matlab_python_parity_journey.png"
    pdf = out_dir / "matlab_python_parity_journey.pdf"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return png


if __name__ == "__main__":
    path = main()
    print(f"Wrote {path}")
    print(f"Wrote {path.with_suffix('.pdf')}")
