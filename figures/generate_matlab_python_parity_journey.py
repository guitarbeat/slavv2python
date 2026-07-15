#!/usr/bin/env python3
"""Standalone parity figures with claims, not dashboards.

Each figure answers one non-trivial question from the exact-parity campaign.
Constants tracked against docs/reference/core/EXACT_PROOF_FINDINGS.md.

  parity_trajectory   — Which fix actually moved candidate recovery?
  parity_funnel       — How did the crop residual collapse (missing vs extra)?
  parity_agreement    — Did full-volume Edges under- then over-select?
  parity_cert_table   — Where does residual remain, in absolute counts?
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.55,
        "xtick.major.width": 0.55,
        "ytick.major.width": 0.55,
        "xtick.major.size": 2.4,
        "ytick.major.size": 2.4,
        "lines.linewidth": 1.15,
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
GREEN = "#2E7D32"
RED = "#C62828"
AMBER = "#E65100"
STEEL = "#455A64"
TEAL = "#00695C"


def _claim(ax: plt.Axes, text: str) -> None:
    """Left-aligned bold title used as the figure's one-sentence claim."""
    ax.set_title(text, loc="left", pad=8, fontweight="bold", fontsize=10, color=INK)


def _footnote(ax: plt.Axes, text: str, y: float = -0.14) -> None:
    ax.text(
        0.0,
        y,
        text,
        transform=ax.transAxes,
        fontsize=6.8,
        color=MUTED,
        ha="left",
        va="top",
        clip_on=False,
        linespacing=1.35,
    )


# ---------------------------------------------------------------------------
# 1. Trajectory — missing pairs (log), not a flat % climb
# ---------------------------------------------------------------------------
def draw_trajectory(ax: plt.Axes) -> None:
    """Only one step recovered ~6k pairs; queue-order cosmetics did nothing."""
    _claim(ax, "One directional-LUT fix recovered ~6,000 missing MATLAB edges")

    # (short label, missing MATLAB final pairs among Python candidates, kind)
    # kind: null | leap | polish | closed
    steps = [
        ("Baseline\nfrontier", 6532, "null"),
        ("Sorted queue\n(no effect)", 6532, "null"),
        ("Directional LUT\n+ suppression", 417, "leap"),
        ("Vertex -Inf\nsentinel", 19, "polish"),
        ("Trace match", 0, "closed"),
    ]
    labels = [s[0] for s in steps]
    missing = np.array([s[1] for s in steps], dtype=float)
    kinds = [s[2] for s in steps]
    x = np.arange(len(steps))

    # Plot missing+1 for log scale so zero is visible as floor
    y_plot = missing + 1.0
    ax.set_yscale("log")
    ax.set_ylim(0.7, 2.0e4)
    ax.set_xlim(-0.45, 4.55)

    # 80% gate as missing-pair equivalent: 20% of 15511 ≈ 3102
    gate_missing = 15511 * 0.20
    ax.axhline(gate_missing, color=MID, ls=(0, (3, 2)), lw=0.9, zorder=1)
    ax.text(
        4.5,
        gate_missing * 1.12,
        "retired 80% gate\n(~3,102 still missing)",
        ha="right",
        va="bottom",
        fontsize=6.5,
        color=MUTED,
        linespacing=1.2,
    )

    color_map = {"null": MID, "leap": ACCENT, "polish": TEAL, "closed": GREEN}
    for i in range(len(x) - 1):
        ax.plot(
            [x[i], x[i + 1]],
            [y_plot[i], y_plot[i + 1]],
            color=color_map[kinds[i + 1]] if kinds[i + 1] == "leap" else ACCENT,
            lw=1.6 if kinds[i + 1] == "leap" else 1.1,
            zorder=2,
            solid_capstyle="round",
        )

    for i, (yi, k) in enumerate(zip(y_plot, kinds)):
        ax.plot(
            x[i],
            yi,
            "o",
            ms=8 if k == "leap" else 6.5,
            color=color_map[k],
            markeredgecolor=LINE,
            markeredgewidth=0.45,
            zorder=4,
        )

    # Value labels — keep clear of the closing point
    for i, m in enumerate(missing):
        if i == len(missing) - 1:
            continue  # handled below
        ax.text(
            x[i],
            y_plot[i] * 1.55,
            f"{int(m):,} missing",
            ha="center",
            va="bottom",
            fontsize=7.2,
            color=INK,
            fontweight="bold" if kinds[i] == "leap" else "normal",
        )
    ax.text(
        x[-1],
        2.8,
        "0 missing\n(gen. closed)",
        ha="center",
        va="bottom",
        fontsize=7.2,
        color=GREEN,
        fontweight="bold",
        linespacing=1.15,
    )

    # Delta annotation on the leap only
    ax.annotate(
        "\u22126,115 pairs\nin one step",
        xy=(2, y_plot[2]),
        xytext=(2.85, 2500),
        fontsize=7.5,
        color=ACCENT,
        fontweight="bold",
        ha="left",
        arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.9),
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=ACCENT, lw=0.55),
    )
    ax.annotate(
        "queue cosmetics:\nno recovery",
        xy=(0.5, y_plot[0]),
        xytext=(0.5, 1200),
        fontsize=6.5,
        color=MUTED,
        ha="center",
        arrowprops=dict(arrowstyle="->", color=MID, lw=0.7),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.2)
    ax.set_ylabel("MATLAB final pairs still absent\nfrom Python candidates (log)")
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: "0" if v <= 1 else f"{int(v):,}")
    )

    _footnote(
        ax,
        "Crop harness n = 15,511 MATLAB final pairs (oracle 180709_E_crop_M_v2). "
        "Y shows generation residual (pairs never emitted as candidates), not final selection.",
        y=-0.17,
    )


# ---------------------------------------------------------------------------
# 2. Residual collapse — missing vs extra (has shape, conflict, resolution)
# ---------------------------------------------------------------------------
def draw_funnel(ax: plt.Axes) -> None:
    """Crop final residual: missing and extra tell different stories."""
    _claim(ax, "Crop residual collapsed from thousands to a single pair swap")

    # Published checkpoints only (label, missing, extra)
    phases = [
        ("Baseline\ngeneration", 6532, np.nan),  # generation gap; final extras not the story
        ("LUT unlocked\ncandidates", 417, 3772),  # ~19,283 candidates − 15,511
        ("Trace match +\nover-select", 149, 365),  # final missing/extra after gen closed
        ("Post-watershed\nfinalization", 1, 1),
    ]

    labels = [p[0] for p in phases]
    missing = np.array([p[1] for p in phases], dtype=float)
    extra = np.array([p[2] for p in phases], dtype=float)
    x = np.arange(len(phases))
    w = 0.36

    # Stacked narrative with log scale for dynamic range
    ax.set_yscale("log")
    ax.set_ylim(0.7, 1.2e4)

    # Missing bars (left)
    ax.bar(
        x - w / 2,
        np.maximum(missing, 0.85),
        width=w,
        color=RED,
        edgecolor=LINE,
        linewidth=0.4,
        label="Missing MATLAB pairs",
        zorder=3,
    )
    # Extra bars (right) — skip NaN
    extra_plot = np.where(np.isnan(extra), 0.0, np.maximum(extra, 0.85))
    bars_e = ax.bar(
        x + w / 2,
        extra_plot,
        width=w,
        color=AMBER,
        edgecolor=LINE,
        linewidth=0.4,
        label="Extra Python pairs",
        zorder=3,
    )
    # Grey out NaN extra at baseline
    for i, e in enumerate(extra):
        if np.isnan(e):
            bars_e[i].set_alpha(0.0)

    for i, (m, e) in enumerate(zip(missing, extra)):
        ax.text(
            x[i] - w / 2,
            max(m, 0.85) * 1.18,
            f"{int(m):,}",
            ha="center",
            va="bottom",
            fontsize=7.0,
            color=RED,
            fontweight="bold",
        )
        if not np.isnan(e):
            ax.text(
                x[i] + w / 2,
                max(e, 0.85) * 1.18,
                f"{int(e):,}",
                ha="center",
                va="bottom",
                fontsize=7.0,
                color=AMBER,
                fontweight="bold",
            )
        else:
            ax.text(
                x[i] + w / 2,
                2.2,
                "n/a",
                ha="center",
                va="bottom",
                fontsize=6.5,
                color=MID,
                style="italic",
            )

    # Callout on the interesting middle: extras dominate once generation is closed
    ax.annotate(
        "Once generation closed,\nextras displaced MATLAB\npairs in faithful cleanup",
        xy=(2, 365),
        xytext=(1.15, 40),
        fontsize=6.8,
        color=MUTED,
        ha="center",
        arrowprops=dict(arrowstyle="->", color=MID, lw=0.75),
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=MID, lw=0.45),
    )
    ax.annotate(
        "equal-count\n1-pair swap",
        xy=(3, 1),
        xytext=(3.35, 12),
        fontsize=7.0,
        color=GREEN,
        fontweight="bold",
        ha="left",
        arrowprops=dict(arrowstyle="->", color=GREEN, lw=0.85),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.2)
    ax.set_ylabel("Final residual pair count (log)")
    ax.legend(frameon=False, loc="upper right", fontsize=7.2)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: "0" if v < 1.1 else f"{int(round(v)):,}")
    )

    _footnote(
        ax,
        "Crop final residual vs MATLAB oracle pairs (180709_E_crop_M_v2). "
        "Middle columns use published checkpoints: generation extras ≈ candidates − 15,511; "
        "later columns are final missing/extra after selection.",
        y=-0.16,
    )


# ---------------------------------------------------------------------------
# 3. Full-volume signed residual — under → over → closed
# ---------------------------------------------------------------------------
def draw_agreement(ax: plt.Axes) -> None:
    """Signed edge residual across canonical audits — the non-monotonic story."""
    _claim(ax, "Full-volume Edges under-selected, then over-selected, then matched")

    # (run label, Python−MATLAB edge connections, network strand residual, note)
    runs = [
        ("v4", -9287, -8426, "pre-fix\naudit"),
        ("v6", -4064, -3454, "Edges ADR\nPASS; Net FAIL"),
        ("v7", -3276, -2632, "generation\nimproved"),
        ("v10", +747, +534, "sign flip\n(over-select)"),
        ("v15", 0, -1, "edges exact;\n1 strand"),
        ("v16", 0, -1, "ADR 0012\nbars PASS"),
    ]
    labels = [r[0] for r in runs]
    edge_d = np.array([r[1] for r in runs], dtype=float)
    net_d = np.array([r[2] for r in runs], dtype=float)
    x = np.arange(len(runs))

    ax.axhline(0, color=INK, lw=0.85, zorder=1)
    ax.fill_between([-0.6, len(runs) - 0.4], 0, 3200, color="#E8F5E9", alpha=0.55, zorder=0)
    ax.fill_between([-0.6, len(runs) - 0.4], -11000, 0, color="#FFEBEE", alpha=0.45, zorder=0)
    ax.text(
        5.35,
        2200,
        "Python over",
        ha="right",
        va="center",
        fontsize=6.5,
        color=GREEN,
        fontweight="bold",
    )
    ax.text(
        5.35,
        -9800,
        "Python under",
        ha="right",
        va="center",
        fontsize=6.5,
        color=RED,
        fontweight="bold",
    )

    ax.plot(
        x,
        edge_d,
        color=ACCENT,
        marker="o",
        ms=7,
        markerfacecolor="white",
        markeredgecolor=ACCENT,
        markeredgewidth=1.2,
        lw=1.4,
        zorder=3,
        label="Edges: Python − MATLAB connections",
    )
    ax.plot(
        x,
        net_d,
        color=STEEL,
        marker="s",
        ms=5.5,
        markerfacecolor="white",
        markeredgecolor=STEEL,
        markeredgewidth=1.0,
        lw=1.15,
        ls=(0, (4, 2)),
        zorder=3,
        label="Network: Python − MATLAB strands",
    )

    for i, ed in enumerate(edge_d):
        sign = f"{ed:+,.0f}" if ed != 0 else "0"
        # Offset up for non-negative, down for under; pull v15/v16 slightly left of line
        if i >= 4:
            ax.text(
                x[i] + 0.08,
                ed + 350,
                sign if ed != 0 else "edges 0",
                ha="left",
                va="bottom",
                fontsize=6.8,
                color=ACCENT,
                fontweight="bold",
            )
        else:
            ax.text(
                x[i],
                ed + (500 if ed >= 0 else -500),
                sign,
                ha="center",
                va="bottom" if ed >= 0 else "top",
                fontsize=7.0,
                color=ACCENT,
                fontweight="bold",
            )
    # Network −1 residual at the end (easy to miss)
    ax.text(
        x[-1] + 0.08,
        net_d[-1] - 550,
        "net \u22121",
        ha="left",
        va="top",
        fontsize=6.8,
        color=STEEL,
        fontweight="bold",
    )

    ax.annotate(
        "ownership PASS\nwhile still \u22124k edges",
        xy=(1, edge_d[1]),
        xytext=(0.05, -7000),
        fontsize=6.6,
        color=MUTED,
        ha="left",
        arrowprops=dict(arrowstyle="->", color=MID, lw=0.7),
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=MID, lw=0.4),
    )
    ax.annotate(
        "axis/finalization fix\nflipped the sign",
        xy=(3, edge_d[3]),
        xytext=(3.45, 2400),
        fontsize=6.6,
        color=AMBER,
        ha="left",
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=AMBER, lw=0.85),
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=AMBER, lw=0.5),
    )
    ax.annotate(
        "Network tracks Edges\n(no independent bug)",
        xy=(2.05, net_d[2]),
        xytext=(2.55, -8200),
        fontsize=6.6,
        color=STEEL,
        ha="center",
        arrowprops=dict(arrowstyle="->", color=STEEL, lw=0.7),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{r[0]}\n{r[3]}" for r in runs],
        fontsize=6.5,
    )
    ax.set_ylabel("Signed residual count\n(Python \u2212 MATLAB)")
    ax.set_xlim(-0.55, len(runs) - 0.15)
    ax.set_ylim(-11000, 3500)
    ax.legend(frameon=False, loc="center right", fontsize=6.6)

    _footnote(
        ax,
        "Canonical full 180709_E audits (MATLAB edges = 69,500; strands = 48,049). "
        "v15/v16 edge residual 0; Network still \u22121 strand under ADR 0012 multiset bar (PASS).",
        y=-0.18,
    )


# ---------------------------------------------------------------------------
# 4. Mismatch budget — absolute residuals that matter
# ---------------------------------------------------------------------------
def draw_cert_table(ax: plt.Axes) -> None:
    """Absolute remaining mismatches — not a wall of PASS."""
    _claim(ax, "On 180M voxels the residual is one edge-pair selection swap")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Rows: (stage, quantity, residual, denom, interpretation)
    rows = [
        ("Energy scale-index", "mismatched voxels", 0, "16.8M", "exact discrete field"),
        ("Energy float", "max |Δ|", "~2e-11", "allclose", "library FP drift only"),
        ("Vertices", "position / scale", 0, "exact", "bit-identical geometry"),
        ("Edges ownership", "disagreed voxels", "~8", "5.84M claimed", "99.9999% map agree"),
        ("Edges connections", "pair multiset Δ", 1, "69,500", "one equal-metric swap"),
        ("Network strands", "strand multiset Δ", 1, "48,049", "downstream of that swap"),
    ]

    # Header
    headers = ("Stage / surface", "What is counted", "Residual", "Of", "Reading")
    col_x = [0.01, 0.28, 0.52, 0.64, 0.76]
    y_head = 0.92
    for j, h in enumerate(headers):
        ax.text(
            col_x[j],
            y_head,
            h,
            fontsize=7.0,
            fontweight="bold",
            color=INK,
            ha="left",
            va="center",
            transform=ax.transAxes,
        )
    ax.plot([0.01, 0.99], [0.875, 0.875], color=LINE, lw=0.65, transform=ax.transAxes)

    row_h = 0.105
    for i, (stage, what, residual, denom, reading) in enumerate(rows):
        yi = 0.80 - i * row_h
        is_hot = residual == 1 or residual == "1"
        if i % 2 == 0:
            ax.axhspan(yi - row_h * 0.42, yi + row_h * 0.42, facecolor=LIGHT, edgecolor="none")
        if is_hot:
            # Highlight residual rows
            rect = FancyBboxPatch(
                (0.005, yi - row_h * 0.42),
                0.99,
                row_h * 0.84,
                boxstyle="round,pad=0.004,rounding_size=0.008",
                facecolor="#FFF8E1",
                edgecolor=AMBER,
                linewidth=0.7,
                transform=ax.transAxes,
                zorder=0,
            )
            ax.add_patch(rect)

        ax.text(col_x[0], yi, stage, fontsize=7.0, color=INK, ha="left", va="center",
                transform=ax.transAxes, fontweight="bold" if is_hot else "normal")
        ax.text(col_x[1], yi, what, fontsize=6.5, color=MUTED, ha="left", va="center",
                transform=ax.transAxes)
        res_s = f"{residual}" if not isinstance(residual, int) else f"{residual:,}"
        res_color = AMBER if is_hot else (GREEN if residual == 0 or residual == "~2e-11" else INK)
        ax.text(
            col_x[2],
            yi,
            res_s,
            fontsize=7.5,
            color=res_color,
            ha="left",
            va="center",
            transform=ax.transAxes,
            fontweight="bold",
        )
        ax.text(col_x[3], yi, denom, fontsize=6.5, color=MUTED, ha="left", va="center",
                transform=ax.transAxes)
        ax.text(col_x[4], yi, reading, fontsize=6.5, color=MUTED, ha="left", va="center",
                transform=ax.transAxes)

    # Bottom insight box
    ax.text(
        0.01,
        0.06,
        "Takeaway: Energy/Vertices are closed. Edges ownership far exceeds the ≥60% ADR 0012 bar.\n"
        "The only open multiset residual is one degree-pruning pair swap on a shared hub vertex\n"
        "(crop: MATLAB [4212, 6281] vs Python [4043, 6281]) — Network’s −1 strand is the same event.",
        fontsize=6.6,
        color=INK,
        ha="left",
        va="bottom",
        transform=ax.transAxes,
        linespacing=1.4,
        bbox=dict(boxstyle="round,pad=0.35", fc="#F5F5F5", ec=MID, lw=0.45),
    )


def _save(fig: plt.Figure, stem: str, out_dir: Path) -> Path:
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")
    return png


def main() -> list[Path]:
    """Write four claim-driven standalone figures."""
    out_dir = Path(__file__).resolve().parent
    written: list[Path] = []

    specs: list[tuple[str, tuple[float, float], tuple[float, float, float, float], object]] = [
        ("parity_trajectory", (7.2, 4.2), (0.12, 0.97, 0.90, 0.18), draw_trajectory),
        ("parity_funnel", (7.2, 4.0), (0.11, 0.97, 0.90, 0.16), draw_funnel),
        ("parity_agreement", (7.4, 4.4), (0.12, 0.97, 0.90, 0.22), draw_agreement),
        ("parity_cert_table", (7.0, 4.0), (0.02, 0.99, 0.92, 0.04), draw_cert_table),
    ]

    for stem, figsize, margins, drawer in specs:
        left, right, top, bottom = margins
        fig, ax = plt.subplots(figsize=figsize)
        fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom)
        drawer(ax)
        written.append(_save(fig, stem, out_dir))

    return written


if __name__ == "__main__":
    paths = main()
    print(f"Generated {len(paths)} claim-driven figures.")
