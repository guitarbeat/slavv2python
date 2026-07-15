#!/usr/bin/env python3
"""Standalone claim-driven MATLAB↔Python parity figures (view layer).

Each figure answers one non-trivial question from the exact-parity campaign.
Campaign numbers live only in parity_campaign_series.py — update that file when
docs/reference/core/EXACT_PROOF_FINDINGS.md moves.

Outputs:
  parity_trajectory.{png,pdf}
  parity_funnel.{png,pdf}
  parity_agreement.{png,pdf}
  parity_cert_table.{png,pdf}
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

# Sibling import when run as `python figures/generate_parity_claim_figures.py`
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from parity_campaign_series import (  # noqa: E402
    AGREEMENT_CLAIM,
    AGREEMENT_FOOTNOTE,
    CANONICAL_AUDITS,
    CERT_CLAIM,
    CERT_ROWS,
    CERT_TAKEAWAY,
    FUNNEL_CLAIM,
    FUNNEL_FOOTNOTE,
    FUNNEL_PHASES,
    LOG_FLOOR_COUNT,
    RETIRED_GATE_MISSING,
    TRAJECTORY_CLAIM,
    TRAJECTORY_FOOTNOTE,
    TRAJECTORY_STEPS,
    Annotation,
    CanonicalAudit,
    CertRow,
    ColorKey,
    FunnelPhase,
    TrajectoryKind,
    TrajectoryStep,
)

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

_COLOR: dict[ColorKey, str] = {
    "accent": ACCENT,
    "muted": MUTED,
    "amber": AMBER,
    "green": GREEN,
    "steel": STEEL,
}

_KIND_COLOR: dict[TrajectoryKind, str] = {
    "null": MID,
    "leap": ACCENT,
    "polish": TEAL,
    "closed": GREEN,
}

_TONE_COLOR = {
    "closed": GREEN,
    "residual": AMBER,
    "neutral": INK,
}

Drawer = Callable[[plt.Axes], None]


def _claim(ax: plt.Axes, text: str) -> None:
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


def _annotate(ax: plt.Axes, i: int, y: float, ann: Annotation) -> None:
    color = _COLOR[ann.color_key]
    ax.annotate(
        ann.text,
        xy=(i, y),
        xytext=(ann.text_x, ann.text_y),
        fontsize=7.0 if ann.bold else 6.6,
        color=color,
        fontweight="bold" if ann.bold else "normal",
        ha="left" if ann.text_x >= i else "center",
        arrowprops={
            "arrowstyle": "->",
            "color": color,
            "lw": 0.85 if ann.bold else 0.7,
        },
        bbox={
            "boxstyle": "round,pad=0.22",
            "fc": "white",
            "ec": color if ann.bold or ann.color_key != "muted" else MID,
            "lw": 0.5 if ann.bold else 0.4,
        },
    )


def _log_count_formatter(v: float, _: object) -> str:
    if v < 1.1:
        return "0"
    return f"{round(v):,}"


# ---------------------------------------------------------------------------
# Drawers (view only)
# ---------------------------------------------------------------------------


def draw_trajectory(ax: plt.Axes) -> None:
    """Only one step recovered ~6k pairs; queue-order cosmetics did nothing."""
    steps: list[TrajectoryStep] = TRAJECTORY_STEPS
    _claim(ax, TRAJECTORY_CLAIM)

    missing = np.array([s.missing for s in steps], dtype=float)
    # +1 so zero residual sits above the log floor
    y_plot = missing + 1.0
    x = np.arange(len(steps))

    ax.set_yscale("log")
    ax.set_ylim(0.7, 2.0e4)
    ax.set_xlim(-0.45, len(steps) - 0.45)

    ax.axhline(RETIRED_GATE_MISSING, color=MID, ls=(0, (3, 2)), lw=0.9, zorder=1)
    ax.text(
        len(steps) - 0.5,
        RETIRED_GATE_MISSING * 1.12,
        f"retired 80% gate\n(~{RETIRED_GATE_MISSING:,} still missing)",
        ha="right",
        va="bottom",
        fontsize=6.5,
        color=MUTED,
        linespacing=1.2,
    )

    for i in range(len(x) - 1):
        edge = _KIND_COLOR[steps[i + 1].kind]
        ax.plot(
            [x[i], x[i + 1]],
            [y_plot[i], y_plot[i + 1]],
            color=edge,
            lw=1.6 if steps[i + 1].kind == "leap" else 1.1,
            zorder=2,
            solid_capstyle="round",
        )

    for i, step in enumerate(steps):
        ax.plot(
            x[i],
            y_plot[i],
            "o",
            ms=8 if step.kind == "leap" else 6.5,
            color=_KIND_COLOR[step.kind],
            markeredgecolor=LINE,
            markeredgewidth=0.45,
            zorder=4,
        )

    for i, step in enumerate(steps):
        if step.kind == "closed":
            ax.text(
                x[i],
                2.8,
                "0 missing\n(gen. closed)",
                ha="center",
                va="bottom",
                fontsize=7.2,
                color=GREEN,
                fontweight="bold",
                linespacing=1.15,
            )
        else:
            ax.text(
                x[i],
                y_plot[i] * 1.55,
                f"{step.missing:,} missing",
                ha="center",
                va="bottom",
                fontsize=7.2,
                color=INK,
                fontweight="bold" if step.kind == "leap" else "normal",
            )
        if step.annotation is not None:
            _annotate(ax, i, float(y_plot[i]), step.annotation)

    ax.set_xticks(x)
    ax.set_xticklabels([s.label for s in steps], fontsize=7.2)
    ax.set_ylabel("MATLAB final pairs still absent\nfrom Python candidates (log)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(_log_count_formatter))
    _footnote(ax, TRAJECTORY_FOOTNOTE, y=-0.17)


def draw_funnel(ax: plt.Axes) -> None:
    """Crop final residual: missing and extra tell different stories."""
    phases: list[FunnelPhase] = FUNNEL_PHASES
    _claim(ax, FUNNEL_CLAIM)

    x = np.arange(len(phases))
    w = 0.36
    ax.set_yscale("log")
    ax.set_ylim(0.7, 1.2e4)

    missing_h = np.array([max(p.missing, LOG_FLOOR_COUNT) for p in phases], dtype=float)
    ax.bar(
        x - w / 2,
        missing_h,
        width=w,
        color=RED,
        edgecolor=LINE,
        linewidth=0.4,
        label="Missing MATLAB pairs",
        zorder=3,
    )

    extra_h = np.array(
        [max(p.extra, LOG_FLOOR_COUNT) if p.extra is not None else 0.0 for p in phases],
        dtype=float,
    )
    bars_e = ax.bar(
        x + w / 2,
        extra_h,
        width=w,
        color=AMBER,
        edgecolor=LINE,
        linewidth=0.4,
        label="Extra Python pairs",
        zorder=3,
    )
    for i, phase in enumerate(phases):
        if phase.extra is None:
            bars_e[i].set_alpha(0.0)

    for i, phase in enumerate(phases):
        ax.text(
            x[i] - w / 2,
            max(phase.missing, LOG_FLOOR_COUNT) * 1.18,
            f"{phase.missing:,}",
            ha="center",
            va="bottom",
            fontsize=7.0,
            color=RED,
            fontweight="bold",
        )
        if phase.extra is None:
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
        else:
            ax.text(
                x[i] + w / 2,
                max(phase.extra, LOG_FLOOR_COUNT) * 1.18,
                f"{phase.extra:,}",
                ha="center",
                va="bottom",
                fontsize=7.0,
                color=AMBER,
                fontweight="bold",
            )
        if phase.annotation is not None:
            # Anchor on the extra bar when present (the interesting series)
            anchor_y = float(phase.extra) if phase.extra is not None else float(phase.missing)
            _annotate(ax, i, max(anchor_y, LOG_FLOOR_COUNT), phase.annotation)

    ax.set_xticks(x)
    ax.set_xticklabels([p.label for p in phases], fontsize=7.2)
    ax.set_ylabel("Final residual pair count (log)")
    ax.legend(frameon=False, loc="upper right", fontsize=7.2)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(_log_count_formatter))
    _footnote(ax, FUNNEL_FOOTNOTE, y=-0.16)


def draw_agreement(ax: plt.Axes) -> None:
    """Signed edge residual across canonical audits — the non-monotonic story."""
    audits: list[CanonicalAudit] = CANONICAL_AUDITS
    _claim(ax, AGREEMENT_CLAIM)

    edge_d = np.array([a.edge_delta for a in audits], dtype=float)
    net_d = np.array([a.network_delta for a in audits], dtype=float)
    x = np.arange(len(audits))

    ax.axhline(0, color=INK, lw=0.85, zorder=1)
    ax.fill_between([-0.6, len(audits) - 0.4], 0, 3200, color="#E8F5E9", alpha=0.55, zorder=0)
    ax.fill_between([-0.6, len(audits) - 0.4], -11000, 0, color="#FFEBEE", alpha=0.45, zorder=0)
    ax.text(
        len(audits) - 0.65,
        2200,
        "Python over",
        ha="right",
        va="center",
        fontsize=6.5,
        color=GREEN,
        fontweight="bold",
    )
    ax.text(
        len(audits) - 0.65,
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
        label="Edges: Python \u2212 MATLAB connections",
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
        label="Network: Python \u2212 MATLAB strands",
    )

    for i, audit in enumerate(audits):
        ed = audit.edge_delta
        if audit.end_label_style:
            ax.text(
                x[i] + 0.08,
                ed + 350,
                "edges 0" if ed == 0 else f"{ed:+,}",
                ha="left",
                va="bottom",
                fontsize=6.8,
                color=ACCENT,
                fontweight="bold",
            )
            if audit.network_delta != 0 and i == len(audits) - 1:
                ax.text(
                    x[i] + 0.08,
                    audit.network_delta - 550,
                    f"net \u2212{abs(audit.network_delta)}",
                    ha="left",
                    va="top",
                    fontsize=6.8,
                    color=STEEL,
                    fontweight="bold",
                )
        else:
            sign = f"{ed:+,}" if ed != 0 else "0"
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
        if audit.annotation is not None:
            # Network-track callout anchors on network series; others on edges
            anchor_y = (
                float(audit.network_delta) if audit.annotation.color_key == "steel" else float(ed)
            )
            _annotate(ax, i, anchor_y, audit.annotation)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{a.label}\n{a.note}" for a in audits], fontsize=6.5)
    ax.set_ylabel("Signed residual count\n(Python \u2212 MATLAB)")
    ax.set_xlim(-0.55, len(audits) - 0.15)
    ax.set_ylim(-11000, 3500)
    ax.legend(frameon=False, loc="center right", fontsize=6.6)
    _footnote(ax, AGREEMENT_FOOTNOTE, y=-0.18)


def draw_cert_table(ax: plt.Axes) -> None:
    """Absolute remaining mismatches — not a wall of PASS."""
    rows: list[CertRow] = CERT_ROWS
    _claim(ax, CERT_CLAIM)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

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
    for i, row in enumerate(rows):
        yi = 0.80 - i * row_h
        if i % 2 == 0:
            ax.axhspan(
                yi - row_h * 0.42,
                yi + row_h * 0.42,
                facecolor=LIGHT,
                edgecolor="none",
            )
        if row.tone == "residual":
            ax.add_patch(
                FancyBboxPatch(
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
            )

        res_color = _TONE_COLOR[row.tone]
        ax.text(
            col_x[0],
            yi,
            row.stage,
            fontsize=7.0,
            color=INK,
            ha="left",
            va="center",
            transform=ax.transAxes,
            fontweight="bold" if row.tone == "residual" else "normal",
        )
        ax.text(
            col_x[1],
            yi,
            row.quantity,
            fontsize=6.5,
            color=MUTED,
            ha="left",
            va="center",
            transform=ax.transAxes,
        )
        ax.text(
            col_x[2],
            yi,
            row.residual_display,
            fontsize=7.5,
            color=res_color,
            ha="left",
            va="center",
            transform=ax.transAxes,
            fontweight="bold",
        )
        ax.text(
            col_x[3],
            yi,
            row.denom,
            fontsize=6.5,
            color=MUTED,
            ha="left",
            va="center",
            transform=ax.transAxes,
        )
        ax.text(
            col_x[4],
            yi,
            row.reading,
            fontsize=6.5,
            color=MUTED,
            ha="left",
            va="center",
            transform=ax.transAxes,
        )

    ax.text(
        0.01,
        0.06,
        CERT_TAKEAWAY,
        fontsize=6.6,
        color=INK,
        ha="left",
        va="bottom",
        transform=ax.transAxes,
        linespacing=1.4,
        bbox={"boxstyle": "round,pad=0.35", "fc": "#F5F5F5", "ec": MID, "lw": 0.45},
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

    specs: list[tuple[str, tuple[float, float], tuple[float, float, float, float], Drawer]] = [
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
