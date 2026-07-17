#!/usr/bin/env python3
"""Standalone claim-driven MATLAB↔Python parity figures (view layer).

Each figure answers one non-trivial question from the exact-parity campaign.
Campaign numbers live only in parity_campaign_series.py — update that file when
docs/reference/core/EXACT_PROOF_FINDINGS.md moves.

Layout is **wrap-first**: canvas ~3.3 in wide so type stays legible at
``0.48\\textwidth`` wrapfigure (and still scales cleanly to full-width).
Long narrative stays in manuscript captions — not in-figure footnotes.

Outputs:
  parity_trajectory.{png,pdf}
  parity_funnel.{png,pdf}
  parity_agreement.{png,pdf}
  parity_cert_table.{png,pdf}
"""

from __future__ import annotations

import sys
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects as pe
from matplotlib.patches import FancyBboxPatch

# Halo keeps data labels legible over lines/markers at wrap scale
_LABEL_HALO = [pe.withStroke(linewidth=2.2, foreground="white")]

# Sibling import when run as `python figures/generate_parity_claim_figures.py`
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from parity_campaign_series import (  # noqa: E402
    AGREEMENT_CLAIM,
    CANONICAL_AUDITS,
    CERT_CLAIM,
    CERT_ROWS,
    CERT_TAKEAWAY,
    FUNNEL_CLAIM,
    FUNNEL_PHASES,
    LOG_FLOOR_COUNT,
    RETIRED_GATE_MISSING,
    TRAJECTORY_CLAIM,
    TRAJECTORY_STEPS,
    Annotation,
    CanonicalAudit,
    CertRow,
    ColorKey,
    FunnelPhase,
    TrajectoryKind,
    TrajectoryStep,
)

# ---------------------------------------------------------------------------
# Wrap-first design tokens (physical inches ≈ half-page wrap)
# ---------------------------------------------------------------------------

# Target print width for in-text wrap (~0.48\textwidth on letter with 1" margins)
WRAP_WIDTH_IN = 3.30

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.0,
        "axes.labelsize": 7.5,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.55,
        "ytick.major.width": 0.55,
        "xtick.major.size": 2.2,
        "ytick.major.size": 2.2,
        "lines.linewidth": 1.25,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 600,
        "text.antialiased": True,
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


@dataclass(frozen=True)
class FigureSpec:
    stem: str
    figsize: tuple[float, float]
    left: float
    right: float
    top: float
    bottom: float
    drawer: Drawer


def _claim(ax: plt.Axes, text: str, *, width: int = 38) -> None:
    """Left-aligned claim title, hard-wrapped for narrow canvas."""
    wrapped = "\n".join(textwrap.wrap(text, width=width, break_long_words=False))
    ax.set_title(
        wrapped,
        loc="left",
        pad=5,
        fontweight="bold",
        fontsize=8.2,
        color=INK,
        linespacing=1.15,
    )


def _annotate(ax: plt.Axes, i: int, y: float, ann: Annotation) -> None:
    color = _COLOR[ann.color_key]
    ax.annotate(
        ann.text,
        xy=(i, y),
        xytext=(ann.text_x, ann.text_y),
        fontsize=6.2 if ann.bold else 5.9,
        color=color,
        fontweight="bold" if ann.bold else "normal",
        ha="left" if ann.text_x >= i else "center",
        arrowprops={
            "arrowstyle": "->",
            "color": color,
            "lw": 0.75 if ann.bold else 0.6,
            "shrinkA": 0,
            "shrinkB": 2,
        },
        bbox={
            "boxstyle": "round,pad=0.18",
            "fc": "white",
            "ec": color if ann.bold or ann.color_key != "muted" else MID,
            "lw": 0.45 if ann.bold else 0.35,
        },
        zorder=6,
    )


def _log_count_formatter(v: float, _: object) -> str:
    if v < 1.1:
        return "0"
    if v >= 1000:
        return f"{round(v):,}"
    return f"{round(v)}"


def _pad_limits(values: np.ndarray, pad_frac: float = 0.12) -> tuple[float, float]:
    lo = float(np.min(values))
    hi = float(np.max(values))
    span = max(hi - lo, 1.0)
    pad = span * pad_frac
    return lo - pad, hi + pad


# ---------------------------------------------------------------------------
# Drawers (view only)
# ---------------------------------------------------------------------------


def draw_trajectory(ax: plt.Axes) -> None:
    """Only one step recovered ~6k pairs; queue-order cosmetics did nothing."""
    steps: list[TrajectoryStep] = TRAJECTORY_STEPS
    _claim(ax, TRAJECTORY_CLAIM, width=36)

    missing = np.array([s.missing for s in steps], dtype=float)
    # +1 so zero residual sits above the log floor
    y_plot = missing + 1.0
    x = np.arange(len(steps))

    ax.set_yscale("log")
    y_lo, y_hi = _pad_limits(y_plot, pad_frac=0.40)
    ax.set_ylim(max(0.7, y_lo * 0.5), max(y_hi, 2.0e4))
    ax.set_xlim(-0.40, len(steps) - 0.40)

    ax.axhline(RETIRED_GATE_MISSING, color=MID, ls=(0, (3, 2)), lw=0.85, zorder=1)
    ax.text(
        len(steps) - 0.45,
        RETIRED_GATE_MISSING * 1.18,
        f"80% gate\n(~{RETIRED_GATE_MISSING:,})",
        ha="right",
        va="bottom",
        fontsize=5.8,
        color=MUTED,
        linespacing=1.1,
    )

    for i in range(len(x) - 1):
        edge = _KIND_COLOR[steps[i + 1].kind]
        ax.plot(
            [x[i], x[i + 1]],
            [y_plot[i], y_plot[i + 1]],
            color=edge,
            lw=1.7 if steps[i + 1].kind == "leap" else 1.15,
            zorder=2,
            solid_capstyle="round",
        )

    for i, step in enumerate(steps):
        ax.plot(
            x[i],
            y_plot[i],
            "o",
            ms=7.0 if step.kind == "leap" else 5.8,
            color=_KIND_COLOR[step.kind],
            markeredgecolor=LINE,
            markeredgewidth=0.4,
            zorder=4,
        )

    # Point labels: show every count (short); closed gets a compact badge
    for i, step in enumerate(steps):
        if step.kind == "closed":
            ax.text(
                x[i],
                max(3.0, y_plot[i] * 3.2),
                "0\nclosed",
                ha="center",
                va="bottom",
                fontsize=6.2,
                color=GREEN,
                fontweight="bold",
                linespacing=1.05,
                path_effects=_LABEL_HALO,
                zorder=5,
            )
        elif step.kind == "leap":
            ax.text(
                x[i],
                y_plot[i] * 1.65,
                f"{step.missing:,}",
                ha="center",
                va="bottom",
                fontsize=6.5,
                color=INK,
                fontweight="bold",
                path_effects=_LABEL_HALO,
                zorder=5,
            )
        else:
            # Skip duplicate baseline label on second flat point to reduce clutter
            if i > 0 and steps[i - 1].missing == step.missing:
                continue
            ax.text(
                x[i],
                y_plot[i] * 1.55,
                f"{step.missing:,}",
                ha="center",
                va="bottom",
                fontsize=6.0,
                color=INK,
                path_effects=_LABEL_HALO,
                zorder=5,
            )
        if step.annotation is not None:
            _annotate(ax, i, float(y_plot[i]), step.annotation)

    ax.set_xticks(x)
    ax.set_xticklabels([s.label for s in steps], fontsize=6.2, linespacing=1.05)
    ax.set_ylabel("Missing candidates (log)", labelpad=2)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(_log_count_formatter))
    ax.tick_params(axis="y", pad=1.5)


def draw_funnel(ax: plt.Axes) -> None:
    """Crop final residual: missing and extra tell different stories."""
    phases: list[FunnelPhase] = FUNNEL_PHASES
    _claim(ax, FUNNEL_CLAIM, width=36)

    x = np.arange(len(phases))
    w = 0.34
    ax.set_yscale("log")
    ax.set_ylim(0.7, 1.4e4)

    missing_h = np.array([max(p.missing, LOG_FLOOR_COUNT) for p in phases], dtype=float)
    ax.bar(
        x - w / 2,
        missing_h,
        width=w,
        color=RED,
        edgecolor=LINE,
        linewidth=0.35,
        label="Missing",
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
        linewidth=0.35,
        label="Extra",
        zorder=3,
    )
    for i, phase in enumerate(phases):
        if phase.extra is None:
            bars_e[i].set_alpha(0.0)

    for i, phase in enumerate(phases):
        ax.text(
            x[i] - w / 2,
            max(phase.missing, LOG_FLOOR_COUNT) * 1.22,
            f"{phase.missing:,}",
            ha="center",
            va="bottom",
            fontsize=5.9,
            color=RED,
            fontweight="bold",
        )
        if phase.extra is None:
            ax.text(
                x[i] + w / 2,
                2.0,
                "n/a",
                ha="center",
                va="bottom",
                fontsize=5.6,
                color=MID,
                style="italic",
            )
        else:
            ax.text(
                x[i] + w / 2,
                max(phase.extra, LOG_FLOOR_COUNT) * 1.22,
                f"{phase.extra:,}",
                ha="center",
                va="bottom",
                fontsize=5.9,
                color=AMBER,
                fontweight="bold",
            )
        if phase.annotation is not None:
            if phase.annotation.series == "missing":
                anchor_y = float(phase.missing)
            elif phase.annotation.series == "extra" and phase.extra is not None:
                anchor_y = float(phase.extra)
            else:
                anchor_y = float(phase.extra) if phase.extra is not None else float(phase.missing)
            _annotate(ax, i, max(anchor_y, LOG_FLOOR_COUNT), phase.annotation)

    ax.set_xticks(x)
    ax.set_xticklabels([p.label for p in phases], fontsize=5.9, linespacing=1.05)
    ax.set_ylabel("Residual pairs (log)", labelpad=2)
    ax.legend(
        frameon=False,
        loc="upper right",
        fontsize=6.2,
        handlelength=1.1,
        borderaxespad=0.15,
        labelspacing=0.25,
    )
    ax.yaxis.set_major_formatter(plt.FuncFormatter(_log_count_formatter))
    ax.tick_params(axis="y", pad=1.5)
    ax.set_xlim(-0.55, len(phases) - 0.35)


def draw_agreement(ax: plt.Axes) -> None:
    """Signed edge residual across canonical audits — the non-monotonic story."""
    audits: list[CanonicalAudit] = CANONICAL_AUDITS
    _claim(ax, AGREEMENT_CLAIM, width=34)

    edge_d = np.array([a.edge_delta for a in audits], dtype=float)
    net_d = np.array([a.network_delta for a in audits], dtype=float)
    x = np.arange(len(audits))
    y_lo, y_hi = _pad_limits(np.concatenate([edge_d, net_d]), pad_frac=0.18)

    ax.axhline(0, color=INK, lw=0.8, zorder=1)
    ax.fill_between(
        [-0.55, len(audits) - 0.35],
        0,
        max(y_hi, 500),
        color="#E8F5E9",
        alpha=0.50,
        zorder=0,
    )
    ax.fill_between(
        [-0.55, len(audits) - 0.35],
        min(y_lo, -500),
        0,
        color="#FFEBEE",
        alpha=0.40,
        zorder=0,
    )
    # Band tags sit on the y-axis side (avoids legend / title collisions)
    ax.text(
        0.02,
        0.97,
        "over",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.5,
        color=GREEN,
        fontweight="bold",
    )
    ax.text(
        0.02,
        0.03,
        "under",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.5,
        color=RED,
        fontweight="bold",
    )

    ax.plot(
        x,
        edge_d,
        color=ACCENT,
        marker="o",
        ms=5.5,
        markerfacecolor="white",
        markeredgecolor=ACCENT,
        markeredgewidth=1.1,
        lw=1.35,
        zorder=3,
        label="Edges Δ",
    )
    ax.plot(
        x,
        net_d,
        color=STEEL,
        marker="s",
        ms=4.5,
        markerfacecolor="white",
        markeredgecolor=STEEL,
        markeredgewidth=0.95,
        lw=1.1,
        ls=(0, (3.5, 1.8)),
        zorder=3,
        label="Network Δ",
    )

    for i, audit in enumerate(audits):
        ed = audit.edge_delta
        if audit.near_zero_edge:
            ax.text(
                x[i],
                ed + 520,
                "0" if ed == 0 else f"{ed:+,}",
                ha="center",
                va="bottom",
                fontsize=5.9,
                color=ACCENT,
                fontweight="bold",
                clip_on=False,
                path_effects=_LABEL_HALO,
                zorder=5,
            )
            if audit.show_network_label and audit.network_delta != 0:
                nd = audit.network_delta
                net_txt = f"net {nd:+d}" if abs(nd) < 100 else f"net {nd:+,}"
                ax.text(
                    x[i],
                    nd - 520,
                    net_txt,
                    ha="center",
                    va="top",
                    fontsize=5.8,
                    color=STEEL,
                    fontweight="bold",
                    clip_on=False,
                    path_effects=_LABEL_HALO,
                    zorder=5,
                )
        else:
            # Place values just below (under) or above (over) the marker
            sign = f"{ed:+,}" if ed != 0 else "0"
            below = ed < 0
            ax.text(
                x[i],
                ed + (-720 if below else 480),
                sign,
                ha="center",
                va="top" if below else "bottom",
                fontsize=5.9,
                color=ACCENT,
                fontweight="bold",
                clip_on=False,
                path_effects=_LABEL_HALO,
                zorder=5,
            )
        if audit.annotation is not None:
            if audit.annotation.series == "network":
                anchor_y = float(audit.network_delta)
            else:
                anchor_y = float(ed)
            _annotate(ax, i, anchor_y, audit.annotation)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{a.label}\n{a.note}" for a in audits], fontsize=5.6, linespacing=1.02)
    ax.set_ylabel("Python − MATLAB", labelpad=2)
    ax.set_xlim(-0.50, len(audits) - 0.25)
    ax.set_ylim(y_lo, y_hi)
    ax.legend(
        frameon=False,
        loc="center right",
        fontsize=5.9,
        handlelength=1.4,
        borderaxespad=0.15,
        labelspacing=0.2,
    )
    ax.tick_params(axis="y", pad=1.2)
    # Compact y tick labels (k for thousands)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k" if abs(v) >= 1000 else f"{v:.0f}")
    )


def draw_cert_table(ax: plt.Axes) -> None:
    """Absolute remaining mismatches — compact stack for wrap width."""
    rows: list[CertRow] = CERT_ROWS
    _claim(ax, CERT_CLAIM, width=34)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Two-column compact rows: stage + residual | quantity / reading
    n = len(rows)
    y_top = 0.88
    y_bot = 0.22
    row_h = (y_top - y_bot) / n

    for i, row in enumerate(rows):
        yi = y_top - (i + 0.5) * row_h
        y0 = yi - row_h * 0.42
        y1 = yi + row_h * 0.42

        if row.tone == "residual":
            ax.add_patch(
                FancyBboxPatch(
                    (0.01, y0),
                    0.98,
                    y1 - y0,
                    boxstyle="round,pad=0.004,rounding_size=0.01",
                    facecolor="#FFF8E1",
                    edgecolor=AMBER,
                    linewidth=0.75,
                    transform=ax.transAxes,
                    zorder=0,
                )
            )
        elif i % 2 == 0:
            ax.add_patch(
                FancyBboxPatch(
                    (0.01, y0),
                    0.98,
                    y1 - y0,
                    boxstyle="round,pad=0.002,rounding_size=0.006",
                    facecolor=LIGHT,
                    edgecolor="none",
                    transform=ax.transAxes,
                    zorder=0,
                )
            )

        res_color = _TONE_COLOR[row.tone]
        ax.text(
            0.04,
            yi + 0.012,
            row.stage,
            fontsize=6.4,
            color=INK,
            ha="left",
            va="center",
            transform=ax.transAxes,
            fontweight="bold" if row.tone == "residual" else "normal",
            zorder=2,
        )
        ax.text(
            0.04,
            yi - 0.018,
            f"{row.quantity}  ·  {row.reading}",
            fontsize=5.4,
            color=MUTED,
            ha="left",
            va="center",
            transform=ax.transAxes,
            zorder=2,
        )
        # Residual as right-aligned badge
        ax.text(
            0.78,
            yi,
            row.residual_display,
            fontsize=8.0,
            color=res_color,
            ha="right",
            va="center",
            transform=ax.transAxes,
            fontweight="bold",
            zorder=2,
            family="sans-serif",
        )
        ax.text(
            0.96,
            yi,
            f"/ {row.denom}",
            fontsize=5.4,
            color=MUTED,
            ha="right",
            va="center",
            transform=ax.transAxes,
            zorder=2,
        )

    # Takeaway: short wrap at bottom
    takeaway = textwrap.fill(CERT_TAKEAWAY.replace("\n", " "), width=52)
    ax.text(
        0.02,
        0.02,
        takeaway,
        fontsize=5.5,
        color=INK,
        ha="left",
        va="bottom",
        transform=ax.transAxes,
        linespacing=1.25,
        bbox={
            "boxstyle": "round,pad=0.28",
            "fc": "#F5F5F5",
            "ec": MID,
            "lw": 0.4,
        },
    )


def _save(fig: plt.Figure, stem: str, out_dir: Path) -> Path:
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    # Tight pad keeps wrap boxes from oversized whitespace
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")
    return png


def main() -> list[Path]:
    """Write four claim-driven wrap-optimized figures."""
    out_dir = Path(__file__).resolve().parent
    written: list[Path] = []
    w = WRAP_WIDTH_IN

    # Height tuned so panels are slightly tall (better than wide when wrapped)
    specs = [
        FigureSpec("parity_trajectory", (w, 3.15), 0.14, 0.97, 0.86, 0.16, draw_trajectory),
        FigureSpec("parity_funnel", (w, 3.05), 0.13, 0.97, 0.86, 0.15, draw_funnel),
        FigureSpec("parity_agreement", (w, 3.25), 0.14, 0.97, 0.84, 0.16, draw_agreement),
        FigureSpec("parity_cert_table", (w, 3.55), 0.02, 0.99, 0.90, 0.02, draw_cert_table),
    ]

    for spec in specs:
        fig, ax = plt.subplots(figsize=spec.figsize)
        fig.subplots_adjust(left=spec.left, right=spec.right, top=spec.top, bottom=spec.bottom)
        spec.drawer(ax)
        written.append(_save(fig, spec.stem, out_dir))

    return written


if __name__ == "__main__":
    paths = main()
    print(f"Generated {len(paths)} claim-driven figures.")
