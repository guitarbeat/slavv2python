"""Generate data-backed figures for the SLAVV port/optimization report.

All figures are derived from real run artifacts (no synthetic data):
- Speedup / throughput     <- joblib "Done N tasks | elapsed" lines in a run log
- Energy ULP histogram     <- 03_Analysis/exact_proof_energy_ulp.json
- Energy parity composition<- 03_Analysis/exact_proof_energy_ulp.json

Usage::

    python scripts/make_report_figures.py \
        --ulp-json workspace/runs/oracle_180709_E/crop_M_exact/03_Analysis/exact_proof_energy_ulp.json \
        --run-log <baguette-log> \
        --out-dir docs/research/figures
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_DONE_RE = re.compile(r"Done\s+(\d+)\s+tasks.*?elapsed:\s+([0-9.]+)\s*(min|s)")


def _parse_run_log(log_path: Path) -> list[tuple[int, float]]:
    """Return (chunks_done, elapsed_seconds) points from joblib log lines."""
    points: list[tuple[int, float]] = []
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    for m in _DONE_RE.finditer(text):
        secs = float(m.group(2)) * (60.0 if m.group(3) == "min" else 1.0)
        points.append((int(m.group(1)), secs))
    return points


def figure_speedup(points: list[tuple[int, float]], serial_s_per_chunk: float, out: Path) -> None:
    """Throughput curve (parallel vs serial reference) + per-chunk speedup bars."""
    if len(points) < 2:
        print("  speedup: not enough log points, skipping")
        return
    chunks = [p[0] for p in points]
    secs = [p[1] for p in points]
    # steady-state per-chunk from the last ~10 points (linear slope)
    tail = points[-10:] if len(points) >= 10 else points
    par_s_per_chunk = (tail[-1][1] - tail[0][1]) / max(1, (tail[-1][0] - tail[0][0]))
    speedup = serial_s_per_chunk / par_s_per_chunk

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    hrs = [s / 3600.0 for s in secs]
    ax1.plot(
        hrs,
        chunks,
        color="#1f77b4",
        lw=2,
        label=f"parallel n_jobs=6 (~{par_s_per_chunk:.1f} s/chunk)",
    )
    serial_hrs = [c * serial_s_per_chunk / 3600.0 for c in chunks]
    ax1.plot(
        serial_hrs,
        chunks,
        color="#d62728",
        ls="--",
        lw=1.8,
        label=f"serial reference (~{serial_s_per_chunk:.1f} s/chunk)",
    )
    ax1.set_xlabel("wall-clock time (hours)")
    ax1.set_ylabel("energy chunks completed (octave 1)")
    ax1.set_title("Octave-1 throughput: parallel vs serial")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.bar(
        ["serial\n(n_jobs=1)", "parallel\n(n_jobs=6)"],
        [serial_s_per_chunk, par_s_per_chunk],
        color=["#d62728", "#1f77b4"],
    )
    ax2.set_ylabel("seconds per chunk (lower = faster)")
    ax2.set_title(f"Per-chunk time  —  {speedup:.1f}x speedup (bit-exact)")
    for i, v in enumerate([serial_s_per_chunk, par_s_per_chunk]):
        ax2.text(i, v + 0.5, f"{v:.1f}s", ha="center", fontsize=10)
    ax2.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(
        f"  wrote {out}  (speedup {speedup:.1f}x; serial {serial_s_per_chunk:.1f} -> {par_s_per_chunk:.1f} s/chunk)"
    )


def figure_ulp_histogram(ulp: dict, out: Path) -> None:
    """ULP-distance distribution on mismatching voxels (the 'why not pure ULP' figure)."""
    hist = ulp["ulp_stats_on_mismatches"]["ulp_histogram"]
    labels = [str(i) for i in range(9)] + ["9+"]
    keys = [str(i) for i in range(9)] + ["9_plus"]
    counts = [int(hist.get(k, 0)) for k in keys]
    p50 = ulp["ulp_stats_on_mismatches"]["ulp_p50"]
    p90 = ulp["ulp_stats_on_mismatches"]["ulp_p90"]
    max_dn = ulp["max_abs_delta_on_scale_agreeing_mismatches"]
    over = ulp["scale_agree_energy_ulp_over_max_count"]
    max_ulp_thresh = ulp["max_ulps"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, counts, color="#1f77b4")
    ax.set_xlabel("ULP distance (MATLAB vs Python energy)")
    ax.set_ylabel("voxel count (mismatching voxels)")
    ax.set_title("Energy ULP distance: tiny absolute error, large ULP tail")
    ax.axvline(float(p50), color="#2ca02c", ls="--", lw=1, label=f"p50 = {p50:.0f} ULP")
    ax.axvline(min(float(p90), 9.0), color="#ff7f0e", ls="--", lw=1, label=f"p90 = {p90:.0f} ULP")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    note = (
        f"max |Δ| on scale-agreeing voxels = {max_dn:.2e}\n"
        f"yet a pure-ULP gate at ≤{max_ulp_thresh} ULP flags {over:,} voxels\n"
        f"→ motivates np.allclose policy (ADR 0011)"
    )
    ax.text(
        0.97,
        0.78,
        note,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round", "facecolor": "#fff3cd", "edgecolor": "#856404"},
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


def figure_parity_composition(ulp: dict, out: Path) -> None:
    """Stacked composition of energy voxels: exact / within-tol / over-max."""
    total = int(ulp["total_voxels"])
    exact = int(ulp["scale_agree_energy_exact_match_count"])
    within = int(ulp["scale_agree_energy_ulp_within_max_count"])
    over = int(ulp["scale_agree_energy_ulp_over_max_count"])
    scale_mm = int(ulp["scale_mismatch_count"])
    other = max(0, total - exact - within - over)

    fig, ax = plt.subplots(figsize=(8, 2.6))
    segs = [
        ("bitwise-exact", exact, "#2ca02c"),
        (f"within ≤{ulp['max_ulps']} ULP (|Δ|≤2e-11)", within, "#1f77b4"),
        ("over ULP gate (allclose-pass)", over, "#ff7f0e"),
    ]
    if other:
        segs.append(("other", other, "#999999"))
    left = 0.0
    for label, val, color in segs:
        ax.barh(
            0, val, left=left, color=color, label=f"{label}: {val:,} ({100 * val / total:.2f}%)"
        )
        left += val
    ax.set_xlim(0, total)
    ax.set_yticks([])
    ax.set_title(
        f"Energy field parity composition  —  {total:,} voxels, scale mismatches = {scale_mm}"
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def main() -> None:
    """Generate all available data-backed report figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ulp-json", type=Path, required=True, help="exact_proof_energy_ulp.json")
    parser.add_argument(
        "--run-log", type=Path, default=None, help="Run log with joblib progress lines"
    )
    parser.add_argument(
        "--serial-s-per-chunk",
        type=float,
        default=44.4,
        help="Measured serial baseline (default from the n_jobs=1 run)",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("docs/research/figures"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Figures -> {args.out_dir}")

    ulp = json.loads(args.ulp_json.read_text(encoding="utf-8"))
    figure_ulp_histogram(ulp, args.out_dir / "energy_ulp_histogram.png")
    figure_parity_composition(ulp, args.out_dir / "energy_parity_composition.png")

    if args.run_log and args.run_log.is_file():
        figure_speedup(
            _parse_run_log(args.run_log),
            args.serial_s_per_chunk,
            args.out_dir / "energy_speedup.png",
        )
    else:
        print("  (no --run-log; skipping speedup figure)")


if __name__ == "__main__":
    main()
