"""Advisory Energy ULP proof for scale-agreeing voxels (does not replace prove-exact)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .constants import ANALYSIS_DIR
from .mismatch_diagnostics import _ordered_float64_bits, _ulp_mismatch_stats
from .utils import now_iso, write_json_with_hash, write_text_with_hash

if TYPE_CHECKING:
    from pathlib import Path

DEFAULT_MAX_ULPS = 8


def build_energy_ulp_proof_report(
    matlab_energy: np.ndarray,
    python_energy: np.ndarray,
    matlab_scales: np.ndarray,
    python_scales: np.ndarray,
    *,
    max_ulps: int = DEFAULT_MAX_ULPS,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare Energy with strict scales and bounded ULP tolerance on floats."""
    matlab_e = np.asarray(matlab_energy, dtype=np.float64)
    python_e = np.asarray(python_energy, dtype=np.float64)
    matlab_s = np.asarray(matlab_scales)
    python_s = np.asarray(python_scales)
    if matlab_e.shape != python_e.shape or matlab_s.shape != python_s.shape:
        raise ValueError("Energy and scale arrays must share shape")

    scale_equal = matlab_s == python_s
    scale_mismatch_count = int(np.count_nonzero(~scale_equal))
    energy_equal = matlab_e == python_e
    scale_agree_mask = scale_equal
    ulp_full = _float64_ulp_array(matlab_e, python_e)
    within_ulp = ulp_full <= max_ulps
    passed_mask = scale_agree_mask & within_ulp
    total_voxels = int(matlab_e.size)
    passed_voxels = int(np.count_nonzero(passed_mask))
    failed_voxels = total_voxels - passed_voxels

    mismatch_mask = scale_agree_mask & ~energy_equal
    ulp_stats = _ulp_mismatch_stats(matlab_e[mismatch_mask], python_e[mismatch_mask])
    delta = np.abs(matlab_e[mismatch_mask] - python_e[mismatch_mask])
    failures: list[str] = []
    if scale_mismatch_count:
        failures.append(f"scale_mismatch_count={scale_mismatch_count}")
    over_ulp = int(np.count_nonzero(scale_agree_mask & ~within_ulp))
    if over_ulp:
        failures.append(f"energy_ulp_over_max={over_ulp}")

    return {
        "schema_version": 1,
        "created_at": now_iso(),
        "kind": "energy_ulp_advisory_proof",
        "max_ulps": int(max_ulps),
        "passed": not failures,
        "failures": failures,
        "provenance": provenance or {},
        "total_voxels": total_voxels,
        "passed_voxels": passed_voxels,
        "failed_voxels": failed_voxels,
        "pass_rate": float(passed_voxels / total_voxels) if total_voxels else 0.0,
        "scale_mismatch_count": scale_mismatch_count,
        "scale_agree_energy_exact_match_count": int(np.count_nonzero(scale_agree_mask & energy_equal)),
        "scale_agree_energy_ulp_within_max_count": int(
            np.count_nonzero(scale_agree_mask & ~energy_equal & within_ulp)
        ),
        "scale_agree_energy_ulp_over_max_count": over_ulp,
        "ulp_stats_on_mismatches": ulp_stats,
        "max_abs_delta_on_scale_agreeing_mismatches": float(np.max(delta)) if delta.size else 0.0,
        "certification_note": (
            "Advisory gate only; Phase 1 certification still requires strict prove-exact "
            "np.equal on energy.energy."
        ),
    }


def persist_energy_ulp_proof_report(dest_run_root: Path, report: dict[str, Any]) -> Path:
    """Write advisory ULP proof JSON beside other analysis artifacts."""
    path = dest_run_root / ANALYSIS_DIR / "exact_proof_energy_ulp.json"
    write_json_with_hash(path, report)
    write_text_with_hash(path.with_suffix(".txt"), _render_energy_ulp_report(report))
    return path


def _float64_ulp_array(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    actual_f = np.asarray(actual, dtype=np.float64)
    expected_f = np.asarray(expected, dtype=np.float64)
    equal = actual_f == expected_f
    ordered = _ordered_float64_bits(actual_f).astype(np.int64)
    ordered_e = _ordered_float64_bits(expected_f).astype(np.int64)
    ulp = np.abs(ordered - ordered_e)
    ulp[equal] = 0
    return ulp.astype(np.int64)


def _render_energy_ulp_report(report: dict[str, Any]) -> str:
    lines = [
        "Energy ULP advisory proof",
        f"passed: {report['passed']}",
        f"max_ulps: {report['max_ulps']}",
        f"pass_rate: {report['pass_rate']:.6f}",
        f"scale_mismatch_count: {report['scale_mismatch_count']}",
        f"energy_ulp_over_max: {report['scale_agree_energy_ulp_over_max_count']}",
    ]
    if report.get("failures"):
        lines.append("failures: " + ", ".join(report["failures"]))
    stats = report.get("ulp_stats_on_mismatches", {})
    if stats:
        lines.append(f"ulp_p50_on_mismatches: {stats.get('ulp_p50')}")
        lines.append(f"ulp_p90_on_mismatches: {stats.get('ulp_p90')}")
        lines.append(f"max_ulp_on_mismatches: {stats.get('max_ulp')}")
    lines.append(str(report.get("certification_note", "")))
    return "\n".join(lines)


__all__ = [
    "DEFAULT_MAX_ULPS",
    "build_energy_ulp_proof_report",
    "persist_energy_ulp_proof_report",
]
