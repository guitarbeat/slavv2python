"""Energy float parity gate: ULP-bounded compare for certification (ADR 0011)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .constants import ANALYSIS_DIR
from .mismatch_diagnostics import _ordered_float64_bits, _ulp_mismatch_stats
from .utils import now_iso, write_json_with_hash, write_text_with_hash

if TYPE_CHECKING:
    from pathlib import Path

DEFAULT_MAX_ULPS = 8
CERTIFICATION_MAX_ULPS = 48
DENORM_ENERGY_THRESHOLD = 1e-3
MAX_ABS_DELTA_NORMAL = 1e-10


@dataclass(frozen=True)
class EnergyFloatGateOptions:
    """Policy for energy.energy float comparison in prove-exact (ADR 0011 Option B)."""

    strict_floats: bool = False
    max_ulps: int = CERTIFICATION_MAX_ULPS
    denorm_threshold: float = DENORM_ENERGY_THRESHOLD
    max_abs_delta: float = MAX_ABS_DELTA_NORMAL


def evaluate_energy_float_gate(
    matlab_energy: np.ndarray,
    python_energy: np.ndarray,
    matlab_scales: np.ndarray,
    python_scales: np.ndarray,
    *,
    options: EnergyFloatGateOptions | None = None,
) -> dict[str, Any]:
    """Return pass/fail and counts for the ADR 0011 energy float policy."""
    gate_options = options or EnergyFloatGateOptions()
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

    if gate_options.strict_floats:
        float_pass_mask = scale_agree_mask & energy_equal
        over_ulp = int(
            np.count_nonzero(scale_agree_mask & ~energy_equal)
        )
        denorm_escape_count = 0
        abs_delta_fail_count = 0
    else:
        ulp_full = _float64_ulp_array(matlab_e, python_e)
        delta = np.abs(matlab_e - python_e)
        denorm_mask = np.abs(matlab_e) < gate_options.denorm_threshold
        within_ulp = ulp_full <= gate_options.max_ulps
        within_delta = delta <= gate_options.max_abs_delta
        float_pass_mask = scale_agree_mask & (
            energy_equal
            | (denorm_mask & within_delta)
            | (within_ulp & within_delta)
        )
        over_ulp = int(np.count_nonzero(scale_agree_mask & ~within_ulp & ~denorm_mask))
        denorm_escape_count = int(
            np.count_nonzero(scale_agree_mask & ~energy_equal & denorm_mask & within_delta)
        )
        abs_delta_fail_count = int(
            np.count_nonzero(scale_agree_mask & ~within_delta & ~denorm_mask)
        )

    total_voxels = int(matlab_e.size)
    passed_voxels = int(np.count_nonzero(scale_equal & float_pass_mask))
    failed_voxels = total_voxels - passed_voxels
    mismatch_mask = scale_agree_mask & ~energy_equal
    ulp_stats = _ulp_mismatch_stats(matlab_e[mismatch_mask], python_e[mismatch_mask])
    mismatch_delta = np.abs(matlab_e[mismatch_mask] - python_e[mismatch_mask])

    failures: list[str] = []
    if scale_mismatch_count:
        failures.append(f"scale_mismatch_count={scale_mismatch_count}")
    if over_ulp:
        failures.append(f"energy_ulp_over_max={over_ulp}")
    if abs_delta_fail_count:
        failures.append(f"energy_abs_delta_over_max={abs_delta_fail_count}")

    return {
        "strict_floats": gate_options.strict_floats,
        "max_ulps": int(gate_options.max_ulps),
        "denorm_threshold": float(gate_options.denorm_threshold),
        "max_abs_delta": float(gate_options.max_abs_delta),
        "passed": not failures,
        "failures": failures,
        "total_voxels": total_voxels,
        "passed_voxels": passed_voxels,
        "failed_voxels": failed_voxels,
        "pass_rate": float(passed_voxels / total_voxels) if total_voxels else 0.0,
        "scale_mismatch_count": scale_mismatch_count,
        "scale_agree_energy_exact_match_count": int(np.count_nonzero(scale_agree_mask & energy_equal)),
        "scale_agree_energy_ulp_over_max_count": over_ulp,
        "scale_agree_denorm_escape_count": denorm_escape_count,
        "scale_agree_abs_delta_over_max_count": abs_delta_fail_count,
        "ulp_stats_on_mismatches": ulp_stats,
        "max_abs_delta_on_scale_agreeing_mismatches": float(np.max(mismatch_delta))
        if mismatch_delta.size
        else 0.0,
    }


def build_energy_ulp_proof_report(
    matlab_energy: np.ndarray,
    python_energy: np.ndarray,
    matlab_scales: np.ndarray,
    python_scales: np.ndarray,
    *,
    max_ulps: int = DEFAULT_MAX_ULPS,
    denorm_threshold: float = DENORM_ENERGY_THRESHOLD,
    max_abs_delta: float = MAX_ABS_DELTA_NORMAL,
    strict_floats: bool = False,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare Energy with strict scales and bounded ULP tolerance on floats."""
    gate = evaluate_energy_float_gate(
        matlab_energy,
        python_energy,
        matlab_scales,
        python_scales,
        options=EnergyFloatGateOptions(
            strict_floats=strict_floats,
            max_ulps=max_ulps,
            denorm_threshold=denorm_threshold,
            max_abs_delta=max_abs_delta,
        ),
    )
    within_ulp_count = int(
        gate["scale_agree_energy_exact_match_count"]
        + gate.get("scale_agree_denorm_escape_count", 0)
    )
    if not strict_floats:
        matlab_e = np.asarray(matlab_energy, dtype=np.float64)
        python_e = np.asarray(python_energy, dtype=np.float64)
        matlab_s = np.asarray(matlab_scales)
        python_s = np.asarray(python_scales)
        scale_equal = matlab_s == python_s
        energy_equal = matlab_e == python_e
        ulp_full = _float64_ulp_array(matlab_e, python_e)
        within_ulp_count += int(
            np.count_nonzero(
                scale_equal & ~energy_equal & (ulp_full <= max_ulps) & (np.abs(matlab_e - python_e) <= max_abs_delta)
            )
        )

    certification_note = (
        "Strict np.equal on energy.energy (--strict-floats)."
        if strict_floats
        else (
            "Certification gate per ADR 0011: strict scale_indices; energy.energy ULP "
            f"≤ {max_ulps} with denormal |Δ| ≤ {max_abs_delta:g} when |oracle| < {denorm_threshold:g}."
        )
    )

    return {
        "schema_version": 1,
        "created_at": now_iso(),
        "kind": "energy_ulp_proof",
        "max_ulps": int(max_ulps),
        "denorm_threshold": float(denorm_threshold),
        "max_abs_delta": float(max_abs_delta),
        "strict_floats": strict_floats,
        "passed": gate["passed"],
        "failures": gate["failures"],
        "provenance": provenance or {},
        "total_voxels": gate["total_voxels"],
        "passed_voxels": gate["passed_voxels"],
        "failed_voxels": gate["failed_voxels"],
        "pass_rate": gate["pass_rate"],
        "scale_mismatch_count": gate["scale_mismatch_count"],
        "scale_agree_energy_exact_match_count": gate["scale_agree_energy_exact_match_count"],
        "scale_agree_energy_ulp_within_max_count": within_ulp_count,
        "scale_agree_energy_ulp_over_max_count": gate["scale_agree_energy_ulp_over_max_count"],
        "scale_agree_denorm_escape_count": gate.get("scale_agree_denorm_escape_count", 0),
        "ulp_stats_on_mismatches": gate["ulp_stats_on_mismatches"],
        "max_abs_delta_on_scale_agreeing_mismatches": gate[
            "max_abs_delta_on_scale_agreeing_mismatches"
        ],
        "certification_note": certification_note,
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
    "CERTIFICATION_MAX_ULPS",
    "DEFAULT_MAX_ULPS",
    "DENORM_ENERGY_THRESHOLD",
    "EnergyFloatGateOptions",
    "MAX_ABS_DELTA_NORMAL",
    "build_energy_ulp_proof_report",
    "evaluate_energy_float_gate",
    "persist_energy_ulp_proof_report",
]
