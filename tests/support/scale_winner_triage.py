"""Cross-language crop-Energy scale-winner probe orchestration and comparison."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from slavv_python.pipeline.energy.voxel_probe import (
    probe_exact_energy_voxel_at_octave,
    resolve_write_chunk_idx_for_voxel,
)
from tests.support.parity_harness import load_crop_image_and_config

REPO_ROOT = Path(__file__).resolve().parents[2]
MATLAB_BATCH_DRIVER = (
    REPO_ROOT / "workspace" / "scratch" / "matlab" / "probe_energy_mismatch_batch.m"
)


def _float_difference(path: str, python_value: Any, matlab_value: Any) -> dict[str, Any] | None:
    python_float = float(python_value)
    matlab_float = float(matlab_value)
    if np.float64(python_float).view(np.uint64) == np.float64(matlab_float).view(np.uint64):
        return None
    return {
        "path": path,
        "python": python_float,
        "matlab": matlab_float,
        "python_hex": f"0x{np.float64(python_float).view(np.uint64).item():016x}",
        "matlab_hex": f"0x{np.float64(matlab_float).view(np.uint64).item():016x}",
        "ulp_distance": _ulp_distance(python_float, matlab_float),
    }


def _ulp_distance(left: float, right: float) -> int:
    left_bits = np.float64(left).view(np.int64).item()
    right_bits = np.float64(right).view(np.int64).item()
    left_ordered = 0x8000000000000000 - left_bits if left_bits < 0 else left_bits
    right_ordered = 0x8000000000000000 - right_bits if right_bits < 0 else right_bits
    return abs(int(left_ordered) - int(right_ordered))


def _structural_difference(
    path: str, python_value: Any, matlab_value: Any
) -> dict[str, Any] | None:
    if isinstance(python_value, dict):
        if not isinstance(matlab_value, dict):
            return {"path": path, "python": python_value, "matlab": matlab_value}
        for key, value in python_value.items():
            if key not in matlab_value:
                return {"path": f"{path}.{key}", "python": value, "matlab": "<missing>"}
            difference = _structural_difference(f"{path}.{key}", value, matlab_value[key])
            if difference:
                return difference
        return None
    if isinstance(python_value, (list, tuple)):
        if not isinstance(matlab_value, list) or len(python_value) != len(matlab_value):
            return {"path": path, "python": python_value, "matlab": matlab_value}
        for index, (left, right) in enumerate(zip(python_value, matlab_value, strict=True)):
            difference = _structural_difference(f"{path}[{index}]", left, right)
            if difference:
                return difference
        return None
    if isinstance(python_value, float):
        return _float_difference(path, python_value, matlab_value)
    return (
        None
        if python_value == matlab_value
        else {"path": path, "python": python_value, "matlab": matlab_value}
    )


def build_python_batch_report(requests_path: Path) -> dict[str, Any]:
    """Recompute only requested crop voxels and preserve full probe payloads."""
    requests = json.loads(requests_path.read_text(encoding="utf-8"))["requests"]
    image, config = load_crop_image_and_config()
    records: list[dict[str, Any]] = []
    for request in requests:
        voxel_zyx = tuple(int(value) for value in request["voxel_zyx"])
        rf_zyx = tuple(int(value) for value in request["rf_zyx"])
        chunk_idx = resolve_write_chunk_idx_for_voxel(
            config, voxel_zyx=voxel_zyx, target_rf_zyx=rf_zyx
        )
        records.append(
            {
                "request_id": request["request_id"],
                "probe": probe_exact_energy_voxel_at_octave(
                    image,
                    config,
                    voxel_zyx=voxel_zyx,
                    target_rf_zyx=rf_zyx,
                    chunk_idx=chunk_idx,
                ),
            }
        )
    return {"version": 1, "records": records}


def _probe_difference(
    python_probe: dict[str, Any], matlab_probe: dict[str, Any]
) -> dict[str, Any] | None:
    stage_fields = (
        (
            "chunk_lattice",
            (
                "consolidated_octave",
                "rf_matlab_yxz",
                "chunk_lattice_dimensions_yxz",
                "chunk_idx",
                "write_index_yxz",
            ),
        ),
        (
            "coarse_support",
            (
                "write_window_zyx",
                "offsets_yxz",
                "strided_read_shape_yxz",
                "padded_shape_yxz",
                "coarse_local_slices_yxz",
            ),
        ),
        ("interpolation_mesh", ("mesh_at_voxel",)),
    )
    for stage, fields in stage_fields:
        for field in fields:
            difference = _structural_difference(field, python_probe[field], matlab_probe.get(field))
            if difference:
                return {"stage": stage, **difference}

    python_scales = {int(record["global_scale"]): record for record in python_probe["per_scale"]}
    matlab_scales = {int(record["global_scale"]): record for record in matlab_probe["per_scale"]}
    if tuple(python_scales) != tuple(matlab_scales):
        return {
            "stage": "scale_enumeration",
            "python": list(python_scales),
            "matlab": list(matlab_scales),
        }
    for scale, python_scale in python_scales.items():
        matlab_scale = matlab_scales[scale]
        difference = _float_difference(
            f"per_scale[{scale}].upsampled_energy",
            python_scale["upsampled_energy"],
            matlab_scale["upsampled_energy"],
        )
        if difference:
            return {"stage": "per_scale_energy", "global_scale": scale, **difference}

    difference = _structural_difference(
        "octave_winner.global_scale",
        python_probe["octave_winner"]["global_scale"],
        matlab_probe["octave_winner"]["global_scale"],
    )
    if difference:
        return {"stage": "winner_tiebreak", **difference}
    return None


def _failure_class(
    difference: dict[str, Any] | None,
    *,
    python_probe: dict[str, Any],
    matlab_probe: dict[str, Any],
) -> str:
    if difference is None:
        return "bit_identical"
    stage = difference.get("stage")
    if stage == "missing_matlab_response":
        return "missing_matlab_response"
    if stage == "winner_tiebreak":
        return "winner_scale_disagreement"
    if stage == "scale_enumeration":
        return "scale_enumeration"
    if stage == "per_scale_energy":
        python_winner = python_probe.get("octave_winner", {})
        matlab_winner = matlab_probe.get("octave_winner", {})
        if python_winner.get("global_scale") == matlab_winner.get("global_scale"):
            return "matching_winner_ulp_drift"
        return "winner_scale_disagreement"
    return "structural"


def compare_batch_reports(
    python_report: dict[str, Any], matlab_report: dict[str, Any]
) -> dict[str, Any]:
    """Classify the first Python/MATLAB divergence for every requested voxel."""
    python_records = {record["request_id"]: record["probe"] for record in python_report["records"]}
    matlab_records = {record["request_id"]: record["probe"] for record in matlab_report["records"]}
    results: list[dict[str, Any]] = []
    classification_counts: Counter[str] = Counter()
    for request_id, python_probe in python_records.items():
        matlab_probe = matlab_records.get(request_id)
        if matlab_probe is None:
            result = {
                "request_id": request_id,
                "passed": False,
                "stage": "missing_matlab_response",
                "failure_class": "missing_matlab_response",
            }
            classification_counts["missing_matlab_response"] += 1
            results.append(result)
            continue
        difference = _probe_difference(python_probe, matlab_probe)
        failure_class = _failure_class(
            difference,
            python_probe=python_probe,
            matlab_probe=matlab_probe,
        )
        classification_counts[failure_class] += 1
        results.append(
            {
                "request_id": request_id,
                "voxel_zyx": python_probe["voxel_zyx"],
                "python_winner": python_probe.get("octave_winner"),
                "matlab_winner": matlab_probe.get("octave_winner"),
                "passed": difference is None,
                "failure_class": failure_class,
                "first_difference": difference,
            }
        )
    passed = sum(1 for result in results if result["passed"])
    return {
        "version": 1,
        "probed": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "classifications": dict(classification_counts),
        "results": results,
    }


def run_matlab_batch(
    requests_path: Path,
    output_path: Path,
    matlab_exe: str | None = None,
    *,
    timeout_seconds: int = 1800,
) -> None:
    """Run all selected crop probes in one MATLAB R2019a process."""
    executable = matlab_exe or os.environ.get("MATLAB_EXE") or shutil.which("matlab.exe")
    if not executable or not Path(executable).is_file():
        raise FileNotFoundError("MATLAB R2019a executable unavailable; set MATLAB_EXE")
    if not MATLAB_BATCH_DRIVER.is_file():
        raise FileNotFoundError(f"MATLAB batch driver missing: {MATLAB_BATCH_DRIVER}")
    driver_dir = str(MATLAB_BATCH_DRIVER.parent).replace("'", "''")
    request_arg = str(requests_path.resolve()).replace("'", "''")
    output_arg = str(output_path.resolve()).replace("'", "''")
    expression = f"addpath('{driver_dir}'); probe_energy_mismatch_batch('', '{request_arg}', '{output_arg}');"
    completed = subprocess.run(
        [executable, "-batch", expression],
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError((completed.stderr or completed.stdout or "MATLAB batch failed").strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--matlab-exe")
    parser.add_argument("--matlab-timeout-seconds", type=int, default=1800)
    parser.add_argument(
        "--python-report",
        type=Path,
        help="Reuse an existing python_batch_probes.json instead of recomputing probes.",
    )
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    python_path = args.output_dir / "python_batch_probes.json"
    if args.python_report:
        python_report = json.loads(args.python_report.read_text(encoding="utf-8"))
        if args.python_report.resolve() != python_path.resolve():
            python_path.write_text(json.dumps(python_report, indent=2), encoding="utf-8")
    else:
        python_report = build_python_batch_report(args.requests)
        python_path.write_text(json.dumps(python_report, indent=2), encoding="utf-8")
    matlab_path = args.output_dir / "matlab_batch_probes.json"
    run_matlab_batch(
        args.requests,
        matlab_path,
        args.matlab_exe,
        timeout_seconds=args.matlab_timeout_seconds,
    )
    matlab_report = json.loads(matlab_path.read_text(encoding="utf-8"))
    comparison = compare_batch_reports(python_report, matlab_report)
    comparison_path = args.output_dir / "scale_winner_triage.json"
    comparison_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(json.dumps(comparison, indent=2))
    return 0 if comparison["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
