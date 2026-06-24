"""Parity probe: replay cross-octave scale-winner reduction for mismatch voxels.

Role: expand mismatch voxels into per-octave probes and classify stored-state
vs replayed reduction (e.g. ``python_stored_state_path``).
Not a MATLAB port.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from tests.support.parity_harness import load_crop_image_and_config
from tests.support.parity_probe_scale_winner import (
    build_python_batch_report,
    run_matlab_batch,
)


def consolidated_octave_profiles(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return one request profile per consolidated octave/RF group."""
    octave_at_scales = np.asarray(config["octave_at_scales"])
    scale_resolution_factors = np.asarray(config["scale_resolution_factors"])
    profiles: list[dict[str, Any]] = []
    for octave in np.unique(octave_at_scales):
        scale_indices = np.where(octave_at_scales == octave)[0]
        rf_zyx = [int(value) for value in scale_resolution_factors[scale_indices[0]]]
        profiles.append(
            {
                "consolidated_octave": int(octave),
                "rf_zyx": rf_zyx,
                "rf_matlab_yxz": [rf_zyx[1], rf_zyx[2], rf_zyx[0]],
                "first_scale": int(scale_indices[0]),
                "last_scale": int(scale_indices[-1]),
            }
        )
    return profiles


def build_cross_octave_requests(
    mismatch_requests_path: Path,
    *,
    max_voxels: int | None = None,
) -> dict[str, Any]:
    """Expand selected mismatch voxels into one probe request per consolidated octave."""
    source = json.loads(mismatch_requests_path.read_text(encoding="utf-8"))
    _, config = load_crop_image_and_config()
    profiles = consolidated_octave_profiles(config)
    seen_voxels: set[tuple[int, int, int]] = set()
    voxel_records: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    for source_request in source["requests"]:
        voxel_zyx = tuple(int(value) for value in source_request["voxel_zyx"])
        if voxel_zyx in seen_voxels:
            continue
        if max_voxels is not None and len(voxel_records) >= max_voxels:
            break
        seen_voxels.add(voxel_zyx)
        parent_id = str(source_request["request_id"])
        voxel_records.append(
            {
                "parent_request_id": parent_id,
                "voxel_zyx": list(voxel_zyx),
                "stored_matlab_scale": int(source_request["matlab_scale"]),
                "stored_python_scale": int(source_request["stored_python_scale"]),
                "mismatch_count": int(source_request["mismatch_count"]),
                "boundary_class": source_request.get("boundary_class"),
            }
        )
        for profile in profiles:
            requests.append(
                {
                    "request_id": f"{parent_id}__oct{profile['consolidated_octave']}",
                    "parent_request_id": parent_id,
                    "voxel_zyx": list(voxel_zyx),
                    "rf_zyx": profile["rf_zyx"],
                    "rf_matlab_yxz": profile["rf_matlab_yxz"],
                    "consolidated_octave": profile["consolidated_octave"],
                    "first_scale": profile["first_scale"],
                    "last_scale": profile["last_scale"],
                    "stored_matlab_scale": int(source_request["matlab_scale"]),
                    "stored_python_scale": int(source_request["stored_python_scale"]),
                }
            )
    return {"version": 1, "voxels": voxel_records, "requests": requests}


def _strict_less_reduction(records: list[dict[str, Any]]) -> dict[str, Any]:
    best_energy = np.float64(0.0)
    best_scale = -1
    best_request_id: str | None = None
    candidates: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda item: int(item["probe"]["consolidated_octave"])):
        winner = record["probe"]["octave_winner"]
        energy = np.float64(winner["upsampled_energy"])
        scale = int(winner["global_scale"])
        candidates.append(
            {
                "request_id": record["request_id"],
                "consolidated_octave": int(record["probe"]["consolidated_octave"]),
                "global_scale": scale,
                "upsampled_energy": float(energy),
            }
        )
        if energy < best_energy:
            best_energy = energy
            best_scale = scale
            best_request_id = record["request_id"]
    return {
        "global_scale": int(best_scale),
        "upsampled_energy": float(best_energy),
        "source_request_id": best_request_id,
        "candidates": candidates,
    }


def compare_cross_octave_reports(
    requests: dict[str, Any],
    python_report: dict[str, Any],
    matlab_report: dict[str, Any],
) -> dict[str, Any]:
    """Compare replayed cross-octave reducers with stored scale mismatch metadata."""
    requests_by_id = {request["request_id"]: request for request in requests["requests"]}
    python_by_parent: dict[str, list[dict[str, Any]]] = {}
    matlab_by_parent: dict[str, list[dict[str, Any]]] = {}
    for record in python_report["records"]:
        parent_id = requests_by_id[record["request_id"]]["parent_request_id"]
        python_by_parent.setdefault(parent_id, []).append(record)
    for record in matlab_report["records"]:
        parent_id = requests_by_id[record["request_id"]]["parent_request_id"]
        matlab_by_parent.setdefault(parent_id, []).append(record)

    results: list[dict[str, Any]] = []
    for voxel_record in requests["voxels"]:
        parent_id = voxel_record["parent_request_id"]
        python_reduction = _strict_less_reduction(python_by_parent.get(parent_id, []))
        matlab_reduction = _strict_less_reduction(matlab_by_parent.get(parent_id, []))
        stored_matlab_scale = int(voxel_record["stored_matlab_scale"])
        stored_python_scale = int(voxel_record["stored_python_scale"])
        replay_scales_agree = python_reduction["global_scale"] == matlab_reduction["global_scale"]
        stored_scales_agree = stored_python_scale == stored_matlab_scale
        python_replay_matches_stored = python_reduction["global_scale"] == stored_python_scale
        matlab_replay_matches_stored = matlab_reduction["global_scale"] == stored_matlab_scale
        if not replay_scales_agree:
            classification = "cross_octave_reduction"
        elif not python_replay_matches_stored:
            classification = "python_stored_state_path"
        elif not matlab_replay_matches_stored:
            classification = "matlab_oracle_state_path"
        elif not stored_scales_agree:
            classification = "stored_mismatch_not_reproduced"
        else:
            classification = "replay_matches_stored"
        results.append(
            {
                **voxel_record,
                "classification": classification,
                "python_reduction": python_reduction,
                "matlab_reduction": matlab_reduction,
                "replay_scales_agree": replay_scales_agree,
                "stored_scales_agree": stored_scales_agree,
                "python_replay_matches_stored": python_replay_matches_stored,
                "matlab_replay_matches_stored": matlab_replay_matches_stored,
            }
        )
    classifications: dict[str, int] = {}
    for result in results:
        classification = str(result["classification"])
        classifications[classification] = classifications.get(classification, 0) + 1
    return {
        "version": 1,
        "voxels": len(results),
        "requests": len(requests["requests"]),
        "classifications": classifications,
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mismatch-requests", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-voxels", type=int)
    parser.add_argument("--matlab-exe")
    parser.add_argument("--matlab-timeout-seconds", type=int, default=1800)
    parser.add_argument("--python-report", type=Path)
    parser.add_argument("--matlab-report", type=Path)
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    requests = build_cross_octave_requests(args.mismatch_requests, max_voxels=args.max_voxels)
    requests_path = args.output_dir / "cross_octave_requests.json"
    requests_path.write_text(json.dumps(requests, indent=2), encoding="utf-8")

    python_path = args.output_dir / "python_cross_octave_probes.json"
    if args.python_report:
        python_report = json.loads(args.python_report.read_text(encoding="utf-8"))
    else:
        python_report = build_python_batch_report(requests_path)
    python_path.write_text(json.dumps(python_report, indent=2), encoding="utf-8")

    matlab_path = args.output_dir / "matlab_cross_octave_probes.json"
    if args.matlab_report:
        matlab_report = json.loads(args.matlab_report.read_text(encoding="utf-8"))
    else:
        run_matlab_batch(
            requests_path,
            matlab_path,
            args.matlab_exe,
            timeout_seconds=args.matlab_timeout_seconds,
        )
        matlab_report = json.loads(matlab_path.read_text(encoding="utf-8"))
    matlab_path.write_text(json.dumps(matlab_report, indent=2), encoding="utf-8")

    comparison = compare_cross_octave_reports(requests, python_report, matlab_report)
    comparison_path = args.output_dir / "cross_octave_reduction.json"
    comparison_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(json.dumps(comparison, indent=2))
    return 0 if not comparison["classifications"].get("cross_octave_reduction") else 1


if __name__ == "__main__":
    raise SystemExit(main())
