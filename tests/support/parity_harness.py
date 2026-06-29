"""Shared helpers for MATLAB-Python bit-parity evaluation in tests and scripts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import numpy.testing as npt

from slavv_python.analytics.parity.oracle.matlab_vector_loader import load_normalized_matlab_vectors
from slavv_python.analytics.parity.oracle.surfaces import load_oracle_surface
from slavv_python.analytics.parity.probes.adaptive_probes import ProbeResult, compare_probe_jsonl
from slavv_python.pipeline.energy.config import _prepare_energy_config
from slavv_python.pipeline.energy.parity_energy_voxel_probe import (
    probe_exact_energy_voxel_at_octave,
    resolve_write_chunk_idx_for_voxel,
)
from slavv_python.storage.loaders.tiff import load_tiff_volume

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
CROP_DATASET_HASH = "0cdf88e930482e9eb818963da22846c43b53b531582bf3aed83678b549863d06"
DEFAULT_CROP_TIFF = (
    REPO_ROOT / "workspace" / "datasets" / CROP_DATASET_HASH / "01_Input" / "180709_E_crop_M.tif"
)
DEFAULT_CROP_ORACLE_ROOT = REPO_ROOT / "workspace" / "oracles" / "180709_E_crop_M"
VOXEL_REGRESSION_FIXTURE = FIXTURES_DIR / "crop_energy_voxel_regression.json"
LINSPACE_TABLE_FIXTURE = FIXTURES_DIR / "matlab_linspace_probe_table.json"
LINSPACE_TABLE_SCRATCH = REPO_ROOT / "workspace" / "scratch" / "matlab_linspace_probe_table.json"

CROP_EXACT_PARAMS: dict[str, Any] = {
    "radius_of_smallest_vessel_in_microns": 1.5,
    "radius_of_largest_vessel_in_microns": 60.0,
    "scales_per_octave": 6.0,
    "approximating_PSF": True,
    "numerical_aperture": 0.95,
    "excitation_wavelength_in_microns": 0.95,
    "sample_index_of_refraction": 1.33,
    "gaussian_to_ideal_ratio": 0.5,
    "spherical_to_annular_ratio": 0.5,
    "energy_projection_mode": "matlab",
    "max_voxels_per_node_energy": 6_000,
    "energy_axis_permutation": [2, 0, 1],
    "microns_per_voxel": [0.916, 0.916, 1.99688],
}

# Strict cross-language probe parity: float64 bitwise equality.
BIT_PARITY_ENERGY_RTOL = 0.0
BIT_PARITY_ENERGY_ATOL = 0.0

# One-voxel recomputation can drift a few ULPs from promoted full-volume oracle vectors.
ORACLE_ENERGY_MAX_ULPS = 8


def crop_tiff_path() -> Path:
    env_path = os.environ.get("SLAVV_CROP_TIFF", "").strip()
    if env_path:
        return Path(env_path).expanduser().resolve()
    return DEFAULT_CROP_TIFF


def crop_oracle_root() -> Path | None:
    env_root = os.environ.get("SLAVV_CROP_ORACLE_ROOT", "").strip()
    if env_root:
        candidate = Path(env_root).expanduser().resolve()
        return candidate if candidate.is_dir() else None
    return DEFAULT_CROP_ORACLE_ROOT if DEFAULT_CROP_ORACLE_ROOT.is_dir() else None


def crop_harness_available() -> bool:
    root = crop_oracle_root()
    return crop_tiff_path().is_file() and root is not None


def linspace_table_path() -> Path | None:
    if LINSPACE_TABLE_FIXTURE.is_file():
        return LINSPACE_TABLE_FIXTURE
    if LINSPACE_TABLE_SCRATCH.is_file():
        return LINSPACE_TABLE_SCRATCH
    return None


def load_crop_image_and_config(
    *,
    params: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    image = load_tiff_volume(crop_tiff_path(), memory_map=True)
    image = np.transpose(image, (2, 0, 1))
    config = _prepare_energy_config(image, params or CROP_EXACT_PARAMS)
    return image, config


def load_crop_oracle_energy() -> tuple[np.ndarray, np.ndarray]:
    oracle_root = crop_oracle_root()
    if oracle_root is None:
        raise FileNotFoundError("crop harness oracle not available")
    oracle = load_oracle_surface(oracle_root)
    payload = load_normalized_matlab_vectors(oracle.matlab_batch_dir, ("energy",))["energy"]
    energy = np.asarray(payload["energy"], dtype=np.float64)
    scales = np.asarray(payload["scale_indices"])
    return energy, scales


def rf_for_scale(config: dict[str, Any], scale: int) -> tuple[int, int, int]:
    octave_at_scales = np.asarray(config["octave_at_scales"])
    scale_resolution_factors = np.asarray(config["scale_resolution_factors"])
    octave = int(octave_at_scales[scale])
    indices = np.where(octave_at_scales == octave)[0]
    rf = tuple(int(v) for v in scale_resolution_factors[indices[0]])
    return rf  # type: ignore[return-value]


def _float64_as_ordered_int(value: float) -> int:
    bits = np.float64(value).view(np.int64).item()
    if bits < 0:
        return int(0x8000000000000000 - bits)
    return int(bits)


def float64_ulp_distance(actual: float, expected: float) -> int:
    """Return the ULP distance between two float64 scalars."""
    if np.float64(actual) == np.float64(expected):
        return 0
    return abs(_float64_as_ordered_int(actual) - _float64_as_ordered_int(expected))


def assert_bit_parity_energy(actual: float, expected: float, *, label: str = "energy") -> None:
    """Require float64 bitwise equality (Python vs live MATLAB probe JSONL)."""
    actual_f = np.float64(actual)
    expected_f = np.float64(expected)
    if actual_f == expected_f:
        return
    npt.assert_allclose(
        actual_f,
        expected_f,
        rtol=BIT_PARITY_ENERGY_RTOL,
        atol=BIT_PARITY_ENERGY_ATOL,
        err_msg=(
            f"{label} bit-parity mismatch: actual={actual!r} expected={expected!r} "
            f"actual_hex={actual_f.view(np.uint64).item():#018x} "
            f"expected_hex={expected_f.view(np.uint64).item():#018x} "
            f"ulp_distance={float64_ulp_distance(actual, expected)}"
        ),
    )


def assert_oracle_energy_parity(
    actual: float,
    expected: float,
    *,
    label: str = "energy",
    max_ulps: int = ORACLE_ENERGY_MAX_ULPS,
) -> None:
    """Compare Python recomputation against a promoted MATLAB oracle scalar."""
    actual_f = np.float64(actual)
    expected_f = np.float64(expected)
    if actual_f == expected_f:
        return
    if float64_ulp_distance(actual_f, expected_f) <= max_ulps:
        return
    npt.assert_allclose(
        actual_f,
        expected_f,
        rtol=BIT_PARITY_ENERGY_RTOL,
        atol=BIT_PARITY_ENERGY_ATOL,
        err_msg=(
            f"{label} oracle parity mismatch: actual={actual!r} expected={expected!r} "
            f"actual_hex={actual_f.view(np.uint64).item():#018x} "
            f"expected_hex={expected_f.view(np.uint64).item():#018x} "
            f"ulp_distance={float64_ulp_distance(actual, expected)} max_ulps={max_ulps}"
        ),
    )


def assert_bit_parity_scale(actual: int, expected: int, *, label: str = "scale") -> None:
    assert int(actual) == int(expected), f"{label} mismatch: actual={actual} expected={expected}"


def probe_voxel(
    image: np.ndarray,
    config: dict[str, Any],
    *,
    voxel_zyx: tuple[int, int, int],
    oracle_scale: int,
) -> dict[str, Any]:
    rf = rf_for_scale(config, oracle_scale)
    chunk_idx = resolve_write_chunk_idx_for_voxel(
        config,
        voxel_zyx=voxel_zyx,
        target_rf_zyx=rf,
    )
    return probe_exact_energy_voxel_at_octave(
        image,
        config,
        voxel_zyx=voxel_zyx,
        target_rf_zyx=rf,
        chunk_idx=chunk_idx,
    )


def compare_voxel_probe_to_oracle(
    probe: dict[str, Any],
    *,
    voxel_zyx: tuple[int, int, int],
    oracle_energy: np.ndarray,
    oracle_scales: np.ndarray,
) -> dict[str, Any]:
    expected_energy = float(oracle_energy[voxel_zyx])
    expected_scale = int(oracle_scales[voxel_zyx])
    actual_energy = float(probe["octave_winner"]["upsampled_energy"])
    actual_scale = int(probe["octave_winner"]["global_scale"])
    energy_ulps = float64_ulp_distance(actual_energy, expected_energy)
    energy_match = energy_ulps <= ORACLE_ENERGY_MAX_ULPS
    scale_match = actual_scale == expected_scale
    return {
        "voxel_zyx": list(voxel_zyx),
        "expected_scale": expected_scale,
        "actual_scale": actual_scale,
        "expected_energy": expected_energy,
        "actual_energy": actual_energy,
        "energy_ulps": energy_ulps,
        "scale_match": scale_match,
        "energy_match": energy_match,
        "passed": scale_match and energy_match,
    }


def probe_result_from_voxel_probe(
    probe: dict[str, Any],
    *,
    request_id: str,
    coordinate_zyx: tuple[int, int, int],
) -> ProbeResult:
    return ProbeResult(
        request_id=request_id,
        coordinate_zyx=coordinate_zyx,
        octave=int(probe["consolidated_octave"]),
        chunk_index=int(probe["chunk_idx"]),
        winner_scale=int(probe["octave_winner"]["global_scale"]),
        winner_energy=float(probe["octave_winner"]["upsampled_energy"]),
        payload=probe,
    )


def probe_result_to_jsonl_record(result: ProbeResult) -> dict[str, Any]:
    """Flat JSONL record for cross-language probe comparison."""
    return {
        "request_id": result.request_id,
        "coordinate_zyx": list(result.coordinate_zyx),
        "octave": result.octave,
        "chunk_index": result.chunk_index,
        "winner_scale": result.winner_scale,
        "winner_energy": result.winner_energy,
    }


def write_probe_jsonl(path: Path, records: list[ProbeResult]) -> Path:
    lines = [json.dumps(probe_result_to_jsonl_record(record), sort_keys=True) for record in records]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path


def load_voxel_regression_cases(path: Path | None = None) -> list[dict[str, Any]]:
    fixture_path = path or VOXEL_REGRESSION_FIXTURE
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    return list(payload["cases"])


def run_voxel_regression_fixture(
    *,
    fixture_path: Path | None = None,
    cases: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    image, config = load_crop_image_and_config()
    oracle_energy, oracle_scales = load_crop_oracle_energy()
    selected = cases if cases is not None else load_voxel_regression_cases(fixture_path)
    results: list[dict[str, Any]] = []
    passed = 0
    for case in selected:
        voxel_zyx = tuple(int(v) for v in case["voxel_zyx"])
        oracle_scale = int(case["oracle_scale"])
        probe = probe_voxel(image, config, voxel_zyx=voxel_zyx, oracle_scale=oracle_scale)
        report = compare_voxel_probe_to_oracle(
            probe,
            voxel_zyx=voxel_zyx,
            oracle_energy=oracle_energy,
            oracle_scales=oracle_scales,
        )
        report["id"] = case.get("id", str(voxel_zyx))
        if "oracle_energy" in case:
            assert_oracle_energy_parity(report["actual_energy"], float(case["oracle_energy"]))
        if report["passed"]:
            passed += 1
        results.append(report)
    return {
        "probed": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "results": results,
    }


def run_mismatch_group_batch(
    probe_requests_path: Path,
    *,
    top_groups: int = 15,
    coordinates_per_group: int = 2,
) -> dict[str, Any]:
    payload = json.loads(probe_requests_path.read_text(encoding="utf-8"))
    groups = [
        group
        for group in payload["groups"]
        if int(group["matlab_scale"]) >= 0
        and int(group["python_scale"]) >= 0
        and int(group["matlab_scale"]) != int(group["python_scale"])
    ]
    groups.sort(key=lambda group: int(group["mismatch_count"]), reverse=True)
    groups = groups[:top_groups]

    image, config = load_crop_image_and_config()
    oracle_energy, oracle_scales = load_crop_oracle_energy()

    results: list[dict[str, Any]] = []
    passed = 0
    total = 0
    for group in groups:
        oracle_scale = int(group["matlab_scale"])
        coords = [tuple(int(v) for v in group["first_coordinate_zyx"])]
        max_coord = tuple(int(v) for v in group["max_delta_coordinate_zyx"])
        if max_coord not in coords and coordinates_per_group > 1:
            coords.append(max_coord)
        for voxel_zyx in coords:
            total += 1
            probe = probe_voxel(image, config, voxel_zyx=voxel_zyx, oracle_scale=oracle_scale)
            report = compare_voxel_probe_to_oracle(
                probe,
                voxel_zyx=voxel_zyx,
                oracle_energy=oracle_energy,
                oracle_scales=oracle_scales,
            )
            report.update(
                {
                    "matlab_scale": oracle_scale,
                    "stored_python_scale": int(group["python_scale"]),
                    "chunk_idx": int(probe["chunk_idx"]),
                    "mismatch_count": int(group["mismatch_count"]),
                    "boundary_class": group["boundary_class"],
                }
            )
            if report["passed"]:
                passed += 1
            results.append(report)

    return {"probed": total, "passed": passed, "failed": total - passed, "results": results}


def select_mismatch_group_requests(
    probe_requests_path: Path,
    *,
    top_groups: int = 15,
    coordinates_per_group: int = 2,
) -> dict[str, Any]:
    """Select deterministic valid scale-winner probes without recomputing Energy."""
    payload = json.loads(probe_requests_path.read_text(encoding="utf-8"))
    groups = [
        group
        for group in payload["groups"]
        if int(group["matlab_scale"]) >= 0
        and int(group["python_scale"]) >= 0
        and int(group["matlab_scale"]) != int(group["python_scale"])
    ]
    groups.sort(key=lambda group: int(group["mismatch_count"]), reverse=True)
    groups = groups[:top_groups]

    _, config = load_crop_image_and_config()
    requests: list[dict[str, Any]] = []
    for group_index, group in enumerate(groups):
        matlab_scale = int(group["matlab_scale"])
        rf_zyx = rf_for_scale(config, matlab_scale)
        coordinates = [tuple(int(value) for value in group["first_coordinate_zyx"])]
        max_coordinate = tuple(int(value) for value in group["max_delta_coordinate_zyx"])
        if coordinates_per_group > 1 and max_coordinate not in coordinates:
            coordinates.append(max_coordinate)
        for coordinate_index, coordinate_zyx in enumerate(coordinates):
            requests.append(
                {
                    "request_id": f"g{group_index:02d}_{coordinate_index}",
                    "voxel_zyx": list(coordinate_zyx),
                    "matlab_scale": matlab_scale,
                    "stored_python_scale": int(group["python_scale"]),
                    "rf_zyx": list(rf_zyx),
                    "rf_matlab_yxz": [rf_zyx[1], rf_zyx[2], rf_zyx[0]],
                    "mismatch_count": int(group["mismatch_count"]),
                    "boundary_class": str(group["boundary_class"]),
                }
            )
    return {"version": 1, "requests": requests}


def export_regression_probe_jsonl(output_path: Path) -> Path:
    image, config = load_crop_image_and_config()
    records: list[ProbeResult] = []
    for case in load_voxel_regression_cases():
        voxel_zyx = tuple(int(v) for v in case["voxel_zyx"])
        oracle_scale = int(case["oracle_scale"])
        probe = probe_voxel(image, config, voxel_zyx=voxel_zyx, oracle_scale=oracle_scale)
        records.append(
            probe_result_from_voxel_probe(
                probe,
                request_id=str(case.get("id", voxel_zyx)),
                coordinate_zyx=voxel_zyx,
            )
        )
    return write_probe_jsonl(output_path, records)


def compare_matlab_python_probe_jsonl(matlab_path: Path, python_path: Path) -> dict[str, Any]:
    return compare_probe_jsonl(matlab_path, python_path)


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MATLAB-Python Energy voxel parity harness")
    subparsers = parser.add_subparsers(dest="command", required=True)

    regression = subparsers.add_parser("regression", help="Run checked-in voxel regression fixture")
    regression.add_argument("--fixture", type=Path, default=VOXEL_REGRESSION_FIXTURE)
    regression.add_argument("--output", type=Path)

    batch = subparsers.add_parser("batch", help="Probe top mismatch groups from prove-exact ledger")
    batch.add_argument("--probe-requests", type=Path, required=True)
    batch.add_argument("--top-groups", type=int, default=15)
    batch.add_argument("--coordinates-per-group", type=int, default=2)
    batch.add_argument("--output", type=Path)

    export = subparsers.add_parser(
        "export-jsonl", help="Export regression probes as JSONL for MATLAB"
    )
    export.add_argument("--output", type=Path, required=True)

    compare = subparsers.add_parser(
        "compare-jsonl", help="Compare MATLAB vs Python probe JSONL files"
    )
    compare.add_argument("--matlab", type=Path, required=True)
    compare.add_argument("--python", type=Path, required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_cli()
    args = parser.parse_args(argv)

    if args.command == "regression":
        summary = run_voxel_regression_fixture(fixture_path=args.fixture)
        if args.output is not None:
            args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"regression: {summary['passed']}/{summary['probed']} passed")
        return 0 if summary["failed"] == 0 else 1

    if args.command == "batch":
        summary = run_mismatch_group_batch(
            args.probe_requests,
            top_groups=args.top_groups,
            coordinates_per_group=args.coordinates_per_group,
        )
        if args.output is not None:
            args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"batch probe: {summary['passed']}/{summary['probed']} passed")
        return 0 if summary["failed"] == 0 else 1

    if args.command == "export-jsonl":
        export_regression_probe_jsonl(args.output)
        print(f"exported probe JSONL: {args.output}")
        return 0

    if args.command == "compare-jsonl":
        report = compare_matlab_python_probe_jsonl(args.matlab, args.python)
        print(json.dumps(report, indent=2))
        return 0 if report["passed"] else 1

    raise AssertionError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
