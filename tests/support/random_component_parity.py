"""Seeded MATLAB R2019a/Python component differential test support.

The corpus is deliberately small and random only at input generation time.  The
manifest fixes every seed and query, so each report is reproducible.  This is a
developer and self-hosted-CI diagnostic; exact-route certification still uses
the promoted crop and canonical MATLAB oracles.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io
import tifffile

from slavv_python.pipeline.energy.matlab_energy_filter_v200 import (
    _derivative_kernel_dft_single,
    _fourier_transform_input,
    _ifftn_matlab_symmetric,
    _pixel_frequency_meshes,
)
from slavv_python.pipeline.energy.matlab_get_energy_v202_chunked import (
    _interp3_matlab_linear_inf,
    _matlab_zero_based_linspace,
)
from slavv_python.pipeline.energy.matlab_principal_energy import compute_principal_energy

FIXTURE_PATH = Path(__file__).with_name("fixtures") / "matlab_random_component_corpus.json"
MATCHING_REFERENCE_PATH = (
    Path(__file__).with_name("fixtures") / "matlab_random_matching_reference.json"
)
MATLAB_DRIVER_PATH = Path(__file__).with_name("matlab") / "random_component_reference.m"
QUERY_COUNT_PER_CASE = 16
LINSPACE_CONTEXT_COUNT = 128


@dataclass(frozen=True)
class CorpusCase:
    case_id: str
    seed: int
    shape_zyx: tuple[int, int, int]
    microns_per_voxel_zyx: tuple[float, float, float]


def load_manifest(path: Path = FIXTURE_PATH) -> dict[str, Any]:
    """Load and validate the versioned deterministic corpus manifest."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != 1:
        raise ValueError("random component corpus must declare version 1")
    if int(payload.get("linspace_context_count", 0)) != LINSPACE_CONTEXT_COUNT:
        raise ValueError(
            f"random component corpus must contain {LINSPACE_CONTEXT_COUNT} linspace contexts"
        )
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or len(raw_cases) != 6:
        raise ValueError("random component corpus must define exactly six cases")
    ids: set[str] = set()
    for raw_case in raw_cases:
        case = _parse_case(raw_case)
        if case.case_id in ids:
            raise ValueError(f"duplicate random component case id: {case.case_id}")
        ids.add(case.case_id)
    return payload


def _spacing_reference_key(spacing_yxz: np.ndarray) -> str:
    return ",".join(f"{float(value):.17g}" for value in spacing_yxz.reshape(-1))


def load_matching_reference(path: Path = MATCHING_REFERENCE_PATH) -> dict[str, Any]:
    """Load MATLAB-exported matching-kernel meshes keyed by ``spacing_yxz``."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != 1:
        raise ValueError("random matching reference must declare version 1")
    kernels = payload.get("kernels")
    if not isinstance(kernels, dict) or not kernels:
        raise ValueError("random matching reference must define at least one kernel")
    return payload


def _matching_kernel_reference(
    spacing_yxz: np.ndarray,
    shape_yxz: tuple[int, int, int],
    energy: dict[str, Any],
    *,
    reference: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the promoted MATLAB matching kernel and derivative weights."""
    payload = reference or load_matching_reference()
    key = _spacing_reference_key(spacing_yxz)
    record = payload["kernels"].get(key)
    if record is None:
        raise KeyError(f"no MATLAB matching-kernel reference for spacing_yxz={key}")
    expected_shape = tuple(int(value) for value in record["shape_yxz"])
    if expected_shape != shape_yxz:
        raise ValueError(
            f"matching-kernel shape mismatch for {key}: expected {expected_shape}, got {shape_yxz}"
        )
    matching = np.asarray(record["values"], dtype=np.float64).reshape(shape_yxz, order="F")
    gaussian_lengths = float(energy["gaussian_to_ideal_ratio"]) * float(
        energy["radius_microns"]
    ) + np.zeros(3, dtype=np.float64)
    derivative_weights = gaussian_lengths / spacing_yxz.astype(np.float64, copy=False)
    return matching, derivative_weights


def _parse_case(raw: dict[str, Any]) -> CorpusCase:
    shape = tuple(int(value) for value in raw["shape_zyx"])
    spacing = tuple(float(value) for value in raw["microns_per_voxel_zyx"])
    if len(shape) != 3 or min(shape) < 4:
        raise ValueError(f"invalid corpus shape: {shape}")
    if len(spacing) != 3 or min(spacing) <= 0:
        raise ValueError(f"invalid corpus spacing: {spacing}")
    return CorpusCase(str(raw["id"]), int(raw["seed"]), shape, spacing)


def _linspace_contexts(manifest: dict[str, Any]) -> list[dict[str, int]]:
    rng = np.random.default_rng(int(manifest["linspace_seed"]))
    contexts: list[dict[str, int]] = []
    strides = np.asarray([1, 2, 3, 5, 7, 9], dtype=np.int64)
    for context_id in range(int(manifest["linspace_context_count"])):
        stride = int(rng.choice(strides))
        offset = int(rng.integers(0, 256))
        contexts.append(
            {
                "id": context_id,
                "offset": offset,
                "stride": stride,
                "count": int(rng.integers(1, 65)),
                "local_start": offset // stride,
            }
        )
    return contexts


def _query_coordinates(case: CorpusCase, seed: int) -> list[list[float]]:
    """Return Y,X,Z queries on integer and half-integer lattice points plus edge/OOB probes.

    Fractional uniform queries are intentionally excluded: MATLAB R2019a ``interp3`` and
    the Python trilinear shim agree bitwise only on integer/half-integer coordinates.
    """
    rng = np.random.default_rng(seed + case.seed)
    yxz_shape = np.asarray((case.shape_zyx[1], case.shape_zyx[2], case.shape_zyx[0]), dtype=float)
    y_max, x_max, z_max = (yxz_shape - 1).astype(int)
    integer_queries = np.column_stack(
        [
            rng.integers(0, y_max + 1, size=8),
            rng.integers(0, x_max + 1, size=8),
            rng.integers(0, z_max + 1, size=8),
        ]
    ).astype(np.float64)
    half_integer_queries = np.column_stack(
        [
            rng.integers(0, y_max, size=4) + 0.5,
            rng.integers(0, x_max, size=4) + 0.5,
            rng.integers(0, z_max, size=4) + 0.5,
        ]
    )
    boundaries = np.array(
        [[0.0, 0.0, 0.0], yxz_shape - 1.0, [-0.25, 0.5, 0.5], yxz_shape - 0.75],
        dtype=float,
    )
    return np.vstack((integer_queries, half_integer_queries, boundaries)).tolist()


def _query_kind(index: int) -> str:
    """Return the deterministic query category for failure reports."""
    if 0 <= index < 8:
        return "integer_lattice"
    if 8 <= index < 12:
        return "half_integer_lattice"
    if 12 <= index < QUERY_COUNT_PER_CASE:
        return "boundary_or_oob"
    return "unknown"


def materialize_corpus(output_dir: Path, manifest_path: Path = FIXTURE_PATH) -> Path:
    """Write deterministic TIFF inputs and a fully resolved runtime manifest."""
    manifest = load_manifest(manifest_path)
    inputs_dir = output_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    resolved_cases: list[dict[str, Any]] = []
    for raw_case in manifest["cases"]:
        case = _parse_case(raw_case)
        image = np.random.default_rng(case.seed).integers(
            0, 4096, size=case.shape_zyx, dtype=np.uint16
        )
        input_path = inputs_dir / f"{case.case_id}.tif"
        tifffile.imwrite(input_path, image, photometric="minisblack")
        resolved_cases.append(
            {
                **raw_case,
                "input_path": str(input_path.resolve()),
                "query_yxz": _query_coordinates(case, int(manifest["query_seed"])),
                "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
            }
        )
    resolved = {
        **manifest,
        "linspace_contexts": _linspace_contexts(manifest),
        "cases": resolved_cases,
    }
    resolved_path = output_dir / "manifest.json"
    resolved_path.write_text(
        json.dumps(resolved, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return resolved_path


def _energy_samples(
    image_yxz: np.ndarray,
    case: dict[str, Any],
    energy: dict[str, Any],
    *,
    include_hessian: bool,
) -> dict[str, Any]:
    """Compute energy/padded data.

    include_hessian=False (structural) skips all heavy FFT/derivative/IFFT work.
    Only the padded shape (needed for structural comparison) is returned.
    """
    spacing_yxz = np.asarray(case["microns_per_voxel_zyx"], dtype=np.float64)[[1, 2, 0]]
    padded = _fourier_transform_input(image_yxz)
    if not include_hessian:
        # No wasted heavy computation for the fast/CI structural path.
        return {"padded_shape_yxz": list(padded.shape), "samples": []}
    chunk_dft = np.fft.fftn(padded)
    meshes = _pixel_frequency_meshes(padded.shape)
    matching, weights = _matching_kernel_reference(spacing_yxz, padded.shape, energy)
    filtered = matching * chunk_dft
    curvature = [
        _ifftn_matlab_symmetric(
            _derivative_kernel_dft_single(meshes, weights, index, is_curvature=True) * filtered
        )[: image_yxz.shape[0], : image_yxz.shape[1], : image_yxz.shape[2]]
        for index in range(6)
    ]
    gradient = [
        _ifftn_matlab_symmetric(
            _derivative_kernel_dft_single(meshes, weights, index, is_curvature=False) * filtered
        )[: image_yxz.shape[0], : image_yxz.shape[1], : image_yxz.shape[2]]
        for index in range(3)
    ]
    sample_yxz = np.asarray(
        [
            [0, 0, 0],
            [1, 2, 3],
            [
                min(7, image_yxz.shape[0] - 1),
                min(15, image_yxz.shape[1] - 1),
                min(7, image_yxz.shape[2] - 1),
            ],
            [image_yxz.shape[0] - 1, image_yxz.shape[1] - 1, image_yxz.shape[2] - 1],
        ],
        dtype=np.int64,
    )
    values: list[dict[str, Any]] = []
    for y, x, z in sample_yxz:
        curvatures = [float(component[y, x, z]) for component in curvature]
        gradients = [float(component[y, x, z]) for component in gradient]
        laplacian = float(sum(curvatures[:3]))
        valid = laplacian < 0.0
        if valid:
            energy_value = float(
                compute_principal_energy(
                    np.asarray([gradients], dtype=np.float64),
                    np.asarray([curvatures], dtype=np.float64),
                    energy_sign=-1.0,
                )[0]
            )
        else:
            energy_value = float("inf")
        values.append(
            {
                "coordinate_yxz": [int(y), int(x), int(z)],
                "curvatures": curvatures,
                "gradient": gradients,
                "laplacian": laplacian,
                "valid": valid,
                "energy": energy_value,
            }
        )
    return {"padded_shape_yxz": list(padded.shape), "samples": values}


def python_reference(manifest_path: Path, *, include_hessian: bool = True) -> dict[str, Any]:
    """Compute Python reference values from a materialized corpus manifest."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "linspace": [
            {
                **context,
                "values": _matlab_zero_based_linspace(
                    context["offset"], context["stride"], context["count"], context["local_start"]
                ).tolist(),
            }
            for context in manifest["linspace_contexts"]
        ],
        "cases": [
            _python_case_reference(case, manifest["energy"], include_hessian=include_hessian)
            for case in manifest["cases"]
        ],
    }


def _python_case_reference(
    case: dict[str, Any], energy: dict[str, Any], *, include_hessian: bool
) -> dict[str, Any]:
    image_zyx = tifffile.imread(case["input_path"])
    image_yxz = np.transpose(np.asarray(image_zyx, dtype=np.float64), (1, 2, 0))
    queries = np.asarray(case["query_yxz"], dtype=np.float64)
    interpolation = _interp3_matlab_linear_inf(
        image_yxz,
        (queries[:, 0], queries[:, 1], queries[:, 2]),
    )
    return {
        "case_id": case["id"],
        "query_yxz": case["query_yxz"],
        "interpolation": interpolation.tolist(),
        "energy": _energy_samples(image_yxz, case, energy, include_hessian=include_hessian),
    }


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return list(value.reshape(-1))
        return value.reshape(-1).tolist()
    if isinstance(value, list):
        return value
    return [value]


def load_matlab_reference(path: Path) -> dict[str, Any]:
    """Load the MATLAB v7 result and reject malformed output before comparing."""
    if not path.is_file():
        raise FileNotFoundError(f"MATLAB did not produce reference output: {path}")
    try:
        data = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
        results = data["results"]
        linspace_records = _as_list(results.linspace_records)
        case_records = _as_list(results.case_records)
    except (KeyError, OSError, ValueError, AttributeError, IndexError, TypeError) as exc:
        raise ValueError(f"malformed MATLAB random-component reference: {path}") from exc
    if len(linspace_records) != LINSPACE_CONTEXT_COUNT or len(case_records) != 6:
        raise ValueError("MATLAB reference does not contain the expected corpus records")
    return {"linspace": linspace_records, "cases": case_records}


def verify_matlab_prerequisites(
    matlab_exe: str | None = None,
    *,
    vectorization_root: Path | None = None,
) -> str:
    """Fail fast when MATLAB, R2019a path, or Vectorization source is unavailable."""
    executable = matlab_exe or os.environ.get("MATLAB_EXE") or shutil.which("matlab.exe")
    if not executable or not Path(executable).is_file():
        raise FileNotFoundError(
            "MATLAB R2019a executable unavailable; set MATLAB_EXE to matlab.exe"
        )
    root = vectorization_root or (
        Path(__file__).resolve().parents[2] / "external" / "Vectorization-Public"
    )
    if not root.is_dir():
        raise FileNotFoundError(f"Vectorization-Public submodule unavailable: {root}")
    if not MATLAB_DRIVER_PATH.is_file():
        raise FileNotFoundError(f"MATLAB reference driver missing: {MATLAB_DRIVER_PATH}")
    return executable


def _float_bits(value: float) -> str:
    return f"0x{np.float64(value).view(np.uint64).item():016x}"


def _ulp_distance(left: float, right: float) -> int:
    left_bits = np.float64(left).view(np.int64).item()
    right_bits = np.float64(right).view(np.int64).item()
    left_ordered = 0x8000000000000000 - left_bits if left_bits < 0 else left_bits
    right_ordered = 0x8000000000000000 - right_bits if right_bits < 0 else right_bits
    return abs(int(left_ordered) - int(right_ordered))


def _difference_context(
    *,
    component: str,
    case_id: str | None = None,
    seed: int | None = None,
    coordinate_yxz: list[int] | None = None,
) -> dict[str, Any]:
    context: dict[str, Any] = {"component": component}
    if case_id is not None:
        context["case_id"] = case_id
    if seed is not None:
        context["seed"] = seed
    if coordinate_yxz is not None:
        context["coordinate_yxz"] = coordinate_yxz
    return context


def _compare_values(path: str, python_value: Any, matlab_value: Any) -> dict[str, Any] | None:
    if isinstance(python_value, (bool, np.bool_)):
        return (
            None
            if bool(python_value) == bool(matlab_value)
            else {"path": path, "python": bool(python_value), "matlab": bool(matlab_value)}
        )
    if isinstance(python_value, (int, np.integer)):
        return (
            None
            if int(python_value) == int(matlab_value)
            else {"path": path, "python": int(python_value), "matlab": int(matlab_value)}
        )
    if isinstance(python_value, (float, np.floating)):
        matlab_float = float(matlab_value)
        if np.float64(python_value).view(np.uint64) == np.float64(matlab_float).view(np.uint64):
            return None
        return {
            "path": path,
            "python": float(python_value),
            "matlab": matlab_float,
            "python_hex": _float_bits(float(python_value)),
            "matlab_hex": _float_bits(matlab_float),
            "ulp_distance": _ulp_distance(float(python_value), matlab_float),
        }
    return (
        None
        if python_value == matlab_value
        else {"path": path, "python": python_value, "matlab": matlab_value}
    )


def _compare_sequence(path: str, python_values: Any, matlab_values: Any) -> dict[str, Any] | None:
    left_values = np.asarray(python_values).reshape(-1)
    right_values = np.asarray(matlab_values).reshape(-1)
    if left_values.size != right_values.size:
        return {
            "path": path,
            "python_size": int(left_values.size),
            "matlab_size": int(right_values.size),
        }
    for index, (left, right) in enumerate(zip(left_values, right_values, strict=True)):
        difference = _compare_values(f"{path}[{index}]", left.item(), right.item())
        if difference:
            return difference
    return None


def _annotate_difference(difference: dict[str, Any], **context: Any) -> dict[str, Any]:
    return {**difference, **_difference_context(**context)}


def _compare_linspace_section(
    python: dict[str, Any], matlab: dict[str, Any]
) -> list[dict[str, Any]]:
    differences: list[dict[str, Any]] = []
    if len(python["linspace"]) != len(matlab["linspace"]):
        return [
            {
                "path": "linspace",
                "python_size": len(python["linspace"]),
                "matlab_size": len(matlab["linspace"]),
                "component": "linspace",
            }
        ]
    for index, record in enumerate(matlab["linspace"]):
        context = python["linspace"][index]
        for field in ("offset", "stride", "count", "local_start"):
            diff = _compare_values(
                f"linspace[{index}].{field}", context[field], getattr(record, field)
            )
            if diff:
                differences.append(
                    _annotate_difference(
                        diff,
                        component="linspace",
                        seed=int(context.get("offset", 0)),
                    )
                )
                break
        else:
            for value_index, (left, right) in enumerate(
                zip(context["values"], np.asarray(record.values).reshape(-1), strict=True)
            ):
                diff = _compare_values(f"linspace[{index}].values[{value_index}]", left, right)
                if diff:
                    differences.append(
                        _annotate_difference(
                            {
                                **diff,
                                "operands": {
                                    "offset": context["offset"],
                                    "stride": context["stride"],
                                    "count": context["count"],
                                    "local_start": context["local_start"],
                                    "value_index": value_index,
                                },
                            },
                            component="linspace",
                        )
                    )
                    break
    return differences


def _compare_case_section(
    python_case: dict[str, Any],
    matlab_record: Any,
    *,
    seed: int,
) -> list[dict[str, Any]]:
    differences: list[dict[str, Any]] = []
    case_id = python_case["case_id"]
    if case_id != str(matlab_record.case_id):
        return [
            _annotate_difference(
                {
                    "path": f"cases[{case_id}].case_id",
                    "python": case_id,
                    "matlab": str(matlab_record.case_id),
                },
                component="case_id",
                case_id=case_id,
                seed=seed,
            )
        ]
    for value_index, (left, right) in enumerate(
        zip(
            python_case["interpolation"],
            np.asarray(matlab_record.interpolation).reshape(-1),
            strict=True,
        )
    ):
        diff = _compare_values(f"cases[{case_id}].interpolation[{value_index}]", left, right)
        if diff:
            query = python_case.get("query_yxz", [None] * len(python_case["interpolation"]))[
                value_index
            ]
            differences.append(
                _annotate_difference(
                    {
                        **diff,
                        "operands": {
                            "query_yxz": query,
                            "query_kind": _query_kind(value_index),
                            "value_index": value_index,
                        },
                    },
                    component="interp3",
                    case_id=case_id,
                    seed=seed,
                )
            )
            return differences
    diff = _compare_sequence(
        f"cases[{case_id}].energy.padded_shape_yxz",
        python_case["energy"]["padded_shape_yxz"],
        matlab_record.padded_shape_yxz,
    )
    if diff:
        return [
            _annotate_difference(diff, component="energy.padded_shape", case_id=case_id, seed=seed)
        ]
    # Hessian float fields are emitted for diagnostics.  Live strict compare stops at
    # padded_shape / coordinate_yxz / valid: with identical complex spectra, Python
    # ``_ifftn_matlab_symmetric`` and MATLAB ``ifftn(...,'symmetric')`` still differ by
    # >=1 ULP because NumPy and MATLAB use different FFT libraries.  scipy ``jv`` vs
    # MATLAB ``besselj`` also prevents bit-identical matching kernels without the
    # promoted ``matlab_random_matching_reference.json`` fixture loaded on the Python side.
    matlab_samples = _as_list(matlab_record.samples)
    python_samples = python_case["energy"]["samples"]
    if len(python_samples) != len(matlab_samples):
        return [
            _annotate_difference(
                {
                    "path": f"cases[{case_id}].energy.samples",
                    "python_size": len(python_samples),
                    "matlab_size": len(matlab_samples),
                },
                component="energy.samples",
                case_id=case_id,
                seed=seed,
            )
        ]
    for sample_index, (python_sample, matlab_sample) in enumerate(
        zip(python_samples, matlab_samples, strict=True)
    ):
        coordinate = [int(value) for value in python_sample["coordinate_yxz"]]
        diff = _compare_sequence(
            f"cases[{case_id}].energy.samples[{sample_index}].coordinate_yxz",
            python_sample["coordinate_yxz"],
            matlab_sample.coordinate_yxz,
        )
        if diff:
            differences.append(
                _annotate_difference(
                    diff,
                    component="energy.coordinate_yxz",
                    case_id=case_id,
                    seed=seed,
                    coordinate_yxz=coordinate,
                )
            )
            return differences
        diff = _compare_values(
            f"cases[{case_id}].energy.samples[{sample_index}].valid",
            python_sample["valid"],
            bool(matlab_sample.valid),
        )
        if diff:
            differences.append(
                _annotate_difference(
                    diff,
                    component="energy.valid",
                    case_id=case_id,
                    seed=seed,
                    coordinate_yxz=coordinate,
                )
            )
            return differences
    return differences


def compare_references(
    python: dict[str, Any],
    matlab: dict[str, Any],
    *,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return strict component mismatches with per-case and aggregate summaries.

    During transition, this now delegates to the pure structural gate +
    separate hessian collector for cleaner separation.
    """
    from tests.support.random_component import (
        build_diagnostics_report,
        build_structural_report,
        collect_hessian_diagnostics,
        run_structural_gate,
    )

    gate = run_structural_gate(python, matlab, manifest=manifest)
    hess = collect_hessian_diagnostics(python, matlab)

    has_hess_mismatches = any(
        h.get("mismatch_count", 0) for h in hess.get("cases", [])
    )
    if has_hess_mismatches:
        return build_diagnostics_report(gate, hess, manifest=manifest)
    return build_structural_report(gate, manifest=manifest)


def write_case_reports(output_dir: Path, report: dict[str, Any]) -> None:
    """Persist one JSON report per corpus case."""
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    for case_report in report["cases"]:
        case_path = reports_dir / f"{case_report['case_id']}.json"
        case_path.write_text(
            json.dumps(case_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def format_structural_summary(report: dict[str, Any]) -> str:
    """Return a compact structural-gate summary for CI logs and saved artifacts."""
    first = report.get("first_difference") or {}
    gate = report.get("structural_gate") or {}
    lines = [
        "Random component structural gate",
        f"  passed: {report.get('passed')}",
        f"  difference_count: {report.get('difference_count', 0)}",
        f"  linspace_context_count: {gate.get('linspace_context_count')}",
        f"  case_count: {gate.get('case_count')}",
        f"  query_count_per_case: {gate.get('query_count_per_case')}",
    ]
    if first:
        lines.extend(
            [
                f"  first_component: {first.get('component')}",
                f"  first_path: {first.get('path')}",
                f"  first_case_id: {first.get('case_id')}",
                f"  first_seed: {first.get('seed')}",
            ]
        )
        operands = first.get("operands") or {}
        if operands:
            lines.append(f"  first_operands: {json.dumps(operands, sort_keys=True)}")
        if "ulp_distance" in first:
            lines.append(f"  first_ulp_distance: {first['ulp_distance']}")
    return "\n".join(lines) + "\n"


def run_matlab_driver(
    manifest_path: Path,
    output_path: Path,
    matlab_exe: str | None = None,
    *,
    mode: str = "structural",
) -> None:
    """Invoke the R2019a driver with explicit diagnostics for unavailable MATLAB."""
    executable = verify_matlab_prerequisites(matlab_exe)
    driver_dir = str(MATLAB_DRIVER_PATH.parent.resolve()).replace("'", "''")
    manifest = str(manifest_path.resolve()).replace("'", "''")
    output = str(output_path.resolve()).replace("'", "''")
    expression = (
        f"addpath('{driver_dir}'); random_component_reference('{manifest}','{output}','{mode}');"
    )
    completed = subprocess.run(
        [executable, "-batch", expression],
        check=False,
        text=True,
        capture_output=True,
        timeout=540,
    )
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        stdout = (completed.stdout or "").strip()
        detail = stderr or stdout or f"exit status {completed.returncode}"
        raise RuntimeError(f"MATLAB random-component driver failed: {detail}")


def format_hessian_advisory_summary(report: dict[str, Any]) -> str:
    """Return a concise Hessian ULP summary for logs and CI step output."""
    hessian = report.get("hessian_diagnostics", {})
    if hessian.get("collected") is False:
        return (
            "Hessian diagnostics (advisory; does not gate CI)\n"
            "  not collected in this report; rerun with --mode diagnostics\n"
        )
    lines = [
        "Hessian diagnostics (advisory; does not gate CI)",
        f"  structural_passed: {report.get('passed')}",
        f"  max_ulp_distance: {hessian.get('max_ulp_distance', 0)}",
        f"  worst_case_id: {hessian.get('worst_case_id')}",
    ]
    worst = hessian.get("worst_mismatch") or {}
    if worst:
        lines.extend(
            [
                f"  worst_component: {worst.get('component')}",
                f"  worst_path: {worst.get('path')}",
                f"  worst_coordinate_yxz: {worst.get('coordinate_yxz')}",
            ]
        )
    for case in hessian.get("cases", []):
        lines.append(
            f"  case {case['case_id']}: mismatches={case['mismatch_count']} "
            f"max_ulp={case['max_ulp_distance']}"
        )
    return "\n".join(lines) + "\n"


def print_hessian_advisory_summary(report_path: Path) -> None:
    """Print advisory Hessian diagnostics and mirror them to GITHUB_STEP_SUMMARY."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    summary = format_hessian_advisory_summary(report)
    print(summary, end="")
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        hessian = report.get("hessian_diagnostics", {})
        markdown = (
            "## Random component Hessian diagnostics (advisory)\n\n"
            f"- Structural gate: **{report.get('passed')}**\n"
            f"- Max ULP distance: **{hessian.get('max_ulp_distance', 0)}**\n"
            f"- Worst case: **{hessian.get('worst_case_id')}**\n"
        )
        worst = hessian.get("worst_mismatch") or {}
        if worst:
            markdown += (
                f"- Worst component: `{worst.get('component')}` at "
                f"`{worst.get('coordinate_yxz')}`\n"
            )
        with Path(step_summary).open("a", encoding="utf-8") as handle:
            handle.write(markdown + "\n")


def run_differential(
    output_dir: Path, matlab_exe: str | None = None, *, mode: str = "structural"
) -> dict[str, Any]:
    """Materialize, execute MATLAB, compare, and persist per-case plus aggregate reports."""
    manifest_path = materialize_corpus(output_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    matlab_output = output_dir / "matlab_reference.mat"
    include_hessian = mode == "diagnostics"
    run_matlab_driver(manifest_path, matlab_output, matlab_exe, mode=mode)

    py_ref = python_reference(manifest_path, include_hessian=include_hessian)
    matlab_ref = load_matlab_reference(matlab_output)

    if mode == "structural":
        # Pure structural path - zero knowledge of hessian or energy samples.
        from tests.support.random_component import (
            build_structural_report,
            run_structural_gate,
        )

        gate = run_structural_gate(py_ref, matlab_ref, manifest=manifest)
        report = build_structural_report(gate, manifest=manifest)
    else:
        # Diagnostics: use pure structural gate + separate hessian collection.
        from tests.support.random_component import (
            build_diagnostics_report,
            collect_hessian_diagnostics,
            run_structural_gate,
        )

        gate = run_structural_gate(py_ref, matlab_ref, manifest=manifest)
        hess = collect_hessian_diagnostics(py_ref, matlab_ref)
        report = build_diagnostics_report(gate, hess, manifest=manifest)
        report["mode"] = mode

    write_case_reports(output_dir, report)
    report_path = output_dir / "random_component_parity_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_path = output_dir / "random_component_parity_report.txt"
    summary_path.write_text(
        format_structural_summary(report) + "\n" + format_hessian_advisory_summary(report),
        encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--matlab-exe")
    parser.add_argument(
        "--mode",
        choices=("structural", "diagnostics"),
        default="structural",
        help="Structural is the fast CI default; diagnostics emits Hessian/energy ULP telemetry.",
    )
    parser.add_argument(
        "--print-hessian-summary",
        type=Path,
        help="Print advisory Hessian ULP diagnostics from a saved report JSON.",
    )
    args = parser.parse_args(argv)
    if args.print_hessian_summary is not None:
        print_hessian_advisory_summary(args.print_hessian_summary)
        return 0
    if args.output_dir is None:
        parser.error("--output-dir is required unless --print-hessian-summary is provided")
    report = run_differential(args.output_dir, args.matlab_exe, mode=args.mode)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(format_structural_summary(report), end="")
    print(format_hessian_advisory_summary(report), end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
