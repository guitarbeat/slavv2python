"""Developer helpers for native-first MATLAB-oracle parity experiments."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from contextlib import suppress
from dataclasses import asdict, dataclass
from itertools import permutations
from pathlib import Path
from shutil import copy2, copytree
from typing import Any, cast

import numpy as np
import pandas as pd
import psutil
from scipy.io import loadmat

REPO_ROOT = Path(__file__).resolve().parents[3]
DEV_RUNS_ROOT = REPO_ROOT / "dev" / "runs"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from source import SLAVVProcessor
from source.core._edges.bridge_vertices import add_vertices_to_edges_matlab_style
from source.core._edges.postprocess import finalize_edges_matlab_style
from source.core._energy.provenance import (
    exact_compatible_energy_origins_text,
    exact_route_gate_description,
    is_exact_compatible_energy_origin,
)
from source.core.edge_candidates import (
    _finalize_matlab_parity_candidates,
    _generate_edge_candidates_matlab_frontier,
)
from source.core.edge_selection import choose_edges_for_workflow
from source.core.vertices import paint_vertex_center_image
from source.io import load_tiff_volume
from source.io.matlab_exact_proof import (
    EXACT_STAGE_ORDER,
    compare_exact_artifacts,
    find_matlab_vector_paths,
    find_single_matlab_batch_dir,
    load_normalized_matlab_vectors,
    load_normalized_python_checkpoints,
    normalize_python_stage_payload,
    render_exact_proof_report,
    sync_exact_vertex_checkpoint_from_matlab,
)
from source.io.matlab_fail_fast import (
    build_candidate_coverage_report,
    build_candidate_snapshot_payload,
    compare_lut_fixture_payload,
    load_builtin_lut_fixture,
    render_candidate_coverage_report,
    render_lut_proof_report,
)
from source.runtime.run_state import (
    atomic_joblib_dump,
    atomic_write_json,
    atomic_write_text,
    fingerprint_array,
    fingerprint_file,
    fingerprint_jsonable,
    load_json_dict,
    stable_json_dumps,
)
from source.utils.safe_unpickle import safe_load

ANALYSIS_DIR = Path("03_Analysis")
ANALYSIS_TABLES_DIR = ANALYSIS_DIR / "tables"
CHECKPOINTS_DIR = Path("02_Output") / "python_results" / "checkpoints"
EXPERIMENT_REFS_DIR = Path("00_Refs")
EXPERIMENT_PARAMS_DIR = Path("01_Params")
HASHES_DIR = ANALYSIS_DIR / "hashes"
NORMALIZED_DIR = ANALYSIS_DIR / "normalized"
METADATA_DIR = Path("99_Metadata")
RUN_MANIFEST_PATH = METADATA_DIR / "run_manifest.json"
ORACLE_MANIFEST_PATH = METADATA_DIR / "oracle_manifest.json"
REPORT_MANIFEST_PATH = METADATA_DIR / "report_manifest.json"
DATASET_MANIFEST_PATH = METADATA_DIR / "dataset_manifest.json"
EXPERIMENT_INDEX_PATH = Path("index.jsonl")
EXPERIMENT_ROOT_SUBDIRS = ("datasets", "oracles", "reports", "runs")
ORACLE_DISCOVERY_STAGES = ("energy", *EXACT_STAGE_ORDER)
DATASET_INPUT_DIR = Path("01_Input")
COMPARISON_REPORT_PATH = ANALYSIS_DIR / "comparison_report.json"
EDGE_CANDIDATE_CHECKPOINT_PATH = CHECKPOINTS_DIR / "checkpoint_edge_candidates.pkl"
EDGE_REPLAY_PROOF_JSON_PATH = ANALYSIS_DIR / "edge_replay_proof.json"
EDGE_REPLAY_PROOF_TEXT_PATH = ANALYSIS_DIR / "edge_replay_proof.txt"
EXACT_PROOF_JSON_PATH = ANALYSIS_DIR / "exact_proof.json"
EXACT_PROOF_TEXT_PATH = ANALYSIS_DIR / "exact_proof.txt"
GAP_DIAGNOSIS_JSON_PATH = ANALYSIS_DIR / "gap_diagnosis.json"
GAP_DIAGNOSIS_TEXT_PATH = ANALYSIS_DIR / "gap_diagnosis.txt"
LUT_PROOF_JSON_PATH = ANALYSIS_DIR / "lut_proof.json"
LUT_PROOF_TEXT_PATH = ANALYSIS_DIR / "lut_proof.txt"
PREFLIGHT_EXACT_JSON_PATH = ANALYSIS_DIR / "preflight_exact.json"
PREFLIGHT_EXACT_TEXT_PATH = ANALYSIS_DIR / "preflight_exact.txt"
RUN_SNAPSHOT_PATH = METADATA_DIR / "run_snapshot.json"
EXPERIMENT_PROVENANCE_PATH = METADATA_DIR / "experiment_provenance.json"
SUMMARY_JSON_PATH = ANALYSIS_DIR / "experiment_summary.json"
SUMMARY_TEXT_PATH = ANALYSIS_DIR / "experiment_summary.txt"
VALIDATED_PARAMS_PATH = METADATA_DIR / "validated_params.json"
CANDIDATE_COVERAGE_JSON_PATH = ANALYSIS_DIR / "candidate_coverage.json"
CANDIDATE_COVERAGE_TEXT_PATH = ANALYSIS_DIR / "candidate_coverage.txt"
CANDIDATE_PROGRESS_JSONL_PATH = ANALYSIS_DIR / "candidate_progress.jsonl"
CANDIDATE_PROGRESS_PLOT_PATH = ANALYSIS_DIR / "candidate_progress.png"
RECORDING_TABLES_INDEX_PATH = ANALYSIS_DIR / "recording_tables.json"
SHARED_PARAMS_PATH = EXPERIMENT_PARAMS_DIR / "shared_params.json"
PYTHON_DERIVED_PARAMS_PATH = EXPERIMENT_PARAMS_DIR / "python_derived_params.json"
PARAM_DIFF_PATH = EXPERIMENT_PARAMS_DIR / "param_diff.json"
HEARTBEAT_INTERVAL_ITERATIONS = 512
DEFAULT_MEMORY_SAFETY_FRACTION = 0.8
EDGE_CANDIDATE_AUDIT_PATH = (
    Path("02_Output") / "python_results" / "stages" / "edges" / "candidate_audit.json"
)
EXACT_SHARED_METHOD_PARAMETER_KEYS = frozenset(
    {
        "approximating_PSF",
        "bandpass_window",
        "direction_tolerance",
        "direction_method",
        "distance_tolerance",
        "distance_tolerance_per_origin_radius",
        "discrete_tracing",
        "edge_method",
        "edge_number_tolerance",
        "energy_method",
        "energy_projection_mode",
        "energy_sign",
        "energy_tolerance",
        "energy_upper_bound",
        "excitation_wavelength_in_microns",
        "gaussian_to_ideal_ratio",
        "length_dilation_ratio",
        "max_edge_energy",
        "max_edge_length_per_origin_radius",
        "max_voxels_per_node",
        "max_voxels_per_node_energy",
        "microns_per_voxel",
        "min_hair_length_in_microns",
        "number_of_edges_per_vertex",
        "numerical_aperture",
        "radius_tolerance",
        "radius_of_largest_vessel_in_microns",
        "radius_of_smallest_vessel_in_microns",
        "sample_index_of_refraction",
        "scales_per_octave",
        "sigma_per_influence_edges",
        "sigma_per_influence_vertices",
        "space_strel_apothem",
        "space_strel_apothem_edges",
        "spherical_to_annular_ratio",
        "step_size_per_origin_radius",
    }
)
EXACT_ALLOWED_ORCHESTRATION_PARAMETER_KEYS = frozenset(
    {
        "comparison_exact_network",
        "energy_storage_format",
        "return_all_scales",
    }
)
EXACT_REQUIRED_PARAMETER_VALUES: dict[str, Any] = {
    "comparison_exact_network": True,
    "direction_method": "hessian",
    "discrete_tracing": False,
    "edge_method": "tracing",
    "energy_method": "hessian",
    "energy_projection_mode": "matlab",
}
EXACT_ROUTE_ARRAY_BYTES_PER_VOXEL: tuple[tuple[str, int], ...] = (
    ("energy", 4),
    ("scale_indices", 2),
    ("vertex_center_image", 4),
    ("energy_map_temp", 4),
    ("energy_map", 4),
    ("branch_order_map", 1),
    ("d_over_r_map", 4),
    ("pointer_map", 4),
    ("vertex_index_map", 4),
    ("size_map", 2),
)
MATLAB_EXACT_EDGE_SOURCE_CONSTANTS: dict[str, Any] = {
    "step_size_per_origin_radius": 1.0,
    "max_edge_energy": 0.0,
    "distance_tolerance_per_origin_radius": 3.0,
    "distance_tolerance": 3.0,
    "edge_number_tolerance": 2,
    "radius_tolerance": 0.5,
    "energy_tolerance": 1.0,
    "direction_tolerance": 1.0,
}


@dataclass(frozen=True)
class RunCounts:
    vertices: int
    edges: int
    strands: int


@dataclass(frozen=True)
class SourceRunSurface:
    run_root: Path
    checkpoints_dir: Path
    comparison_report_path: Path
    validated_params_path: Path
    run_snapshot_path: Path | None


@dataclass(frozen=True)
class ExactProofSourceSurface:
    run_root: Path
    checkpoints_dir: Path
    validated_params_path: Path
    oracle_surface: OracleSurface
    matlab_batch_dir: Path
    matlab_vector_paths: dict[str, Path]


@dataclass(frozen=True)
class ExactPreflightSurface:
    source_surface: ExactProofSourceSurface
    dest_run_root: Path
    image_shape: tuple[int, int, int]


@dataclass(frozen=True)
class OracleSurface:
    oracle_root: Path
    manifest_path: Path | None
    matlab_batch_dir: Path
    matlab_vector_paths: dict[str, Path]
    oracle_id: str | None
    matlab_source_version: str | None
    dataset_hash: str | None


@dataclass(frozen=True)
class DatasetSurface:
    dataset_root: Path
    manifest_path: Path
    input_file: Path
    dataset_hash: str


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _string_or_none(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    return None


def _entity_id_from_path(path: Path) -> str:
    return path.name or path.resolve().name


def _resolve_python_commit() -> str | None:
    try:
        import subprocess

        completed = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            check=False,
            encoding="utf-8",
        )
    except OSError:
        return None
    commit = completed.stdout.strip()
    return commit or None


def _resolve_experiment_root(path: Path) -> Path | None:
    resolved = path.expanduser().resolve()
    for candidate in (resolved, *resolved.parents):
        if candidate.name in EXPERIMENT_ROOT_SUBDIRS:
            return candidate.parent
        if all((candidate / subdir).is_dir() for subdir in EXPERIMENT_ROOT_SUBDIRS):
            return candidate
    return None


def _ensure_experiment_root_layout(root: Path) -> None:
    for subdir in EXPERIMENT_ROOT_SUBDIRS:
        (root / subdir).mkdir(parents=True, exist_ok=True)


def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            records.append(cast("dict[str, Any]", payload))
    return records


def _upsert_index_record(root: Path | None, payload: dict[str, Any]) -> None:
    if root is None:
        return
    _ensure_experiment_root_layout(root)
    index_path = root / EXPERIMENT_INDEX_PATH
    payload_id = _string_or_none(payload.get("id")) or _entity_id_from_path(Path(payload["path"]))
    payload_kind = _string_or_none(payload.get("kind")) or "artifact"
    retained: list[dict[str, Any]] = []
    for existing in _load_jsonl_records(index_path):
        if (
            _string_or_none(existing.get("id")) == payload_id
            and _string_or_none(existing.get("kind")) == payload_kind
        ):
            continue
        retained.append(existing)
    retained.append(payload)
    atomic_write_text(
        index_path,
        "".join(f"{stable_json_dumps(record)}\n" for record in retained),
    )


def _hashable_payload_summary(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        return {
            "kind": "ndarray",
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sha256": fingerprint_array(array),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {
            str(key): _hashable_payload_summary(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_hashable_payload_summary(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _payload_hash(payload: Any) -> str:
    return fingerprint_jsonable(_hashable_payload_summary(payload))


def _resolve_python_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            check=False,
            encoding="utf-8",
        )
    except OSError:
        return None
    commit = completed.stdout.strip()
    return commit or None


def _write_hash_sidecar(path: Path) -> Path:
    hash_path = path.with_name(f"{path.name}.sha256")
    atomic_write_text(hash_path, fingerprint_file(path))
    return hash_path


def _write_payload_hash_sidecar(path: Path, payload: Any) -> Path:
    hash_path = path.with_name(f"{path.name}.sha256")
    atomic_write_text(hash_path, _payload_hash(payload))
    return hash_path


def _write_json_with_hash(path: Path, payload: dict[str, Any]) -> Path:
    atomic_write_json(path, payload)
    _write_hash_sidecar(path)
    return path


def _write_text_with_hash(path: Path, text: str) -> Path:
    atomic_write_text(path, text)
    _write_hash_sidecar(path)
    return path


def _write_joblib_with_hash(path: Path, payload: Any) -> Path:
    atomic_joblib_dump(payload, path)
    _write_payload_hash_sidecar(path, payload)
    return path


def _persist_normalized_payloads(
    dest_run_root: Path,
    *,
    group_name: str,
    payloads: dict[str, Any],
) -> dict[str, str]:
    written: dict[str, str] = {}
    normalized_root = dest_run_root / NORMALIZED_DIR / group_name
    normalized_root.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        artifact_path = normalized_root / f"{name}.pkl"
        _write_joblib_with_hash(artifact_path, payload)
        written[name] = str(artifact_path)
    return written


def _build_exact_param_storage(
    params: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    audit = build_exact_params_audit(params)
    shared_params = cast("dict[str, Any]", dict(audit.get("shared_method_params", {})))
    derived_keys = sorted(set(params) - set(shared_params))
    orchestration_params = {
        key: _normalize_param_value(params[key])
        for key in derived_keys
        if key in EXACT_ALLOWED_ORCHESTRATION_PARAMETER_KEYS
    }
    python_only_params = {
        key: _normalize_param_value(params[key])
        for key in derived_keys
        if key.startswith("parity_")
    }
    unclassified_params = {
        key: _normalize_param_value(params[key])
        for key in derived_keys
        if key not in orchestration_params and key not in python_only_params
    }
    python_derived = {
        "orchestration_params": orchestration_params,
        "python_only_params": python_only_params,
        "unclassified_params": unclassified_params,
    }
    param_diff = {
        "shared_param_count": len(shared_params),
        "shared_param_keys": sorted(shared_params),
        "derived_param_keys": derived_keys,
        "required_exact_values": cast(
            "dict[str, Any]", dict(audit.get("required_exact_values", {}))
        ),
        "required_exact_mismatches": cast(
            "list[dict[str, Any]]",
            list(audit.get("required_exact_mismatches", [])),
        ),
        "disallowed_python_only_keys": cast(
            "list[str]", list(audit.get("disallowed_python_only_keys", []))
        ),
        "unclassified_keys": cast("list[str]", list(audit.get("unclassified_keys", []))),
        "shared_params_hash": fingerprint_jsonable(shared_params),
        "python_derived_params_hash": fingerprint_jsonable(python_derived),
    }
    return shared_params, python_derived, param_diff


def _persist_param_storage(dest_run_root: Path, params: dict[str, Any]) -> dict[str, Any]:
    shared_params, python_derived, param_diff = _build_exact_param_storage(params)
    _write_json_with_hash(dest_run_root / SHARED_PARAMS_PATH, shared_params)
    _write_json_with_hash(dest_run_root / PYTHON_DERIVED_PARAMS_PATH, python_derived)
    _write_json_with_hash(dest_run_root / PARAM_DIFF_PATH, param_diff)
    return {
        "shared_params": shared_params,
        "python_derived_params": python_derived,
        "param_diff": param_diff,
    }


def _load_oracle_surface(oracle_root: Path) -> OracleSurface:
    resolved_root = oracle_root.expanduser().resolve()
    manifest_path = resolved_root / ORACLE_MANIFEST_PATH
    manifest = load_json_dict(manifest_path)
    matlab_batch_dir = find_single_matlab_batch_dir(resolved_root)
    return OracleSurface(
        oracle_root=resolved_root,
        manifest_path=manifest_path if manifest_path.is_file() else None,
        matlab_batch_dir=matlab_batch_dir,
        matlab_vector_paths=find_matlab_vector_paths(matlab_batch_dir, ORACLE_DISCOVERY_STAGES),
        oracle_id=_string_or_none(manifest.get("oracle_id")) if manifest else None,
        matlab_source_version=(
            _string_or_none(manifest.get("matlab_source_version")) if manifest else None
        ),
        dataset_hash=_string_or_none(manifest.get("dataset_hash")) if manifest else None,
    )


def _load_dataset_surface(dataset_root: Path) -> DatasetSurface:
    resolved_root = dataset_root.expanduser().resolve()
    manifest_path = resolved_root / DATASET_MANIFEST_PATH
    manifest = load_json_dict(manifest_path)
    if manifest is None:
        raise ValueError(f"missing dataset manifest: {manifest_path}")
    stored_input = _string_or_none(manifest.get("stored_input_file"))
    dataset_hash = _string_or_none(manifest.get("dataset_hash"))
    if stored_input is None:
        raise ValueError(f"dataset manifest is missing stored_input_file: {manifest_path}")
    if dataset_hash is None:
        raise ValueError(f"dataset manifest is missing dataset_hash: {manifest_path}")
    input_file = Path(stored_input).expanduser().resolve()
    if not input_file.is_file():
        raise ValueError(f"dataset input file not found: {input_file}")
    actual_hash = fingerprint_file(input_file)
    if actual_hash != dataset_hash:
        raise ValueError(
            "dataset manifest hash does not match stored input file: "
            f"expected {dataset_hash}, found {actual_hash}"
        )
    return DatasetSurface(
        dataset_root=resolved_root,
        manifest_path=manifest_path,
        input_file=input_file,
        dataset_hash=dataset_hash,
    )


def _load_matlab_settings_payload(path: Path) -> dict[str, Any]:
    payload = loadmat(path, squeeze_me=True, struct_as_record=False)
    return {str(key): value for key, value in payload.items() if not str(key).startswith("__")}


def _normalize_matlab_setting_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _normalize_matlab_setting_value(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_normalize_matlab_setting_value(item) for item in value]
    if isinstance(value, list):
        return [_normalize_matlab_setting_value(item) for item in value]
    return value


def _settings_timestamp_token_from_vector_name(vector_name: str, stage: str) -> str | None:
    stem = Path(vector_name).stem
    prefix_options = [f"{stage}_", f"curated_{stage}_"]
    for prefix in prefix_options:
        if not stem.startswith(prefix):
            continue
        remainder = stem[len(prefix) :]
        timestamp = remainder.split("_", 1)[0]
        if timestamp:
            return timestamp
    return None


def _select_oracle_settings_paths(oracle_surface: OracleSurface) -> dict[str, Path]:
    settings_dir = oracle_surface.matlab_batch_dir / "settings"
    if not settings_dir.is_dir():
        raise ValueError(f"missing MATLAB settings directory: {settings_dir}")

    energy_candidates = sorted(settings_dir.glob("energy_*.mat"))
    if len(energy_candidates) != 1:
        joined = ", ".join(str(path) for path in energy_candidates)
        raise ValueError(f"expected one energy settings file under {settings_dir}, found: {joined}")

    selected_paths = {"energy": energy_candidates[0]}
    for stage in ("vertices", "edges", "network"):
        vector_path = oracle_surface.matlab_vector_paths[stage]
        timestamp = _settings_timestamp_token_from_vector_name(vector_path.name, stage)
        if timestamp is None:
            raise ValueError(
                f"could not infer {stage} settings timestamp from vector file: {vector_path.name}"
            )
        settings_path = settings_dir / f"{stage}_{timestamp}.mat"
        if not settings_path.is_file():
            raise ValueError(f"missing MATLAB {stage} settings file: {settings_path}")
        selected_paths[stage] = settings_path
    return selected_paths


def derive_exact_params_from_oracle(
    oracle_surface: OracleSurface,
) -> tuple[dict[str, Any], dict[str, str], dict[str, dict[str, Any]]]:
    settings_paths = _select_oracle_settings_paths(oracle_surface)
    settings_payloads = {
        stage: _load_matlab_settings_payload(path) for stage, path in settings_paths.items()
    }
    energy_settings = settings_payloads["energy"]
    vertex_settings = settings_payloads["vertices"]
    edge_settings = settings_payloads["edges"]

    params: dict[str, Any] = {
        "comparison_exact_network": True,
        "direction_method": "hessian",
        "discrete_tracing": False,
        "edge_method": "tracing",
        "energy_method": "hessian",
        "energy_projection_mode": "matlab",
        "microns_per_voxel": _normalize_matlab_setting_value(energy_settings["microns_per_voxel"]),
        "radius_of_smallest_vessel_in_microns": _normalize_matlab_setting_value(
            energy_settings["radius_of_smallest_vessel_in_microns"]
        ),
        "radius_of_largest_vessel_in_microns": _normalize_matlab_setting_value(
            energy_settings["radius_of_largest_vessel_in_microns"]
        ),
        "sample_index_of_refraction": _normalize_matlab_setting_value(
            energy_settings["sample_index_of_refraction"]
        ),
        "numerical_aperture": _normalize_matlab_setting_value(
            energy_settings["numerical_aperture"]
        ),
        "excitation_wavelength_in_microns": _normalize_matlab_setting_value(
            energy_settings["excitation_wavelength_in_microns"]
        ),
        "scales_per_octave": _normalize_matlab_setting_value(energy_settings["scales_per_octave"]),
        "max_voxels_per_node_energy": _normalize_matlab_setting_value(
            energy_settings["max_voxels_per_node_energy"]
        ),
        "gaussian_to_ideal_ratio": _normalize_matlab_setting_value(
            energy_settings["gaussian_to_ideal_ratio"]
        ),
        "spherical_to_annular_ratio": _normalize_matlab_setting_value(
            energy_settings["spherical_to_annular_ratio"]
        ),
        "approximating_PSF": bool(
            _normalize_matlab_setting_value(energy_settings["approximating_PSF"])
        ),
        "space_strel_apothem": _normalize_matlab_setting_value(
            vertex_settings["space_strel_apothem"]
        ),
        "energy_upper_bound": _normalize_matlab_setting_value(
            vertex_settings["energy_upper_bound"]
        ),
        "max_voxels_per_node": _normalize_matlab_setting_value(
            vertex_settings["max_voxels_per_node"]
        ),
        "length_dilation_ratio": _normalize_matlab_setting_value(
            vertex_settings["length_dilation_ratio"]
        ),
        "max_edge_length_per_origin_radius": _normalize_matlab_setting_value(
            edge_settings["max_edge_length_per_origin_radius"]
        ),
        "space_strel_apothem_edges": _normalize_matlab_setting_value(
            edge_settings["space_strel_apothem_edges"]
        ),
        "number_of_edges_per_vertex": _normalize_matlab_setting_value(
            edge_settings["number_of_edges_per_vertex"]
        ),
    }
    params.update(MATLAB_EXACT_EDGE_SOURCE_CONSTANTS)
    return (
        params,
        {stage: str(path) for stage, path in settings_paths.items()},
        {
            stage: {
                str(key): _normalize_matlab_setting_value(value) for key, value in payload.items()
            }
            for stage, payload in settings_payloads.items()
        },
    )


def _materialize_dataset_record(
    experiment_root: Path | None,
    *,
    dataset_hash: str | None,
    dataset_file: Path | None,
) -> str | None:
    resolved_hash = dataset_hash
    if resolved_hash is None and dataset_file is not None and dataset_file.is_file():
        resolved_hash = fingerprint_file(dataset_file)
    if experiment_root is None or resolved_hash is None:
        return resolved_hash

    _ensure_experiment_root_layout(experiment_root)
    dataset_root = experiment_root / "datasets" / resolved_hash
    dataset_root.mkdir(parents=True, exist_ok=True)
    input_file: Path | None = None
    input_bytes: int | None = None
    if dataset_file is not None and dataset_file.is_file():
        input_root = dataset_root / DATASET_INPUT_DIR
        input_root.mkdir(parents=True, exist_ok=True)
        input_file = input_root / dataset_file.name
        if input_file.exists():
            existing_hash = fingerprint_file(input_file)
            if existing_hash != resolved_hash:
                raise ValueError(
                    "dataset root already contains a different payload for this dataset hash: "
                    f"{input_file}"
                )
        else:
            copy2(dataset_file, input_file)
        _write_hash_sidecar(input_file)
        input_bytes = input_file.stat().st_size

    manifest_path = dataset_root / DATASET_MANIFEST_PATH
    existing_manifest = load_json_dict(manifest_path) or {}
    manifest_payload = {
        "manifest_version": 1,
        "kind": "dataset",
        "dataset_hash": resolved_hash,
        "dataset_root": str(dataset_root),
        "stored_input_file": (
            str(input_file)
            if input_file is not None
            else _string_or_none(existing_manifest.get("stored_input_file"))
        ),
        "input_filename": (
            dataset_file.name
            if dataset_file is not None and dataset_file.is_file()
            else _string_or_none(existing_manifest.get("input_filename"))
        ),
        "input_bytes": (
            input_bytes if input_bytes is not None else existing_manifest.get("input_bytes")
        ),
        "source_file": (
            str(dataset_file)
            if dataset_file is not None
            else _string_or_none(existing_manifest.get("source_file"))
        ),
        "retention": "preserved",
        "timestamps": {
            "created_at": _string_or_none(existing_manifest.get("created_at"))
            or _string_or_none(
                cast("dict[str, Any]", dict(existing_manifest.get("timestamps", {}))).get(
                    "created_at"
                )
            )
            or _now_iso(),
            "updated_at": _now_iso(),
        },
    }
    _write_json_with_hash(manifest_path, manifest_payload)
    _upsert_index_record(
        experiment_root,
        {
            "kind": "dataset",
            "id": resolved_hash,
            "path": str(dataset_root),
            "status": "ready",
            "dataset_hash": resolved_hash,
            "updated_at": manifest_payload["timestamps"]["updated_at"],
        },
    )
    return resolved_hash


def _materialize_oracle_root(
    *,
    matlab_batch_dir: Path,
    oracle_root: Path,
    dataset_hash: str | None,
    oracle_id: str | None,
    matlab_source_version: str | None,
) -> OracleSurface:
    oracle_root = oracle_root.expanduser().resolve()
    (oracle_root / METADATA_DIR).mkdir(parents=True, exist_ok=True)
    normalized_root = oracle_root / NORMALIZED_DIR
    hashes_root = oracle_root / HASHES_DIR
    normalized_root.mkdir(parents=True, exist_ok=True)
    hashes_root.mkdir(parents=True, exist_ok=True)

    raw_results_root = oracle_root / "01_Input" / "matlab_results"
    raw_results_root.mkdir(parents=True, exist_ok=True)
    raw_batch_dir = raw_results_root / matlab_batch_dir.name
    if not raw_batch_dir.exists():
        copytree(matlab_batch_dir, raw_batch_dir)

    vector_paths = find_matlab_vector_paths(raw_batch_dir, ORACLE_DISCOVERY_STAGES)
    normalized_payloads = load_normalized_matlab_vectors(raw_batch_dir, EXACT_STAGE_ORDER)
    normalized_artifacts = _persist_normalized_payloads(
        oracle_root,
        group_name="oracle",
        payloads=normalized_payloads,
    )
    raw_vector_hashes: dict[str, str] = {}
    for stage, path in vector_paths.items():
        raw_vector_hash = fingerprint_file(path)
        raw_vector_hashes[stage] = raw_vector_hash
        atomic_write_text(hashes_root / f"oracle_raw_{stage}.sha256", raw_vector_hash)

    resolved_oracle_id = (
        oracle_id or f"{matlab_batch_dir.name}_{fingerprint_jsonable(raw_vector_hashes)[:12]}"
    )
    manifest_payload = {
        "manifest_version": 1,
        "kind": "matlab_oracle",
        "oracle_id": resolved_oracle_id,
        "oracle_root": str(oracle_root),
        "dataset_hash": dataset_hash,
        "matlab_source_version": matlab_source_version,
        "matlab_batch_dir": str(raw_batch_dir),
        "raw_vector_hashes": raw_vector_hashes,
        "normalized_artifacts": normalized_artifacts,
        "timestamps": {
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
        },
    }
    _write_json_with_hash(oracle_root / ORACLE_MANIFEST_PATH, manifest_payload)
    experiment_root = _resolve_experiment_root(oracle_root)
    _upsert_index_record(
        experiment_root,
        {
            "kind": "matlab_oracle",
            "id": resolved_oracle_id,
            "path": str(oracle_root),
            "status": "ready",
            "dataset_hash": dataset_hash,
            "updated_at": manifest_payload["timestamps"]["updated_at"],
        },
    )
    return OracleSurface(
        oracle_root=oracle_root,
        manifest_path=oracle_root / ORACLE_MANIFEST_PATH,
        matlab_batch_dir=raw_batch_dir,
        matlab_vector_paths=vector_paths,
        oracle_id=resolved_oracle_id,
        matlab_source_version=matlab_source_version,
        dataset_hash=dataset_hash,
    )


def _resolve_oracle_surface(
    source_run_root: Path,
    *,
    oracle_root: Path | None,
) -> OracleSurface:
    if oracle_root is not None:
        return _load_oracle_surface(oracle_root)
    run_manifest = load_json_dict(source_run_root / RUN_MANIFEST_PATH)
    manifest_oracle_root = (
        _string_or_none(run_manifest.get("oracle_root")) if run_manifest else None
    )
    if manifest_oracle_root is not None:
        return _load_oracle_surface(Path(manifest_oracle_root))

    matlab_batch_dir = find_single_matlab_batch_dir(source_run_root)
    experiment_root = _resolve_experiment_root(source_run_root)
    dataset_hash = _dataset_hash_from_run_root(source_run_root)
    matlab_source_version = (
        _string_or_none(run_manifest.get("matlab_source_version"))
        if run_manifest is not None
        else None
    ) or "external/Vectorization-Public"
    if experiment_root is None:
        return _load_oracle_surface(source_run_root)

    _ensure_experiment_root_layout(experiment_root)
    resolved_oracle_id = (
        f"{matlab_batch_dir.name}_"
        f"{fingerprint_file(find_matlab_vector_paths(matlab_batch_dir, ORACLE_DISCOVERY_STAGES)['edges'])[:12]}"
    )
    promoted_oracle_root = experiment_root / "oracles" / resolved_oracle_id
    return _materialize_oracle_root(
        matlab_batch_dir=matlab_batch_dir,
        oracle_root=promoted_oracle_root,
        dataset_hash=dataset_hash,
        oracle_id=resolved_oracle_id,
        matlab_source_version=matlab_source_version,
    )


def _dataset_hash_from_run_root(run_root: Path) -> str | None:
    run_manifest = load_json_dict(run_root / RUN_MANIFEST_PATH)
    if run_manifest is not None:
        dataset_hash = _string_or_none(run_manifest.get("dataset_hash"))
        if dataset_hash is not None:
            return dataset_hash
    run_snapshot = load_json_dict(run_root / RUN_SNAPSHOT_PATH)
    if run_snapshot is not None:
        dataset_hash = _string_or_none(run_snapshot.get("input_fingerprint"))
        if dataset_hash is not None:
            return dataset_hash
    return None


def _write_run_manifest(
    dest_run_root: Path,
    *,
    run_kind: str,
    status: str,
    command: str,
    dataset_hash: str | None,
    oracle_surface: OracleSurface | None,
    params_payload: dict[str, Any] | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_manifest = load_json_dict(dest_run_root / RUN_MANIFEST_PATH) or {}
    run_id = _string_or_none(source_manifest.get("run_id")) or _entity_id_from_path(dest_run_root)
    experiment_root = _resolve_experiment_root(dest_run_root)
    _materialize_dataset_record(
        experiment_root,
        dataset_hash=dataset_hash,
        dataset_file=None,
    )
    timestamps = cast("dict[str, Any]", dict(source_manifest.get("timestamps", {})))
    created_at = (
        _string_or_none(timestamps.get("created_at"))
        or _string_or_none(source_manifest.get("created_at"))
        or _now_iso()
    )
    updated_at = _now_iso()
    manifest_payload = {
        "manifest_version": 1,
        "kind": run_kind,
        "run_id": run_id,
        "command": command,
        "run_root": str(dest_run_root),
        "status": status,
        "dataset_hash": dataset_hash,
        "oracle_id": oracle_surface.oracle_id if oracle_surface is not None else None,
        "oracle_root": str(oracle_surface.oracle_root) if oracle_surface is not None else None,
        "python_commit": _string_or_none(source_manifest.get("python_commit"))
        or _resolve_python_commit(),
        "matlab_source_version": (
            oracle_surface.matlab_source_version if oracle_surface is not None else None
        ),
        "timestamps": {
            "created_at": created_at,
            "updated_at": updated_at,
            "completed_at": timestamps.get("completed_at"),
        },
        "retention": "disposable" if dest_run_root.parent.name == "runs" else "promoted",
        "promotion_state": _string_or_none(source_manifest.get("promotion_state")) or "ephemeral",
        "shared_params_hash": (
            fingerprint_jsonable(load_json_dict(dest_run_root / SHARED_PARAMS_PATH) or {})
            if (dest_run_root / SHARED_PARAMS_PATH).is_file()
            else None
        ),
        "python_derived_params_hash": (
            fingerprint_jsonable(load_json_dict(dest_run_root / PYTHON_DERIVED_PARAMS_PATH) or {})
            if (dest_run_root / PYTHON_DERIVED_PARAMS_PATH).is_file()
            else None
        ),
        "params_path": str(dest_run_root / VALIDATED_PARAMS_PATH)
        if params_payload is not None
        else None,
        "analysis_dir": str(dest_run_root / ANALYSIS_DIR),
        "normalized_dir": str(dest_run_root / NORMALIZED_DIR),
        "hashes_dir": str(dest_run_root / HASHES_DIR),
        "stage_metrics": source_manifest.get("stage_metrics", {}),
    }
    if extra:
        manifest_payload.update(extra)
    _write_json_with_hash(dest_run_root / RUN_MANIFEST_PATH, manifest_payload)
    _upsert_index_record(
        experiment_root,
        {
            "kind": run_kind,
            "id": run_id,
            "command": command,
            "status": status,
            "path": str(dest_run_root),
            "dataset_hash": dataset_hash,
            "oracle_id": oracle_surface.oracle_id if oracle_surface is not None else None,
            "updated_at": updated_at,
        },
    )
    return manifest_payload


def build_parser() -> argparse.ArgumentParser:
    """Build the developer parity experiment parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Developer helpers for rerunning and proving native-first exact-route parity "
            "against an existing staged comparison run."
        ),
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    rerun = subparsers.add_parser(
        "rerun-python",
        help="Copy a reusable Python checkpoint surface into a fresh run root and rerun from edges or network.",
    )
    rerun.add_argument(
        "--source-run-root",
        required=True,
        help="Existing staged comparison run root that still retains python checkpoints.",
    )
    rerun.add_argument(
        "--dest-run-root",
        required=True,
        help=f"Fresh destination run root for the current-code experiment. Recommended: {DEV_RUNS_ROOT}",
    )
    rerun.add_argument(
        "--input",
        help=(
            "Override the input volume path. If omitted, the script resolves the path from "
            "99_Metadata/run_snapshot.json provenance."
        ),
    )
    rerun.add_argument(
        "--rerun-from",
        choices=("edges", "network"),
        default="edges",
        help="Pipeline stage to recompute after copying reusable checkpoints.",
    )
    rerun.add_argument(
        "--params-file",
        help=(
            "Optional JSON parameters file. Defaults to 99_Metadata/validated_params.json "
            "from the source run root."
        ),
    )
    rerun.set_defaults(handler=_handle_rerun_python)

    summarize = subparsers.add_parser(
        "summarize",
        help="Print the saved experiment summary for a destination run root.",
    )
    summarize.add_argument(
        "--run-root",
        required=True,
        help="Run root containing 03_Analysis/experiment_summary.{txt,json}.",
    )
    summarize.set_defaults(handler=_handle_summarize)

    normalize = subparsers.add_parser(
        "normalize-recordings",
        help="Flatten recorded run artifacts into CSV/JSONL tables under 03_Analysis/tables.",
    )
    normalize.add_argument(
        "--run-root",
        required=True,
        help="Run root containing run snapshot, manifest, and optional edge-analysis recordings.",
    )
    normalize.set_defaults(handler=_handle_normalize_recordings)

    diagnose = subparsers.add_parser(
        "diagnose-gaps",
        help="Join candidate coverage with origin-level diagnostics to surface gap hotspots.",
    )
    diagnose.add_argument(
        "--run-root",
        required=True,
        help="Run root containing candidate_coverage.json and optional origin-level diagnostics.",
    )
    diagnose.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum hotspot rows to include for missing and extra gaps.",
    )
    diagnose.set_defaults(handler=_handle_diagnose_gaps)

    prove = subparsers.add_parser(
        "prove-exact",
        help="Compare rerun Python checkpoints against preserved raw MATLAB vectors on the native-first exact route.",
    )
    prove.add_argument(
        "--source-run-root",
        required=True,
        help="Existing staged comparison run root containing raw MATLAB vectors and exact-route provenance.",
    )
    prove.add_argument(
        "--oracle-root",
        help=(
            "Optional promoted oracle root under <experiment-root>/oracles. "
            "When omitted, the runner falls back to any oracle recorded in the source run "
            "manifest, then to embedded MATLAB vectors under the source run root."
        ),
    )
    prove.add_argument(
        "--dest-run-root",
        required=True,
        help=(
            "Destination run root containing the current-code rerun Python checkpoints to prove. "
            f"Recommended: {DEV_RUNS_ROOT}"
        ),
    )
    prove.add_argument(
        "--stage",
        choices=(*EXACT_STAGE_ORDER, "all"),
        default="all",
        help="Proof surface to compare. Defaults to all exact-route stages.",
    )
    prove.add_argument(
        "--report-path",
        help=(
            "Optional proof report override. Defaults to 03_Analysis/exact_proof.{json,txt} "
            "under the destination run root."
        ),
    )
    prove.set_defaults(handler=_handle_prove_exact)

    preflight = subparsers.add_parser(
        "preflight-exact",
        help="Validate exact-route source/destination surfaces, memory budget, and process collisions.",
    )
    preflight.add_argument("--source-run-root", required=True)
    preflight.add_argument(
        "--oracle-root",
        help="Optional promoted oracle root that replaces embedded MATLAB vectors from the source run.",
    )
    preflight.add_argument(
        "--dest-run-root",
        required=True,
        help=f"Destination run root for the preflighted experiment. Recommended: {DEV_RUNS_ROOT}",
    )
    preflight.add_argument(
        "--memory-safety-fraction",
        type=float,
        default=DEFAULT_MEMORY_SAFETY_FRACTION,
        help="Refuse the run if projected exact-route memory exceeds this fraction of available RAM.",
    )
    preflight.add_argument(
        "--force",
        action="store_true",
        help="Ignore destination-root process collisions during preflight.",
    )
    preflight.set_defaults(handler=_handle_preflight_exact)

    prove_luts = subparsers.add_parser(
        "prove-luts",
        help="Compare the shared Python watershed LUT builder against the checked-in fixture surface.",
    )
    prove_luts.add_argument("--source-run-root", required=True)
    prove_luts.add_argument(
        "--oracle-root",
        help="Optional promoted oracle root that replaces embedded MATLAB vectors from the source run.",
    )
    prove_luts.add_argument(
        "--dest-run-root",
        required=True,
        help=f"Destination run root for LUT proof artifacts. Recommended: {DEV_RUNS_ROOT}",
    )
    prove_luts.set_defaults(handler=_handle_prove_luts)

    capture = subparsers.add_parser(
        "capture-candidates",
        help="Run only exact candidate generation and write a slim candidate checkpoint plus coverage report.",
    )
    capture.add_argument("--source-run-root", required=True)
    capture.add_argument(
        "--oracle-root",
        help="Optional promoted oracle root that replaces embedded MATLAB vectors from the source run.",
    )
    capture.add_argument(
        "--dest-run-root",
        required=True,
        help=f"Destination run root for candidate artifacts. Recommended: {DEV_RUNS_ROOT}",
    )
    capture.add_argument(
        "--debug-maps",
        action="store_true",
        help="Include full-volume debug maps in checkpoint_edge_candidates.pkl.",
    )
    capture.set_defaults(handler=_handle_capture_candidates)

    replay = subparsers.add_parser(
        "replay-edges",
        help="Replay exact edge choosing from a saved candidate snapshot without regenerating candidates.",
    )
    replay.add_argument("--source-run-root", required=True)
    replay.add_argument(
        "--oracle-root",
        help="Optional promoted oracle root that replaces embedded MATLAB vectors from the source run.",
    )
    replay.add_argument(
        "--dest-run-root",
        required=True,
        help=f"Destination run root for replay artifacts. Recommended: {DEV_RUNS_ROOT}",
    )
    replay.set_defaults(handler=_handle_replay_edges)

    fail_fast = subparsers.add_parser(
        "fail-fast",
        help="Run the fail-fast exact parity funnel and stop at the first failing gate.",
    )
    fail_fast.add_argument("--source-run-root", required=True)
    fail_fast.add_argument(
        "--oracle-root",
        help="Optional promoted oracle root that replaces embedded MATLAB vectors from the source run.",
    )
    fail_fast.add_argument(
        "--dest-run-root",
        required=True,
        help=f"Destination run root for fail-fast artifacts. Recommended: {DEV_RUNS_ROOT}",
    )
    fail_fast.add_argument(
        "--memory-safety-fraction",
        type=float,
        default=DEFAULT_MEMORY_SAFETY_FRACTION,
    )
    fail_fast.add_argument("--debug-maps", action="store_true")
    fail_fast.add_argument("--force", action="store_true")
    fail_fast.set_defaults(handler=_handle_fail_fast)

    promote_oracle = subparsers.add_parser(
        "promote-oracle",
        help="Copy preserved MATLAB vectors into a standalone oracle root with manifest and hashes.",
    )
    promote_oracle.add_argument("--matlab-batch-dir", required=True)
    promote_oracle.add_argument("--oracle-root", required=True)
    promote_oracle.add_argument("--dataset-file")
    promote_oracle.add_argument("--dataset-hash")
    promote_oracle.add_argument("--oracle-id")
    promote_oracle.add_argument("--matlab-source-version", default="external/Vectorization-Public")
    promote_oracle.set_defaults(handler=_handle_promote_oracle)

    promote_dataset = subparsers.add_parser(
        "promote-dataset",
        help="Copy an immutable input volume into datasets/ and register its manifest and hash.",
    )
    promote_dataset.add_argument("--dataset-file", required=True)
    promote_dataset.add_argument(
        "--experiment-root",
        required=True,
        help="Structured experiment root containing datasets/, oracles/, reports/, and runs/.",
    )
    promote_dataset.set_defaults(handler=_handle_promote_dataset)

    init_exact_run = subparsers.add_parser(
        "init-exact-run",
        help=(
            "Seed a fresh exact-route source run from a promoted dataset and oracle by "
            "rerunning Python through energy or vertices."
        ),
    )
    init_exact_run.add_argument("--dataset-root", required=True)
    init_exact_run.add_argument("--oracle-root", required=True)
    init_exact_run.add_argument(
        "--dest-run-root",
        required=True,
        help=f"Fresh destination run root for the exact bootstrap. Recommended: {DEV_RUNS_ROOT}",
    )
    init_exact_run.add_argument(
        "--stop-after",
        choices=("energy", "vertices"),
        default="vertices",
        help="Bootstrap depth. 'vertices' is the maintained default for exact edge work.",
    )
    init_exact_run.add_argument(
        "--energy-storage-format",
        choices=("auto", "npy", "zarr"),
        default="npy",
        help="Orchestration-only energy checkpoint storage for the bootstrap run.",
    )
    init_exact_run.set_defaults(handler=_handle_init_exact_run)

    promote_report = subparsers.add_parser(
        "promote-report",
        help="Copy a disposable run's analysis surface into reports/ with a report manifest.",
    )
    promote_report.add_argument("--run-root", required=True)
    promote_report.add_argument("--report-root")
    promote_report.set_defaults(handler=_handle_promote_report)
    return parser


def validate_source_run_surface(source_run_root: Path) -> SourceRunSurface:
    """Validate the reusable staged source surface for a Python rerun."""
    run_root = source_run_root.resolve()
    checkpoints_dir = run_root / CHECKPOINTS_DIR
    comparison_report_path = run_root / COMPARISON_REPORT_PATH
    validated_params_path = run_root / VALIDATED_PARAMS_PATH
    run_snapshot_path = run_root / RUN_SNAPSHOT_PATH

    missing: list[Path] = []
    if not checkpoints_dir.is_dir():
        missing.append(checkpoints_dir)
    if not comparison_report_path.is_file():
        missing.append(comparison_report_path)
    if not validated_params_path.is_file():
        missing.append(validated_params_path)
    if missing:
        joined = ", ".join(str(path) for path in missing)
        raise ValueError(f"source run root is missing required artifacts: {joined}")

    return SourceRunSurface(
        run_root=run_root,
        checkpoints_dir=checkpoints_dir,
        comparison_report_path=comparison_report_path,
        validated_params_path=validated_params_path,
        run_snapshot_path=run_snapshot_path if run_snapshot_path.is_file() else None,
    )


def validate_exact_proof_source_surface(
    source_run_root: Path,
    *,
    oracle_root: Path | None = None,
) -> ExactProofSourceSurface:
    """Validate the staged source surface for full-artifact exact proof."""
    run_root = source_run_root.resolve()
    checkpoints_dir = run_root / CHECKPOINTS_DIR
    validated_params_path = run_root / VALIDATED_PARAMS_PATH
    energy_checkpoint_path = checkpoints_dir / "checkpoint_energy.pkl"

    missing: list[Path] = []
    if not checkpoints_dir.is_dir():
        missing.append(checkpoints_dir)
    if not validated_params_path.is_file():
        missing.append(validated_params_path)
    if not energy_checkpoint_path.is_file():
        missing.append(energy_checkpoint_path)
    if missing:
        joined = ", ".join(str(path) for path in missing)
        raise ValueError(f"source run root is missing required exact-proof artifacts: {joined}")

    validated_params = load_json_dict(validated_params_path)
    if validated_params is None:
        raise ValueError(f"expected JSON object in params file: {validated_params_path}")
    if not bool(validated_params.get("comparison_exact_network")):
        raise ValueError(
            f"source run root does not enable comparison_exact_network in {validated_params_path}"
        )

    energy_payload = _expect_mapping(
        safe_load(energy_checkpoint_path),
        str(energy_checkpoint_path),
    )
    energy_origin = energy_payload.get("energy_origin", energy_payload.get("energy_source"))
    if not is_exact_compatible_energy_origin(energy_origin):
        accepted = exact_compatible_energy_origins_text()
        raise ValueError(
            "source energy provenance must be exact-compatible "
            f"({accepted}), found: {energy_origin!r}"
        )

    oracle_surface = _resolve_oracle_surface(run_root, oracle_root=oracle_root)
    return ExactProofSourceSurface(
        run_root=run_root,
        checkpoints_dir=checkpoints_dir,
        validated_params_path=validated_params_path,
        oracle_surface=oracle_surface,
        matlab_batch_dir=oracle_surface.matlab_batch_dir,
        matlab_vector_paths=oracle_surface.matlab_vector_paths,
    )


def validate_exact_preflight_surface(
    source_run_root: Path,
    dest_run_root: Path,
    *,
    oracle_root: Path | None = None,
) -> ExactPreflightSurface:
    """Validate the source exact surface and load the image shape needed for memory preflight."""
    source_surface = validate_exact_proof_source_surface(source_run_root, oracle_root=oracle_root)
    dest_root = dest_run_root.expanduser().resolve()
    if dest_root.exists() and not dest_root.is_dir():
        raise ValueError(f"destination run root must be a directory path: {dest_root}")

    energy_payload = _load_exact_energy_payload(source_surface)
    energy = np.asarray(energy_payload.get("energy"))
    if energy.ndim != 3:
        raise ValueError(
            f"expected 3D energy volume in {source_surface.checkpoints_dir / 'checkpoint_energy.pkl'}"
        )
    return ExactPreflightSurface(
        source_surface=source_surface,
        dest_run_root=dest_root,
        image_shape=cast("tuple[int, int, int]", tuple(int(value) for value in energy.shape)),
    )


def build_exact_preflight_report(
    source_run_root: Path,
    dest_run_root: Path,
    *,
    oracle_root: Path | None,
    memory_safety_fraction: float,
    force: bool,
) -> dict[str, Any]:
    """Build the fail-fast preflight report for a native-first exact run."""
    try:
        surface = validate_exact_preflight_surface(
            source_run_root,
            dest_run_root,
            oracle_root=oracle_root,
        )
    except ValueError as exc:
        return {
            "passed": False,
            "source_run_root": str(source_run_root.expanduser().resolve()),
            "dest_run_root": str(dest_run_root.expanduser().resolve()),
            "report_scope": "native-first exact preflight only",
            "error": str(exc),
        }

    params_payload = _load_json_object(surface.source_surface.validated_params_path)
    params_audit = build_exact_params_audit(params_payload)
    memory_estimate = estimate_exact_route_memory(surface.image_shape)
    available_memory_bytes = int(psutil.virtual_memory().available)
    safety_fraction = float(max(min(memory_safety_fraction, 1.0), 0.05))
    allowed_memory_bytes = int(available_memory_bytes * safety_fraction)
    collisions = find_parity_process_collisions(surface.dest_run_root)
    passed = (
        bool(params_audit.get("passed"))
        and (memory_estimate["estimated_required_bytes"] <= allowed_memory_bytes)
        and (force or not collisions)
    )
    return {
        "passed": passed,
        "source_run_root": str(surface.source_surface.run_root),
        "dest_run_root": str(surface.dest_run_root),
        "oracle_root": str(surface.source_surface.oracle_surface.oracle_root),
        "oracle_id": surface.source_surface.oracle_surface.oracle_id,
        "dataset_hash": (
            _dataset_hash_from_run_root(surface.source_surface.run_root)
            or surface.source_surface.oracle_surface.dataset_hash
        ),
        "report_scope": "native-first exact preflight only",
        "image_shape": list(surface.image_shape),
        "memory_estimate": memory_estimate,
        "available_memory_bytes": available_memory_bytes,
        "allowed_memory_bytes": allowed_memory_bytes,
        "memory_safety_fraction": safety_fraction,
        "collision_count": len(collisions),
        "collisions": collisions,
        "force": bool(force),
        "params_audit": params_audit,
    }


def render_exact_preflight_report(report_payload: dict[str, Any]) -> str:
    """Render a compact exact-route preflight report."""
    lines = [
        "Exact preflight report",
        f"Status: {'PASS' if report_payload.get('passed') else 'FAIL'}",
        f"Source run root: {report_payload.get('source_run_root')}",
        f"Destination run root: {report_payload.get('dest_run_root')}",
        f"Oracle root: {report_payload.get('oracle_root')}",
    ]
    error_text = report_payload.get("error")
    if isinstance(error_text, str) and error_text.strip():
        lines.append(f"Error: {error_text}")
        return "\n".join(lines)

    lines.extend(
        [
            f"Image shape: {report_payload.get('image_shape')}",
        ]
    )
    memory_estimate = _mapping_item(report_payload, "memory_estimate")
    lines.append(
        "Memory: "
        f"estimated={memory_estimate.get('estimated_required_bytes')} "
        f"allowed={report_payload.get('allowed_memory_bytes')} "
        f"available={report_payload.get('available_memory_bytes')}"
    )
    params_audit = report_payload.get("params_audit")
    if isinstance(params_audit, dict):
        lines.append(
            f"Params audit: {'PASS' if params_audit.get('passed') else 'FAIL'} "
            f"(shared={params_audit.get('shared_method_param_count', 0)})"
        )
        disallowed_keys = params_audit.get("disallowed_python_only_keys", [])
        if disallowed_keys:
            joined = ", ".join(str(value) for value in disallowed_keys)
            lines.append(f"Disallowed Python-only parity keys: {joined}")
        required_mismatches = params_audit.get("required_exact_mismatches", [])
        if required_mismatches:
            mismatch_summaries = [
                f"{item['key']} expected={item['expected']!r} found={item['found']!r}"
                for item in required_mismatches
                if isinstance(item, dict)
            ]
            lines.append("Required exact mismatches: " + "; ".join(mismatch_summaries))
        unclassified_keys = params_audit.get("unclassified_keys", [])
        if unclassified_keys:
            joined = ", ".join(str(value) for value in unclassified_keys)
            lines.append(f"Unclassified params: {joined}")
    lines.append(f"Collision count: {report_payload.get('collision_count', 0)}")
    collisions = report_payload.get("collisions", [])
    if collisions:
        lines.append(f"Collisions: {collisions}")
    return "\n".join(lines)


def estimate_exact_route_memory(image_shape: tuple[int, int, int]) -> dict[str, Any]:
    """Estimate the peak exact-route memory footprint from the planned full-volume arrays."""
    voxel_count = int(np.prod(np.asarray(image_shape, dtype=np.int64)))
    planned_arrays: list[dict[str, int | str]] = []
    subtotal_bytes = 0
    for name, bytes_per_voxel in EXACT_ROUTE_ARRAY_BYTES_PER_VOXEL:
        estimated_bytes = int(voxel_count * bytes_per_voxel)
        planned_arrays.append(
            {
                "name": name,
                "bytes_per_voxel": bytes_per_voxel,
                "estimated_bytes": estimated_bytes,
            }
        )
        subtotal_bytes += estimated_bytes
    overhead_bytes = round(subtotal_bytes * 0.25)
    return {
        "voxel_count": voxel_count,
        "planned_arrays": planned_arrays,
        "subtotal_bytes": subtotal_bytes,
        "overhead_bytes": overhead_bytes,
        "estimated_required_bytes": subtotal_bytes + overhead_bytes,
    }


def find_parity_process_collisions(dest_run_root: Path) -> list[dict[str, Any]]:
    """Return live parity processes already targeting the same destination run root."""
    current_process = psutil.Process()
    ignored_pids = {int(current_process.pid)}
    with suppress(psutil.Error, OSError):
        ignored_pids.update(int(parent.pid) for parent in current_process.parents())
    collisions: list[dict[str, Any]] = []
    normalized_dest = str(dest_run_root.resolve()).lower()
    owner_commands = {"rerun-python", "capture-candidates", "replay-edges", "fail-fast"}
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        info = process.info
        pid = int(info.get("pid", -1))
        if pid in ignored_pids:
            continue
        cmdline = info.get("cmdline") or []
        if not isinstance(cmdline, list):
            continue
        joined = " ".join(str(part) for part in cmdline).lower()
        if "parity_experiment.py" not in joined:
            continue
        if normalized_dest not in joined:
            continue
        if not any(f" {command} " in f" {joined} " for command in owner_commands):
            continue
        collisions.append(
            {
                "pid": pid,
                "name": str(info.get("name", "")),
                "cmdline": [str(part) for part in cmdline],
            }
        )
    return collisions


def resolve_input_file(
    source_surface: SourceRunSurface,
    input_arg: str | None,
    *,
    repo_root: Path = REPO_ROOT,
) -> Path:
    """Resolve the input file either from the CLI or the source run snapshot provenance."""
    if input_arg:
        candidate = Path(input_arg).expanduser()
    else:
        if source_surface.run_snapshot_path is None:
            raise ValueError(
                "source run root does not contain 99_Metadata/run_snapshot.json and no --input was provided"
            )
        snapshot_payload = load_json_dict(source_surface.run_snapshot_path)
        provenance = (
            snapshot_payload.get("provenance", {}) if isinstance(snapshot_payload, dict) else {}
        )
        raw_input = provenance.get("input_file") if isinstance(provenance, dict) else None
        if not isinstance(raw_input, str) or not raw_input.strip():
            raise ValueError(
                "source run snapshot does not record provenance.input_file; pass --input explicitly"
            )
        candidate = Path(raw_input)
        if not candidate.is_absolute():
            candidate = repo_root / candidate

    resolved = candidate.resolve()
    if not resolved.is_file():
        raise ValueError(f"input file not found: {resolved}")
    return resolved


def _load_json_object(json_path: Path) -> dict[str, Any]:
    payload = load_json_dict(json_path)
    if payload is None:
        raise ValueError(f"expected JSON object in params file: {json_path}")
    return cast("dict[str, Any]", payload)


def _normalize_param_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _normalize_param_value(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_normalize_param_value(item) for item in value]
    if isinstance(value, list):
        return [_normalize_param_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalize_param_value(item) for key, item in value.items()}
    return value


def build_exact_params_audit(params: dict[str, Any]) -> dict[str, Any]:
    """Build a fairness audit for exact-route parameter payloads."""
    param_keys = {str(key) for key in params}
    shared_method_keys = sorted(param_keys & EXACT_SHARED_METHOD_PARAMETER_KEYS)
    orchestration_keys = sorted(param_keys & EXACT_ALLOWED_ORCHESTRATION_PARAMETER_KEYS)
    disallowed_python_only_keys = sorted(key for key in param_keys if key.startswith("parity_"))

    required_exact_mismatches: list[dict[str, Any]] = []
    for key, expected_value in EXACT_REQUIRED_PARAMETER_VALUES.items():
        actual_value = _normalize_param_value(params.get(key))
        normalized_expected = _normalize_param_value(expected_value)
        if actual_value != normalized_expected:
            required_exact_mismatches.append(
                {
                    "key": key,
                    "expected": normalized_expected,
                    "found": actual_value,
                }
            )

    known_keys = (
        EXACT_SHARED_METHOD_PARAMETER_KEYS
        | EXACT_ALLOWED_ORCHESTRATION_PARAMETER_KEYS
        | set(EXACT_REQUIRED_PARAMETER_VALUES)
    )
    unclassified_keys = sorted(
        key
        for key in param_keys
        if key not in known_keys and key not in disallowed_python_only_keys
    )

    return {
        "passed": not disallowed_python_only_keys and not required_exact_mismatches,
        "shared_method_param_count": len(shared_method_keys),
        "shared_method_params": {
            key: _normalize_param_value(params[key]) for key in shared_method_keys if key in params
        },
        "allowed_orchestration_params": {
            key: _normalize_param_value(params[key]) for key in orchestration_keys if key in params
        },
        "required_exact_values": {
            key: _normalize_param_value(value)
            for key, value in EXACT_REQUIRED_PARAMETER_VALUES.items()
        },
        "required_exact_mismatches": required_exact_mismatches,
        "disallowed_python_only_keys": disallowed_python_only_keys,
        "unclassified_keys": unclassified_keys,
    }


def _raise_if_unfair_exact_params(params: dict[str, Any], *, params_path: Path) -> None:
    audit = build_exact_params_audit(params)
    if bool(audit.get("passed")):
        return

    disallowed_keys = ", ".join(
        str(value) for value in audit.get("disallowed_python_only_keys", [])
    )
    mismatch_items = [
        f"{item['key']} expected={item['expected']!r} found={item['found']!r}"
        for item in audit.get("required_exact_mismatches", [])
        if isinstance(item, dict)
    ]
    mismatch_text = "; ".join(mismatch_items)
    problems: list[str] = []
    if disallowed_keys:
        problems.append(f"disallowed Python-only parity keys: {disallowed_keys}")
    if mismatch_text:
        problems.append(f"required exact setting mismatches: {mismatch_text}")
    details = "; ".join(problems) if problems else "unknown fairness mismatch"
    raise ValueError(
        f"exact-route params are not a fair MATLAB comparison in {params_path}: {details}"
    )


def load_params_file(
    source_surface: SourceRunSurface,
    params_file: str | None,
) -> dict[str, Any]:
    """Load the JSON parameter payload for the rerun."""
    params_path = (
        Path(params_file).expanduser().resolve()
        if params_file is not None
        else source_surface.validated_params_path
    )
    payload = _load_json_object(params_path)
    if bool(payload.get("comparison_exact_network")):
        _raise_if_unfair_exact_params(payload, params_path=params_path)
    return payload


def load_exact_params_file(source_surface: ExactProofSourceSurface) -> dict[str, Any]:
    """Load the exact-route validated params payload."""
    payload = _load_json_object(source_surface.validated_params_path)
    _raise_if_unfair_exact_params(payload, params_path=source_surface.validated_params_path)
    return payload


def ensure_dest_run_layout(dest_run_root: Path) -> None:
    """Ensure the minimal staged directories used by the developer parity helpers."""
    experiment_root = _resolve_experiment_root(dest_run_root)
    if experiment_root is not None:
        _ensure_experiment_root_layout(experiment_root)
    (dest_run_root / EXPERIMENT_REFS_DIR).mkdir(parents=True, exist_ok=True)
    (dest_run_root / EXPERIMENT_PARAMS_DIR).mkdir(parents=True, exist_ok=True)
    (dest_run_root / CHECKPOINTS_DIR).mkdir(parents=True, exist_ok=True)
    (dest_run_root / ANALYSIS_DIR).mkdir(parents=True, exist_ok=True)
    (dest_run_root / NORMALIZED_DIR).mkdir(parents=True, exist_ok=True)
    (dest_run_root / HASHES_DIR).mkdir(parents=True, exist_ok=True)
    (dest_run_root / METADATA_DIR).mkdir(parents=True, exist_ok=True)


def write_capture_candidates_snapshot(
    dest_run_root: Path,
    *,
    detail: str,
    iteration_count: int = 0,
    candidate_count: int = 0,
    elapsed_seconds: float = 0.0,
    telemetry_point_count: int = 0,
    progress_artifacts: dict[str, Any] | None = None,
) -> None:
    """Persist lightweight run progress for long candidate-capture runs."""
    progress_artifacts = dict(progress_artifacts or {})
    artifact_payload: dict[str, str] = {
        "edge_candidate_iterations": str(int(iteration_count)),
        "edge_candidate_count": str(int(candidate_count)),
        "candidate_progress_elapsed_seconds": f"{float(elapsed_seconds):.3f}",
        "candidate_progress_point_count": str(int(telemetry_point_count)),
    }
    for key, value in progress_artifacts.items():
        artifact_payload[str(key)] = str(value)

    _write_json_with_hash(
        dest_run_root / RUN_SNAPSHOT_PATH,
        {
            "run_id": "capture-candidates",
            "status": "running",
            "target_stage": "edges",
            "current_stage": "edges",
            "current_detail": detail,
            "overall_progress": 0.0,
            "last_event": detail,
            "stages": {
                "edges": {
                    "name": "edges",
                    "status": "running",
                    "progress": 0.0,
                    "detail": detail,
                    "substage": "generate_candidates",
                    "units_total": 1,
                    "units_completed": 0,
                    "artifacts": {},
                    "peak_memory_bytes": int(getattr(psutil.Process().memory_info(), "rss", 0)),
                }
            },
            "optional_tasks": {},
            "artifacts": artifact_payload,
            "errors": [],
            "provenance": {},
        },
    )


def _build_candidate_progress_record(
    *,
    phase: str,
    detail: str,
    started_at_monotonic: float,
    iteration_count: int,
    candidate_count: int,
    previous_record: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build one candidate-capture telemetry point."""
    now_monotonic = time.monotonic()
    elapsed_seconds = max(0.0, now_monotonic - started_at_monotonic)
    memory_rss_bytes = int(getattr(psutil.Process().memory_info(), "rss", 0))

    candidate_delta = int(candidate_count)
    iteration_delta = int(iteration_count)
    elapsed_delta = elapsed_seconds
    if previous_record is not None:
        candidate_delta -= int(previous_record.get("candidate_count", 0))
        iteration_delta -= int(previous_record.get("iteration_count", 0))
        elapsed_delta -= float(previous_record.get("elapsed_seconds", 0.0))

    candidates_per_second = (
        float(candidate_delta) / float(elapsed_delta) if elapsed_delta > 1e-9 else 0.0
    )
    candidates_per_iteration = (
        float(candidate_delta) / float(iteration_delta) if iteration_delta > 0 else 0.0
    )

    return {
        "timestamp": _now_iso(),
        "phase": str(phase),
        "detail": str(detail),
        "iteration_count": int(iteration_count),
        "candidate_count": int(candidate_count),
        "elapsed_seconds": float(round(elapsed_seconds, 6)),
        "memory_rss_bytes": memory_rss_bytes,
        "candidate_delta": int(candidate_delta),
        "iteration_delta": int(iteration_delta),
        "elapsed_delta_seconds": float(round(max(0.0, elapsed_delta), 6)),
        "candidates_per_second": float(round(candidates_per_second, 6)),
        "candidates_per_iteration": float(round(candidates_per_iteration, 6)),
    }


def _write_candidate_progress_plot(
    plot_path: Path,
    telemetry_points: list[dict[str, Any]],
) -> None:
    """Render a lightweight candidate-capture progress plot."""
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    iterations = [int(point.get("iteration_count", 0)) for point in telemetry_points]
    candidates = [int(point.get("candidate_count", 0)) for point in telemetry_points]
    elapsed = [float(point.get("elapsed_seconds", 0.0)) for point in telemetry_points]
    memory_mb = [
        float(point.get("memory_rss_bytes", 0)) / (1024.0 * 1024.0) for point in telemetry_points
    ]
    candidate_rates = [float(point.get("candidates_per_second", 0.0)) for point in telemetry_points]

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    figure.suptitle("Candidate Capture Progress", fontsize=14)

    axes[0, 0].plot(iterations, candidates, marker="o", linewidth=1.5, markersize=3)
    axes[0, 0].set_title("Candidates vs Iteration")
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Candidates")
    axes[0, 0].grid(True, alpha=0.25)

    axes[0, 1].plot(elapsed, candidates, marker="o", linewidth=1.5, markersize=3)
    axes[0, 1].set_title("Candidates vs Elapsed Time")
    axes[0, 1].set_xlabel("Elapsed seconds")
    axes[0, 1].set_ylabel("Candidates")
    axes[0, 1].grid(True, alpha=0.25)

    axes[1, 0].plot(elapsed, candidate_rates, marker="o", linewidth=1.5, markersize=3)
    axes[1, 0].set_title("Candidate Rate")
    axes[1, 0].set_xlabel("Elapsed seconds")
    axes[1, 0].set_ylabel("Candidates / second")
    axes[1, 0].grid(True, alpha=0.25)

    axes[1, 1].plot(elapsed, memory_mb, marker="o", linewidth=1.5, markersize=3)
    axes[1, 1].set_title("Resident Memory")
    axes[1, 1].set_xlabel("Elapsed seconds")
    axes[1, 1].set_ylabel("Memory (MB)")
    axes[1, 1].grid(True, alpha=0.25)

    if telemetry_points:
        last_point = telemetry_points[-1]
        phase = str(last_point.get("phase", "unknown"))
        detail = str(last_point.get("detail", ""))
        figure.text(
            0.5,
            0.01,
            f"Latest phase: {phase} | {detail}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)


def persist_candidate_progress_artifacts(
    dest_run_root: Path,
    telemetry_points: list[dict[str, Any]],
) -> dict[str, Any]:
    """Persist candidate-capture telemetry and the latest plot artifact."""
    jsonl_path = dest_run_root / CANDIDATE_PROGRESS_JSONL_PATH
    jsonl_payload = "".join(f"{stable_json_dumps(dict(point))}\n" for point in telemetry_points)
    atomic_write_text(jsonl_path, jsonl_payload)

    artifact_summary: dict[str, Any] = {
        "candidate_progress_jsonl": str(jsonl_path),
        "candidate_progress_point_count": len(telemetry_points),
    }

    plot_path = dest_run_root / CANDIDATE_PROGRESS_PLOT_PATH
    try:
        _write_candidate_progress_plot(plot_path, telemetry_points)
    except Exception as exc:  # pragma: no cover - defensive artifact path
        artifact_summary["candidate_progress_plot_error"] = str(exc)
    else:
        artifact_summary["candidate_progress_plot"] = str(plot_path)

    return artifact_summary


def _read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.is_file():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if isinstance(payload, dict):
            records.append(cast("dict[str, Any]", payload))
    return records


def _coerce_table_cell(value: Any) -> Any:
    normalized = _normalize_param_value(value)
    if isinstance(normalized, float) and normalized != normalized:
        return None
    if isinstance(normalized, (list, dict)):
        return stable_json_dumps(normalized)
    return normalized


def _persist_table_records(
    tables_root: Path,
    *,
    table_name: str,
    records: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not records:
        return None

    normalized_records = [
        cast("dict[str, Any]", _normalize_param_value(dict(record))) for record in records
    ]
    frame = pd.json_normalize(normalized_records, sep=".")
    frame = frame.reindex(sorted(frame.columns), axis=1)
    frame = frame.apply(lambda column: column.map(_coerce_table_cell))

    jsonl_path = tables_root / f"{table_name}.jsonl"
    csv_path = tables_root / f"{table_name}.csv"

    row_payloads = [
        {str(key): _coerce_table_cell(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]
    jsonl_text = "".join(f"{stable_json_dumps(row)}\n" for row in row_payloads)
    atomic_write_text(jsonl_path, jsonl_text)
    _write_hash_sidecar(jsonl_path)

    csv_text = frame.to_csv(index=False)
    atomic_write_text(csv_path, csv_text)
    _write_hash_sidecar(csv_path)

    return {
        "name": table_name,
        "row_count": len(row_payloads),
        "column_count": len(frame.columns),
        "columns": [str(column) for column in frame.columns.tolist()],
        "jsonl_path": str(jsonl_path),
        "csv_path": str(csv_path),
    }


def _build_run_snapshot_tables(
    run_root: Path,
    snapshot_payload: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    run_id = _string_or_none(snapshot_payload.get("run_id")) or _entity_id_from_path(run_root)
    root_row = {
        **{
            str(key): value
            for key, value in snapshot_payload.items()
            if key not in {"stages", "optional_tasks", "artifacts", "errors"}
        },
        "run_root": str(run_root),
        "stage_count": len(snapshot_payload.get("stages", {})),
        "optional_task_count": len(snapshot_payload.get("optional_tasks", {})),
        "artifact_count": len(snapshot_payload.get("artifacts", {})),
        "error_count": len(snapshot_payload.get("errors", [])),
    }

    stage_rows: list[dict[str, Any]] = []
    stage_artifact_rows: list[dict[str, Any]] = []
    for stage_key, stage_payload in sorted(
        cast("dict[str, dict[str, Any]]", snapshot_payload.get("stages", {})).items()
    ):
        row = {
            "run_root": str(run_root),
            "run_id": run_id,
            "stage_key": str(stage_key),
            **{str(key): value for key, value in dict(stage_payload).items() if key != "artifacts"},
            "artifact_count": len(stage_payload.get("artifacts", {})),
        }
        stage_rows.append(row)
        for artifact_key, artifact_value in sorted(
            cast("dict[str, Any]", stage_payload.get("artifacts", {})).items()
        ):
            stage_artifact_rows.append(
                {
                    "run_root": str(run_root),
                    "run_id": run_id,
                    "stage_key": str(stage_key),
                    "artifact_key": str(artifact_key),
                    "artifact_value": artifact_value,
                }
            )

    optional_task_rows: list[dict[str, Any]] = []
    optional_task_artifact_rows: list[dict[str, Any]] = []
    for task_key, task_payload in sorted(
        cast("dict[str, dict[str, Any]]", snapshot_payload.get("optional_tasks", {})).items()
    ):
        row = {
            "run_root": str(run_root),
            "run_id": run_id,
            "task_key": str(task_key),
            **{str(key): value for key, value in dict(task_payload).items() if key != "artifacts"},
            "artifact_count": len(task_payload.get("artifacts", {})),
        }
        optional_task_rows.append(row)
        for artifact_key, artifact_value in sorted(
            cast("dict[str, Any]", task_payload.get("artifacts", {})).items()
        ):
            optional_task_artifact_rows.append(
                {
                    "run_root": str(run_root),
                    "run_id": run_id,
                    "task_key": str(task_key),
                    "artifact_key": str(artifact_key),
                    "artifact_value": artifact_value,
                }
            )

    artifact_rows = [
        {
            "run_root": str(run_root),
            "run_id": run_id,
            "artifact_key": str(artifact_key),
            "artifact_value": artifact_value,
        }
        for artifact_key, artifact_value in sorted(
            cast("dict[str, Any]", snapshot_payload.get("artifacts", {})).items()
        )
    ]
    error_rows = [
        {
            "run_root": str(run_root),
            "run_id": run_id,
            "error_index": error_index,
            "error_payload": error_payload,
        }
        for error_index, error_payload in enumerate(
            cast("list[Any]", snapshot_payload.get("errors", []))
        )
    ]

    return {
        "run_snapshot": [root_row],
        "run_snapshot_stages": stage_rows,
        "run_snapshot_stage_artifacts": stage_artifact_rows,
        "run_snapshot_optional_tasks": optional_task_rows,
        "run_snapshot_optional_task_artifacts": optional_task_artifact_rows,
        "run_snapshot_artifacts": artifact_rows,
        "run_snapshot_errors": error_rows,
    }


def _build_run_manifest_tables(
    run_root: Path,
    manifest_payload: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    run_id = _string_or_none(manifest_payload.get("run_id")) or _entity_id_from_path(run_root)
    root_row = {
        **{str(key): value for key, value in manifest_payload.items() if key != "stage_metrics"},
        "run_root": str(run_root),
        "stage_metric_count": len(manifest_payload.get("stage_metrics", {})),
    }
    stage_metric_rows = [
        {
            "run_root": str(run_root),
            "run_id": run_id,
            "stage_key": str(stage_key),
            **dict(stage_metric_payload),
        }
        for stage_key, stage_metric_payload in sorted(
            cast("dict[str, dict[str, Any]]", manifest_payload.get("stage_metrics", {})).items()
        )
    ]
    return {
        "run_manifest": [root_row],
        "run_manifest_stage_metrics": stage_metric_rows,
    }


def _build_candidate_audit_tables(
    run_root: Path,
    audit_payload: dict[str, Any],
    *,
    artifact_path: Path,
) -> dict[str, list[dict[str, Any]]]:
    per_origin_summary = cast("list[dict[str, Any]]", audit_payload.get("per_origin_summary", []))
    metric_map_keys = (
        "frontier_per_origin_candidate_counts",
        "frontier_per_origin_terminal_accepts",
        "frontier_per_origin_terminal_hits",
        "frontier_per_origin_terminal_rejections",
        "geodesic_per_origin_candidate_counts",
        "watershed_per_origin_candidate_counts",
    )
    summary_payload = {
        str(key): value
        for key, value in audit_payload.items()
        if key not in {"per_origin_summary", *metric_map_keys}
    }
    summary_row = {
        **summary_payload,
        "run_root": str(run_root),
        "artifact_path": str(artifact_path),
        "per_origin_row_count": len(per_origin_summary),
    }
    metric_rows: list[dict[str, Any]] = []
    for metric_name in metric_map_keys:
        metric_payload = cast("dict[str, Any]", audit_payload.get(metric_name, {}))
        for origin_key, metric_value in sorted(
            metric_payload.items(), key=lambda item: str(item[0])
        ):
            with suppress(ValueError, TypeError):
                origin_key = int(origin_key)
            metric_rows.append(
                {
                    "run_root": str(run_root),
                    "artifact_path": str(artifact_path),
                    "metric_name": metric_name,
                    "origin_index": origin_key,
                    "metric_value": metric_value,
                }
            )

    per_origin_rows = [
        {
            "run_root": str(run_root),
            "artifact_path": str(artifact_path),
            **dict(row),
        }
        for row in per_origin_summary
    ]
    return {
        "candidate_audit_summary": [summary_row],
        "candidate_audit_per_origin": per_origin_rows,
        "candidate_audit_origin_metrics": metric_rows,
    }


def _build_candidate_coverage_tables(
    run_root: Path,
    coverage_payload: dict[str, Any],
    *,
    artifact_path: Path,
) -> dict[str, list[dict[str, Any]]]:
    top_missing = cast("list[dict[str, Any]]", coverage_payload.get("top_missing_vertices", []))
    top_extra = cast("list[dict[str, Any]]", coverage_payload.get("top_extra_vertices", []))
    missing_pairs = cast("list[list[int]]", coverage_payload.get("missing_pairs", []))
    extra_pairs = cast("list[list[int]]", coverage_payload.get("extra_pairs", []))
    summary_row = {
        **{
            str(key): value
            for key, value in coverage_payload.items()
            if key not in {"top_missing_vertices", "top_extra_vertices", "missing_pairs", "extra_pairs"}
        },
        "run_root": str(run_root),
        "artifact_path": str(artifact_path),
        "top_missing_vertex_count": len(top_missing),
        "top_extra_vertex_count": len(top_extra),
        "missing_pair_row_count": len(missing_pairs),
        "extra_pair_row_count": len(extra_pairs),
    }
    missing_rows = [
        {
            "run_root": str(run_root),
            "artifact_path": str(artifact_path),
            "rank": rank,
            **dict(row),
        }
        for rank, row in enumerate(top_missing, start=1)
    ]
    extra_rows = [
        {
            "run_root": str(run_root),
            "artifact_path": str(artifact_path),
            "rank": rank,
            **dict(row),
        }
        for rank, row in enumerate(top_extra, start=1)
    ]
    missing_pair_rows = [
        {
            "run_root": str(run_root),
            "artifact_path": str(artifact_path),
            "rank": rank,
            "start_vertex": int(pair[0]),
            "end_vertex": int(pair[1]),
        }
        for rank, pair in enumerate(missing_pairs, start=1)
        if len(pair) >= 2
    ]
    extra_pair_rows = [
        {
            "run_root": str(run_root),
            "artifact_path": str(artifact_path),
            "rank": rank,
            "start_vertex": int(pair[0]),
            "end_vertex": int(pair[1]),
        }
        for rank, pair in enumerate(extra_pairs, start=1)
        if len(pair) >= 2
    ]
    return {
        "candidate_coverage_summary": [summary_row],
        "candidate_coverage_top_missing_vertices": missing_rows,
        "candidate_coverage_top_extra_vertices": extra_rows,
        "candidate_coverage_missing_pairs": missing_pair_rows,
        "candidate_coverage_extra_pairs": extra_pair_rows,
    }


def _load_candidate_diagnostics_payload(run_root: Path) -> dict[str, Any] | None:
    candidate_audit_payload = load_json_dict(run_root / EDGE_CANDIDATE_AUDIT_PATH)
    if candidate_audit_payload is not None:
        return candidate_audit_payload

    checkpoint_path = run_root / EDGE_CANDIDATE_CHECKPOINT_PATH
    if not checkpoint_path.is_file():
        return None
    checkpoint_payload = _expect_mapping(safe_load(checkpoint_path), str(checkpoint_path))
    diagnostics = checkpoint_payload.get("diagnostics")
    if not isinstance(diagnostics, dict):
        return None
    return cast("dict[str, Any]", diagnostics)


def _build_gap_hotspot_rows(
    vertex_rows: list[dict[str, Any]],
    per_origin_summary: list[dict[str, Any]],
    *,
    gap_kind: str,
    limit: int,
) -> list[dict[str, Any]]:
    origin_frame = pd.DataFrame(per_origin_summary)
    if origin_frame.empty:
        origin_frame = pd.DataFrame(columns=["origin_index"])
    if "origin_index" in origin_frame.columns:
        origin_frame["origin_index"] = pd.to_numeric(
            origin_frame["origin_index"],
            errors="coerce",
        ).astype("Int64")

    vertex_frame = pd.DataFrame(vertex_rows[: max(0, int(limit))])
    if vertex_frame.empty:
        return []
    vertex_frame = vertex_frame.rename(columns={"vertex": "origin_index", "count": "gap_count"})
    vertex_frame["origin_index"] = pd.to_numeric(
        vertex_frame["origin_index"],
        errors="coerce",
    ).astype("Int64")

    merged = vertex_frame.merge(origin_frame, how="left", on="origin_index", suffixes=("", "_audit"))
    merged["gap_kind"] = gap_kind

    columns = [
        "gap_kind",
        "rank",
        "origin_index",
        "gap_count",
        "candidate_connection_count",
        "fallback_candidate_count",
        "frontier_candidate_count",
        "geodesic_candidate_count",
        "watershed_candidate_count",
        "candidate_endpoint_pair_count",
    ]
    available_columns = [column for column in columns if column in merged.columns]
    merged = merged.reindex(columns=available_columns)
    merged = merged.replace({pd.NA: None})
    return cast("list[dict[str, Any]]", merged.to_dict(orient="records"))


def build_gap_diagnosis_report(
    run_root: Path,
    *,
    limit: int = 10,
    coverage_payload: dict[str, Any] | None = None,
    audit_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a joined gap diagnosis from candidate coverage and origin-level diagnostics."""
    normalized_run_root = run_root.expanduser().resolve()
    coverage = coverage_payload or load_json_dict(normalized_run_root / CANDIDATE_COVERAGE_JSON_PATH)
    if coverage is None:
        raise ValueError(
            f"missing candidate coverage report under {normalized_run_root / CANDIDATE_COVERAGE_JSON_PATH}"
        )

    audit = audit_payload if audit_payload is not None else _load_candidate_diagnostics_payload(normalized_run_root)
    per_origin_summary = cast("list[dict[str, Any]]", (audit or {}).get("per_origin_summary", []))
    diagnostic_counters = cast("dict[str, Any]", (audit or {}).get("diagnostic_counters", {}))
    pair_source_breakdown = cast("dict[str, Any]", (audit or {}).get("pair_source_breakdown", {}))

    warnings: list[str] = []
    if audit is None:
        warnings.append("No origin-level candidate audit was found; hotspot joins are limited.")
    if audit is not None and audit.get("use_frontier_tracer") is False:
        warnings.append("Exact frontier tracer was disabled in this recording.")
    if int(diagnostic_counters.get("watershed_total_pairs", 0)) == 0:
        warnings.append("No watershed candidate pairs were recorded.")
    if (
        int(pair_source_breakdown.get("fallback_only_pair_count", 0)) > 0
        and int(pair_source_breakdown.get("frontier_only_pair_count", 0)) == 0
        and int(pair_source_breakdown.get("watershed_only_pair_count", 0)) == 0
        and int(pair_source_breakdown.get("geodesic_only_pair_count", 0)) == 0
    ):
        warnings.append("All recorded candidate endpoint pairs were fallback-only.")

    top_missing_vertices = cast("list[dict[str, Any]]", coverage.get("top_missing_vertices", []))
    top_extra_vertices = cast("list[dict[str, Any]]", coverage.get("top_extra_vertices", []))
    hotspot_limit = max(1, int(limit))
    top_missing_hotspots = _build_gap_hotspot_rows(
        top_missing_vertices,
        per_origin_summary,
        gap_kind="missing",
        limit=hotspot_limit,
    )
    top_extra_hotspots = _build_gap_hotspot_rows(
        top_extra_vertices,
        per_origin_summary,
        gap_kind="extra",
        limit=hotspot_limit,
    )

    return {
        "run_root": str(normalized_run_root),
        "created_at": _now_iso(),
        "report_scope": "candidate gap diagnosis",
        "passed": bool(coverage.get("passed")),
        "missing_pair_count": int(coverage.get("missing_pair_count", 0)),
        "extra_pair_count": int(coverage.get("extra_pair_count", 0)),
        "matched_pair_count": int(coverage.get("matched_pair_count", 0)),
        "matlab_pair_count": int(coverage.get("matlab_pair_count", 0)),
        "python_pair_count": int(coverage.get("python_pair_count", 0)),
        "missing_pair_samples": cast("list[list[int]]", coverage.get("missing_pair_samples", [])),
        "extra_pair_samples": cast("list[list[int]]", coverage.get("extra_pair_samples", [])),
        "missing_pairs": cast("list[list[int]]", coverage.get("missing_pairs", [])),
        "extra_pairs": cast("list[list[int]]", coverage.get("extra_pairs", [])),
        "warnings": warnings,
        "audit_present": audit is not None,
        "origin_summary_count": len(per_origin_summary),
        "diagnostic_counters": diagnostic_counters,
        "pair_source_breakdown": pair_source_breakdown,
        "top_missing_vertices": top_missing_vertices[:hotspot_limit],
        "top_extra_vertices": top_extra_vertices[:hotspot_limit],
        "top_missing_hotspots": top_missing_hotspots,
        "top_extra_hotspots": top_extra_hotspots,
    }


def render_gap_diagnosis_report(report_payload: dict[str, Any]) -> str:
    """Render a concise gap diagnosis focused on actionable hotspots."""
    lines = [
        "Gap diagnosis report",
        f"Status: {'PASS' if report_payload.get('passed') else 'FAIL'}",
        (
            "Counts: "
            f"MATLAB={report_payload.get('matlab_pair_count', 0)} "
            f"Python={report_payload.get('python_pair_count', 0)} "
            f"matched={report_payload.get('matched_pair_count', 0)} "
            f"missing={report_payload.get('missing_pair_count', 0)} "
            f"extra={report_payload.get('extra_pair_count', 0)}"
        ),
    ]

    warnings = cast("list[str]", report_payload.get("warnings", []))
    if warnings:
        lines.extend(["", "Warnings"])
        lines.extend(f"- {warning}" for warning in warnings)

    missing_hotspots = cast("list[dict[str, Any]]", report_payload.get("top_missing_hotspots", []))
    if missing_hotspots:
        lines.extend(["", "Top Missing Hotspots"])
        for row in missing_hotspots:
            lines.append(
                "- "
                f"origin={row.get('origin_index')} gap_count={row.get('gap_count')} "
                f"candidates={row.get('candidate_connection_count')} "
                f"fallback={row.get('fallback_candidate_count')} "
                f"frontier={row.get('frontier_candidate_count')} "
                f"watershed={row.get('watershed_candidate_count')}"
            )

    extra_hotspots = cast("list[dict[str, Any]]", report_payload.get("top_extra_hotspots", []))
    if extra_hotspots:
        lines.extend(["", "Top Extra Hotspots"])
        for row in extra_hotspots:
            lines.append(
                "- "
                f"origin={row.get('origin_index')} gap_count={row.get('gap_count')} "
                f"candidates={row.get('candidate_connection_count')} "
                f"fallback={row.get('fallback_candidate_count')} "
                f"frontier={row.get('frontier_candidate_count')} "
                f"watershed={row.get('watershed_candidate_count')}"
            )

    missing_samples = cast("list[list[int]]", report_payload.get("missing_pair_samples", []))
    if missing_samples:
        lines.append("")
        lines.append(f"Missing pair samples: {missing_samples[:5]}")
    extra_samples = cast("list[list[int]]", report_payload.get("extra_pair_samples", []))
    if extra_samples:
        lines.append(f"Extra pair samples: {extra_samples[:5]}")
    return "\n".join(lines)


def persist_gap_diagnosis_report(
    run_root: Path,
    *,
    limit: int = 10,
    coverage_payload: dict[str, Any] | None = None,
    audit_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist a joined gap diagnosis report under ``03_Analysis``."""
    normalized_run_root = run_root.expanduser().resolve()
    ensure_dest_run_layout(normalized_run_root)
    report_payload = build_gap_diagnosis_report(
        normalized_run_root,
        limit=limit,
        coverage_payload=coverage_payload,
        audit_payload=audit_payload,
    )
    persist_text_and_json_report(
        normalized_run_root / GAP_DIAGNOSIS_JSON_PATH,
        normalized_run_root / GAP_DIAGNOSIS_TEXT_PATH,
        report_payload,
        renderer=render_gap_diagnosis_report,
    )
    return report_payload


def _build_gap_diagnosis_tables(
    run_root: Path,
    report_payload: dict[str, Any],
    *,
    artifact_path: Path,
) -> dict[str, list[dict[str, Any]]]:
    warning_rows = [
        {
            "run_root": str(run_root),
            "artifact_path": str(artifact_path),
            "rank": rank,
            "warning": warning,
        }
        for rank, warning in enumerate(cast("list[str]", report_payload.get("warnings", [])), start=1)
    ]
    missing_rows = [
        {
            "run_root": str(run_root),
            "artifact_path": str(artifact_path),
            **dict(row),
        }
        for row in cast("list[dict[str, Any]]", report_payload.get("top_missing_hotspots", []))
    ]
    extra_rows = [
        {
            "run_root": str(run_root),
            "artifact_path": str(artifact_path),
            **dict(row),
        }
        for row in cast("list[dict[str, Any]]", report_payload.get("top_extra_hotspots", []))
    ]
    summary_row = {
        **{
            str(key): value
            for key, value in report_payload.items()
            if key
            not in {
                "warnings",
                "top_missing_hotspots",
                "top_extra_hotspots",
                "top_missing_vertices",
                "top_extra_vertices",
                "missing_pairs",
                "extra_pairs",
                "missing_pair_samples",
                "extra_pair_samples",
                "diagnostic_counters",
                "pair_source_breakdown",
            }
        },
        "run_root": str(run_root),
        "artifact_path": str(artifact_path),
        "warning_count": len(warning_rows),
        "missing_hotspot_count": len(missing_rows),
        "extra_hotspot_count": len(extra_rows),
    }
    return {
        "gap_diagnosis_summary": [summary_row],
        "gap_diagnosis_warnings": warning_rows,
        "gap_diagnosis_missing_hotspots": missing_rows,
        "gap_diagnosis_extra_hotspots": extra_rows,
    }


def persist_recording_tables(run_root: Path) -> dict[str, Any]:
    """Flatten workflow recordings into CSV/JSONL tables under ``03_Analysis/tables``."""
    normalized_run_root = run_root.expanduser().resolve()
    ensure_dest_run_layout(normalized_run_root)
    tables_root = normalized_run_root / ANALYSIS_TABLES_DIR
    tables_root.mkdir(parents=True, exist_ok=True)

    table_entries: list[dict[str, Any]] = []
    source_artifacts: list[str] = []

    snapshot_payload = load_json_dict(normalized_run_root / RUN_SNAPSHOT_PATH)
    if snapshot_payload is not None:
        source_artifacts.append(str(normalized_run_root / RUN_SNAPSHOT_PATH))
        for table_name, records in _build_run_snapshot_tables(
            normalized_run_root,
            snapshot_payload,
        ).items():
            table_entry = _persist_table_records(
                tables_root,
                table_name=table_name,
                records=records,
            )
            if table_entry is not None:
                table_entries.append(table_entry)

    run_manifest_payload = load_json_dict(normalized_run_root / RUN_MANIFEST_PATH)
    if run_manifest_payload is not None:
        source_artifacts.append(str(normalized_run_root / RUN_MANIFEST_PATH))
        for table_name, records in _build_run_manifest_tables(
            normalized_run_root,
            run_manifest_payload,
        ).items():
            table_entry = _persist_table_records(
                tables_root,
                table_name=table_name,
                records=records,
            )
            if table_entry is not None:
                table_entries.append(table_entry)

    candidate_progress_records = _read_jsonl_records(
        normalized_run_root / CANDIDATE_PROGRESS_JSONL_PATH
    )
    if candidate_progress_records:
        source_artifacts.append(str(normalized_run_root / CANDIDATE_PROGRESS_JSONL_PATH))
        table_entry = _persist_table_records(
            tables_root,
            table_name="candidate_progress",
            records=[
                {
                    "run_root": str(normalized_run_root),
                    **dict(record),
                }
                for record in candidate_progress_records
            ],
        )
        if table_entry is not None:
            table_entries.append(table_entry)

    candidate_audit_path = normalized_run_root / EDGE_CANDIDATE_AUDIT_PATH
    candidate_audit_payload = load_json_dict(candidate_audit_path)
    if candidate_audit_payload is not None:
        source_artifacts.append(str(candidate_audit_path))
        for table_name, records in _build_candidate_audit_tables(
            normalized_run_root,
            candidate_audit_payload,
            artifact_path=candidate_audit_path,
        ).items():
            table_entry = _persist_table_records(
                tables_root,
                table_name=table_name,
                records=records,
            )
            if table_entry is not None:
                table_entries.append(table_entry)

    candidate_coverage_path = normalized_run_root / CANDIDATE_COVERAGE_JSON_PATH
    candidate_coverage_payload = load_json_dict(candidate_coverage_path)
    if candidate_coverage_payload is not None:
        source_artifacts.append(str(candidate_coverage_path))
        for table_name, records in _build_candidate_coverage_tables(
            normalized_run_root,
            candidate_coverage_payload,
            artifact_path=candidate_coverage_path,
        ).items():
            table_entry = _persist_table_records(
                tables_root,
                table_name=table_name,
                records=records,
            )
            if table_entry is not None:
                table_entries.append(table_entry)

    gap_diagnosis_path = normalized_run_root / GAP_DIAGNOSIS_JSON_PATH
    gap_diagnosis_payload = load_json_dict(gap_diagnosis_path)
    if gap_diagnosis_payload is not None:
        source_artifacts.append(str(gap_diagnosis_path))
        for table_name, records in _build_gap_diagnosis_tables(
            normalized_run_root,
            gap_diagnosis_payload,
            artifact_path=gap_diagnosis_path,
        ).items():
            table_entry = _persist_table_records(
                tables_root,
                table_name=table_name,
                records=records,
            )
            if table_entry is not None:
                table_entries.append(table_entry)

    if not table_entries:
        raise ValueError(f"no supported recording artifacts found under {normalized_run_root}")

    index_payload = {
        "run_root": str(normalized_run_root),
        "created_at": _now_iso(),
        "tables_root": str(tables_root),
        "table_count": len(table_entries),
        "row_count_total": sum(int(entry.get("row_count", 0)) for entry in table_entries),
        "source_artifacts": source_artifacts,
        "tables": table_entries,
    }
    _write_json_with_hash(normalized_run_root / RECORDING_TABLES_INDEX_PATH, index_payload)
    return index_payload


def _load_exact_energy_payload(source_surface: ExactProofSourceSurface) -> dict[str, Any]:
    checkpoint_path = source_surface.checkpoints_dir / "checkpoint_energy.pkl"
    return _expect_mapping(safe_load(checkpoint_path), str(checkpoint_path))


def _load_exact_vertices_payload(source_surface: ExactProofSourceSurface) -> dict[str, Any]:
    checkpoint_path = source_surface.checkpoints_dir / "checkpoint_vertices.pkl"
    payload = dict(_expect_mapping(safe_load(checkpoint_path), str(checkpoint_path)))
    normalized_vertices = load_normalized_matlab_vectors(
        source_surface.matlab_batch_dir,
        ("vertices",),
    )["vertices"]
    payload["positions"] = np.asarray(normalized_vertices["positions"], dtype=np.float32)
    payload["scales"] = np.asarray(normalized_vertices["scales"], dtype=np.int16)
    payload["energies"] = np.asarray(normalized_vertices["energies"], dtype=np.float32)
    payload["count"] = len(payload["positions"])
    return payload


def _lumen_radius_pixels_axes(
    energy_payload: dict[str, Any],
    params: dict[str, Any],
) -> np.ndarray:
    if "lumen_radius_pixels_axes" in energy_payload:
        return cast(
            "np.ndarray",
            np.asarray(energy_payload["lumen_radius_pixels_axes"], dtype=np.float32),
        )
    lumen_radius_microns = np.asarray(
        energy_payload["lumen_radius_microns"], dtype=np.float32
    ).reshape(-1, 1)
    microns_per_voxel = np.asarray(
        params.get("microns_per_voxel", [1.0, 1.0, 1.0]),
        dtype=np.float32,
    ).reshape(1, 3)
    return cast(
        "np.ndarray",
        lumen_radius_microns / np.maximum(microns_per_voxel, 1e-6),
    )


def extract_matlab_counts(report_payload: dict[str, Any]) -> RunCounts:
    """Extract preserved MATLAB count truth from a comparison report."""
    matlab = _mapping_item(report_payload, "matlab")
    vertices = _coerce_int(
        matlab.get("vertices_count", _mapping_item(report_payload, "vertices").get("matlab_count")),
        label="matlab vertices count",
    )
    edges = _coerce_int(
        matlab.get("edges_count", _mapping_item(report_payload, "edges").get("matlab_count")),
        label="matlab edges count",
    )
    strands = _coerce_int(
        matlab.get(
            "strand_count", _mapping_item(report_payload, "network").get("matlab_strand_count")
        ),
        label="matlab strand count",
    )
    return RunCounts(vertices=vertices, edges=edges, strands=strands)


def extract_source_python_counts(report_payload: dict[str, Any]) -> RunCounts:
    """Extract the preserved source-run Python counts from a comparison report."""
    python_counts = _mapping_item(report_payload, "python")
    vertices = _coerce_int(
        python_counts.get(
            "vertices_count",
            _mapping_item(report_payload, "vertices").get("python_count"),
        ),
        label="source python vertices count",
    )
    edges = _coerce_int(
        python_counts.get(
            "edges_count", _mapping_item(report_payload, "edges").get("python_count")
        ),
        label="source python edges count",
    )
    strands = _coerce_int(
        python_counts.get(
            "network_strands_count",
            _mapping_item(report_payload, "network").get("python_strand_count"),
        ),
        label="source python strand count",
    )
    return RunCounts(vertices=vertices, edges=edges, strands=strands)


def read_python_counts_from_run(run_root: Path) -> RunCounts:
    """Read Python stage counts from the structured checkpoint surface."""
    checkpoints_dir = run_root / CHECKPOINTS_DIR
    vertices_payload = _expect_mapping(
        safe_load(checkpoints_dir / "checkpoint_vertices.pkl"),
        "checkpoint_vertices.pkl",
    )
    edges_payload = _expect_mapping(
        safe_load(checkpoints_dir / "checkpoint_edges.pkl"),
        "checkpoint_edges.pkl",
    )
    network_payload = _expect_mapping(
        safe_load(checkpoints_dir / "checkpoint_network.pkl"),
        "checkpoint_network.pkl",
    )
    return RunCounts(
        vertices=_payload_count(
            vertices_payload, preferred_keys=("positions", "count"), label="vertices"
        ),
        edges=_payload_count(
            edges_payload, preferred_keys=("connections", "traces", "count"), label="edges"
        ),
        strands=_payload_count(
            network_payload, preferred_keys=("strands", "count"), label="network"
        ),
    )


def build_experiment_summary(
    *,
    source_run_root: Path,
    dest_run_root: Path,
    input_file: Path,
    rerun_from: str,
    matlab_counts: RunCounts,
    source_python_counts: RunCounts,
    new_python_counts: RunCounts,
) -> dict[str, Any]:
    """Build a JSON-serializable experiment summary."""
    return {
        "source_run_root": str(source_run_root),
        "dest_run_root": str(dest_run_root),
        "input_file": str(input_file),
        "rerun_from": rerun_from,
        "matlab_counts": asdict(matlab_counts),
        "source_python_counts": asdict(source_python_counts),
        "new_python_counts": asdict(new_python_counts),
        "diff_vs_matlab": _diff_counts(new_python_counts, matlab_counts),
        "diff_vs_source_python": _diff_counts(new_python_counts, source_python_counts),
    }


def render_experiment_summary(summary_payload: dict[str, Any]) -> str:
    """Render a human-readable experiment summary."""
    matlab_counts = _mapping_item(summary_payload, "matlab_counts")
    source_python_counts = _mapping_item(summary_payload, "source_python_counts")
    new_python_counts = _mapping_item(summary_payload, "new_python_counts")
    diff_vs_matlab = _mapping_item(summary_payload, "diff_vs_matlab")
    diff_vs_source_python = _mapping_item(summary_payload, "diff_vs_source_python")
    return "\n".join(
        [
            "Parity experiment summary",
            f"Source run root: {summary_payload['source_run_root']}",
            f"Destination run root: {summary_payload['dest_run_root']}",
            f"Input file: {summary_payload['input_file']}",
            f"Rerun from: {summary_payload['rerun_from']}",
            "",
            "Counts",
            f"MATLAB: vertices={matlab_counts['vertices']} edges={matlab_counts['edges']} strands={matlab_counts['strands']}",
            (
                "Source Python: "
                f"vertices={source_python_counts['vertices']} "
                f"edges={source_python_counts['edges']} "
                f"strands={source_python_counts['strands']}"
            ),
            (
                "New Python: "
                f"vertices={new_python_counts['vertices']} "
                f"edges={new_python_counts['edges']} "
                f"strands={new_python_counts['strands']}"
            ),
            "",
            "Delta vs MATLAB",
            _format_delta_line(diff_vs_matlab),
            "Delta vs source Python",
            _format_delta_line(diff_vs_source_python),
        ]
    )


def copy_source_surface(source_surface: SourceRunSurface, dest_run_root: Path) -> None:
    """Copy the reusable source checkpoints and reference metadata into a fresh destination root."""
    destination = dest_run_root.resolve()
    if destination.exists():
        raise ValueError(f"destination run root already exists: {destination}")

    ensure_dest_run_layout(destination)
    checkpoints_dir = destination / CHECKPOINTS_DIR
    for artifact in source_surface.checkpoints_dir.iterdir():
        target = checkpoints_dir / artifact.name
        if artifact.is_dir():
            copytree(artifact, target)
        else:
            copy2(artifact, target)

    (destination / "01_Input").mkdir(parents=True, exist_ok=True)
    refs_dir = destination / EXPERIMENT_REFS_DIR
    copy2(source_surface.validated_params_path, refs_dir / "source_validated_params.json")
    copy2(source_surface.comparison_report_path, refs_dir / "source_comparison_report.json")
    if source_surface.run_snapshot_path is not None:
        copy2(source_surface.run_snapshot_path, refs_dir / "source_run_snapshot.json")
    source_run_manifest = source_surface.run_root / RUN_MANIFEST_PATH
    if source_run_manifest.is_file():
        copy2(source_run_manifest, refs_dir / "source_run_manifest.json")


def _copy_exact_bootstrap_refs(
    dest_run_root: Path,
    *,
    dataset_surface: DatasetSurface,
    oracle_surface: OracleSurface,
) -> None:
    refs_dir = dest_run_root / EXPERIMENT_REFS_DIR
    copy2(dataset_surface.manifest_path, refs_dir / "dataset_manifest.json")
    if oracle_surface.manifest_path is not None and oracle_surface.manifest_path.is_file():
        copy2(oracle_surface.manifest_path, refs_dir / "oracle_manifest.json")
    selection_manifest_path = oracle_surface.matlab_batch_dir / "selection_manifest.json"
    if selection_manifest_path.is_file():
        copy2(selection_manifest_path, refs_dir / "oracle_selection_manifest.json")


def _oracle_energy_size_of_image(oracle_surface: OracleSurface) -> tuple[int, int, int] | None:
    energy_metadata_path = oracle_surface.matlab_vector_paths.get("energy")
    if energy_metadata_path is None or not energy_metadata_path.is_file():
        return None
    energy_metadata = cast(
        "dict[str, Any]",
        loadmat(energy_metadata_path, squeeze_me=True, struct_as_record=False),
    )
    raw_size = energy_metadata.get("size_of_image")
    if raw_size is None:
        return None
    size_vector = np.asarray(raw_size, dtype=np.int64).reshape(-1)
    if size_vector.size != 3:
        return None
    return cast("tuple[int, int, int]", tuple(int(value) for value in size_vector))


def _reorient_exact_input_volume(
    image: np.ndarray,
    oracle_surface: OracleSurface,
) -> tuple[np.ndarray, tuple[int, int, int] | None, tuple[int, int, int] | None]:
    expected_shape = _oracle_energy_size_of_image(oracle_surface)
    if expected_shape is None:
        return image, None, None

    actual_shape = cast("tuple[int, int, int]", tuple(int(value) for value in image.shape))
    if actual_shape == expected_shape:
        return image, expected_shape, None

    for permutation in permutations(range(image.ndim)):
        if tuple(actual_shape[index] for index in permutation) == expected_shape:
            return (
                np.asarray(np.transpose(image, permutation)),
                expected_shape,
                cast("tuple[int, int, int]", tuple(int(index) for index in permutation)),
            )

    raise ValueError(
        "loaded TIFF shape does not match oracle size_of_image under any axis permutation: "
        f"loaded={actual_shape}, oracle={expected_shape}"
    )


def _update_run_snapshot_provenance(
    dest_run_root: Path, provenance_updates: dict[str, Any]
) -> None:
    snapshot_payload = load_json_dict(dest_run_root / RUN_SNAPSHOT_PATH) or {}
    provenance = cast("dict[str, Any]", dict(snapshot_payload.get("provenance", {})))
    provenance.update(provenance_updates)
    snapshot_payload["provenance"] = provenance
    if "input_fingerprint" not in snapshot_payload and "dataset_hash" in provenance_updates:
        snapshot_payload["input_fingerprint"] = provenance_updates["dataset_hash"]
    _write_json_with_hash(dest_run_root / RUN_SNAPSHOT_PATH, snapshot_payload)


def _exact_bootstrap_provenance_updates(
    *,
    dataset_surface: DatasetSurface,
    oracle_surface: OracleSurface,
    selected_settings_paths: dict[str, str],
    oracle_size_of_image: tuple[int, int, int] | None,
    input_axis_permutation: tuple[int, int, int] | None,
    stop_after: str,
) -> dict[str, Any]:
    return {
        "source": "parity_experiment.init-exact-run",
        "input_file": str(dataset_surface.input_file),
        "dataset_root": str(dataset_surface.dataset_root),
        "dataset_hash": dataset_surface.dataset_hash,
        "oracle_root": str(oracle_surface.oracle_root),
        "oracle_id": oracle_surface.oracle_id,
        "oracle_manifest": (
            str(oracle_surface.manifest_path) if oracle_surface.manifest_path is not None else None
        ),
        "matlab_source_version": oracle_surface.matlab_source_version,
        "selected_settings_paths": selected_settings_paths,
        "oracle_size_of_image": list(oracle_size_of_image)
        if oracle_size_of_image is not None
        else None,
        "input_axis_permutation": (
            list(input_axis_permutation) if input_axis_permutation is not None else None
        ),
        "stop_after": stop_after,
    }


def _finalize_init_exact_run(
    *,
    dest_run_root: Path,
    dataset_surface: DatasetSurface,
    oracle_surface: OracleSurface,
    params: dict[str, Any],
    selected_settings_paths: dict[str, str],
    oracle_size_of_image: tuple[int, int, int] | None,
    input_axis_permutation: tuple[int, int, int] | None,
    stop_after: str,
) -> None:
    _write_json_with_hash(dest_run_root / VALIDATED_PARAMS_PATH, params)
    _persist_param_storage(dest_run_root, params)
    _update_run_snapshot_provenance(
        dest_run_root,
        _exact_bootstrap_provenance_updates(
            dataset_surface=dataset_surface,
            oracle_surface=oracle_surface,
            selected_settings_paths=selected_settings_paths,
            oracle_size_of_image=oracle_size_of_image,
            input_axis_permutation=input_axis_permutation,
            stop_after=stop_after,
        ),
    )
    _write_run_manifest(
        dest_run_root,
        run_kind="parity_source_run",
        status="seeded",
        command="init-exact-run",
        dataset_hash=dataset_surface.dataset_hash,
        oracle_surface=oracle_surface,
        params_payload=params,
        extra={"seed_stop_after": stop_after},
    )


def _resolve_existing_init_exact_run(
    *,
    dest_run_root: Path,
    dataset_surface: DatasetSurface,
    oracle_surface: OracleSurface,
    stop_after: str,
) -> bool:
    if not dest_run_root.exists():
        return False
    if not dest_run_root.is_dir():
        raise ValueError(f"destination run root must be a directory: {dest_run_root}")

    provenance_path = dest_run_root / EXPERIMENT_PROVENANCE_PATH
    provenance = load_json_dict(provenance_path)
    if provenance is None:
        raise ValueError(f"destination run root already exists: {dest_run_root}")

    if _string_or_none(provenance.get("bootstrap_kind")) != "init-exact-run":
        raise ValueError(
            f"destination run root already exists but is not an init-exact-run bootstrap: {dest_run_root}"
        )
    if _string_or_none(provenance.get("dataset_hash")) != dataset_surface.dataset_hash:
        raise ValueError(
            "existing init-exact-run dataset hash does not match the requested dataset: "
            f"{_string_or_none(provenance.get('dataset_hash'))!r} != {dataset_surface.dataset_hash!r}"
        )
    existing_oracle_id = _string_or_none(provenance.get("oracle_id"))
    if existing_oracle_id != oracle_surface.oracle_id:
        raise ValueError(
            "existing init-exact-run oracle id does not match the requested oracle: "
            f"{existing_oracle_id!r} != {oracle_surface.oracle_id!r}"
        )
    if _string_or_none(provenance.get("stop_after")) != stop_after:
        raise ValueError(
            "existing init-exact-run stop-after target does not match the requested bootstrap: "
            f"{_string_or_none(provenance.get('stop_after'))!r} != {stop_after!r}"
        )

    snapshot_payload = load_json_dict(dest_run_root / RUN_SNAPSHOT_PATH) or {}
    snapshot_status = _string_or_none(snapshot_payload.get("status"))
    if snapshot_status == "running":
        raise ValueError(
            "destination run root already exists and the init-exact-run bootstrap is still active: "
            f"{dest_run_root}"
        )

    checkpoint_dir = dest_run_root / CHECKPOINTS_DIR
    required_checkpoints = [checkpoint_dir / "checkpoint_energy.pkl"]
    if stop_after == "vertices":
        required_checkpoints.append(checkpoint_dir / "checkpoint_vertices.pkl")
    missing_checkpoints = [path for path in required_checkpoints if not path.is_file()]
    if missing_checkpoints:
        missing_text = ", ".join(str(path) for path in missing_checkpoints)
        raise ValueError(
            "destination run root already exists but the bootstrap checkpoints are incomplete: "
            f"{missing_text}"
        )

    return True


def maybe_sync_exact_vertex_checkpoint(
    source_run_root: Path,
    dest_run_root: Path,
    *,
    oracle_root: Path | None = None,
) -> bool:
    """Refresh the destination vertex checkpoint from canonical MATLAB vectors on the exact route."""
    try:
        exact_surface = validate_exact_proof_source_surface(
            source_run_root,
            oracle_root=oracle_root,
        )
    except ValueError:
        return False

    checkpoint_path = dest_run_root / CHECKPOINTS_DIR / "checkpoint_vertices.pkl"
    if not checkpoint_path.is_file():
        raise ValueError(f"destination run root is missing vertex checkpoint: {checkpoint_path}")
    sync_exact_vertex_checkpoint_from_matlab(checkpoint_path, exact_surface.matlab_batch_dir)
    return True


def persist_experiment_summary(dest_run_root: Path, summary_payload: dict[str, Any]) -> None:
    """Persist the JSON and text summaries under 03_Analysis."""
    summary_text = render_experiment_summary(summary_payload)
    _write_json_with_hash(dest_run_root / SUMMARY_JSON_PATH, summary_payload)
    _write_text_with_hash(dest_run_root / SUMMARY_TEXT_PATH, summary_text)


def persist_exact_proof_report(
    json_path: Path,
    text_path: Path,
    report_payload: dict[str, Any],
) -> None:
    """Persist the JSON and text exact-proof reports."""
    _write_json_with_hash(json_path, report_payload)
    _write_text_with_hash(text_path, render_exact_proof_report(report_payload))


def persist_text_and_json_report(
    json_path: Path,
    text_path: Path,
    report_payload: dict[str, Any],
    *,
    renderer: Any,
) -> None:
    """Persist a JSON report and its paired text rendering."""
    _write_json_with_hash(json_path, report_payload)
    _write_text_with_hash(text_path, str(renderer(report_payload)))


def _build_candidate_exact_proof_report(
    matlab_edges_payload: dict[str, Any],
    candidate_payload: dict[str, Any],
) -> dict[str, Any]:
    """Build an exact-proof-compatible report from the candidate boundary only."""
    coverage_report = build_candidate_coverage_report(matlab_edges_payload, candidate_payload)
    first_failure: dict[str, Any] | None = None
    if not bool(coverage_report.get("passed")):
        first_failure = {
            "stage": "edges",
            "field_path": "edges.connections",
            "mismatch_type": "value mismatch",
            "matlab_preview": f"pair_count={int(coverage_report.get('matlab_pair_count', 0))}",
            "python_preview": (
                f"pair_count={int(coverage_report.get('python_pair_count', 0))} "
                f"matched={int(coverage_report.get('matched_pair_count', 0))} "
                f"missing={int(coverage_report.get('missing_pair_count', 0))} "
                f"extra={int(coverage_report.get('extra_pair_count', 0))}"
            ),
        }

    stage_summary: dict[str, Any] = {
        "passed": bool(coverage_report.get("passed")),
        "field_count": 1,
        "proof_surface": "candidate_connections_only",
    }
    if first_failure is not None:
        stage_summary["first_failure"] = first_failure

    return {
        "passed": bool(coverage_report.get("passed")),
        "stages": ["edges"],
        "stage_summaries": {"edges": stage_summary},
        "first_failing_stage": first_failure["stage"] if first_failure is not None else None,
        "first_failing_field_path": first_failure["field_path"]
        if first_failure is not None
        else None,
        "first_failure": first_failure,
        "candidate_surface": coverage_report,
        "report_scope": "candidate boundary fallback (edges.connections only)",
    }


def _run_preflight_exact(
    *,
    source_run_root: Path,
    dest_run_root: Path,
    oracle_root: Path | None = None,
    memory_safety_fraction: float,
    force: bool,
) -> tuple[dict[str, Any], Path, Path]:
    report_payload = build_exact_preflight_report(
        source_run_root,
        dest_run_root,
        oracle_root=oracle_root,
        memory_safety_fraction=memory_safety_fraction,
        force=force,
    )
    dest_root = dest_run_root.expanduser().resolve()
    ensure_dest_run_layout(dest_root)
    json_path = dest_root / PREFLIGHT_EXACT_JSON_PATH
    text_path = dest_root / PREFLIGHT_EXACT_TEXT_PATH
    persist_text_and_json_report(
        json_path,
        text_path,
        report_payload,
        renderer=render_exact_preflight_report,
    )
    try:
        source_surface = validate_exact_proof_source_surface(
            source_run_root,
            oracle_root=oracle_root,
        )
        params = load_exact_params_file(source_surface)
        _persist_param_storage(dest_root, params)
        _write_run_manifest(
            dest_root,
            run_kind="parity_run",
            status="passed" if bool(report_payload.get("passed")) else "failed",
            command="preflight-exact",
            dataset_hash=_dataset_hash_from_run_root(source_run_root),
            oracle_surface=source_surface.oracle_surface,
            params_payload=params,
            extra={"preflight_report": str(json_path)},
        )
    except ValueError:
        pass
    return report_payload, json_path, text_path


def _run_prove_luts(
    *,
    source_run_root: Path,
    dest_run_root: Path,
    oracle_root: Path | None = None,
) -> tuple[dict[str, Any], Path, Path]:
    source_surface = validate_exact_proof_source_surface(
        source_run_root,
        oracle_root=oracle_root,
    )
    params = load_exact_params_file(source_surface)
    energy_payload = _load_exact_energy_payload(source_surface)
    size_of_image = cast(
        "tuple[int, int, int]",
        tuple(int(value) for value in np.asarray(energy_payload["energy"]).shape),
    )
    microns_per_voxel = np.asarray(
        params.get("microns_per_voxel", [1.0, 1.0, 1.0]), dtype=np.float32
    ).reshape(-1)
    lumen_radius_microns = np.asarray(
        energy_payload["lumen_radius_microns"], dtype=np.float32
    ).reshape(-1)
    fixture_payload = load_builtin_lut_fixture()
    fixture_size_of_image = tuple(int(value) for value in fixture_payload.get("size_of_image", []))
    fixture_microns_per_voxel = np.asarray(
        fixture_payload.get("microns_per_voxel", []), dtype=np.float32
    ).reshape(-1)
    fixture_lumen_radius_microns = np.asarray(
        fixture_payload.get("lumen_radius_microns", []), dtype=np.float32
    ).reshape(-1)
    fixture_matches_source = (
        fixture_size_of_image == size_of_image
        and np.array_equal(fixture_microns_per_voxel, microns_per_voxel)
        and np.array_equal(fixture_lumen_radius_microns, lumen_radius_microns)
    )
    if fixture_matches_source:
        report_payload = compare_lut_fixture_payload(
            fixture_payload,
            size_of_image=size_of_image,
            microns_per_voxel=microns_per_voxel,
            lumen_radius_microns=lumen_radius_microns,
        )
    else:
        scales_payload = fixture_payload.get("scales", {})
        report_payload = {
            "passed": True,
            "skipped": True,
            "skip_reason": "builtin LUT fixture inputs do not match the source exact run",
            "report_scope": "exact LUT parity only",
            "size_of_image": list(size_of_image),
            "scale_count": len(scales_payload) if isinstance(scales_payload, dict) else 0,
            "stage_summaries": {},
            "fixture_inputs": {
                "size_of_image": list(fixture_size_of_image),
                "microns_per_voxel": fixture_microns_per_voxel.tolist(),
                "lumen_radius_microns": fixture_lumen_radius_microns.tolist(),
            },
            "source_inputs": {
                "size_of_image": list(size_of_image),
                "microns_per_voxel": microns_per_voxel.tolist(),
                "lumen_radius_microns": lumen_radius_microns.tolist(),
            },
        }
    report_payload.update(
        {
            "source_run_root": str(source_surface.run_root),
            "dest_run_root": str(dest_run_root.expanduser().resolve()),
        }
    )
    dest_root = dest_run_root.expanduser().resolve()
    ensure_dest_run_layout(dest_root)
    _persist_param_storage(dest_root, params)
    json_path = dest_root / LUT_PROOF_JSON_PATH
    text_path = dest_root / LUT_PROOF_TEXT_PATH
    persist_text_and_json_report(
        json_path,
        text_path,
        report_payload,
        renderer=render_lut_proof_report,
    )
    _write_run_manifest(
        dest_root,
        run_kind="parity_run",
        status="passed" if bool(report_payload.get("passed")) else "failed",
        command="prove-luts",
        dataset_hash=_dataset_hash_from_run_root(source_run_root),
        oracle_surface=source_surface.oracle_surface,
        params_payload=params,
        extra={"lut_report": str(json_path)},
    )
    return report_payload, json_path, text_path


def _run_capture_candidates(
    *,
    source_run_root: Path,
    dest_run_root: Path,
    oracle_root: Path | None = None,
    include_debug_maps: bool,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    source_surface = validate_exact_proof_source_surface(
        source_run_root,
        oracle_root=oracle_root,
    )
    params = load_exact_params_file(source_surface)
    energy_payload = _load_exact_energy_payload(source_surface)
    vertices_payload = _load_exact_vertices_payload(source_surface)

    dest_root = dest_run_root.expanduser().resolve()
    ensure_dest_run_layout(dest_root)
    _persist_param_storage(dest_root, params)

    energy = np.asarray(energy_payload["energy"], dtype=np.float32)
    scale_indices = energy_payload.get("scale_indices")
    vertex_positions = np.asarray(vertices_payload["positions"], dtype=np.float32)
    vertex_scales = np.asarray(vertices_payload["scales"], dtype=np.int16)
    lumen_radius_microns = np.asarray(energy_payload["lumen_radius_microns"], dtype=np.float32)
    microns_per_voxel = np.asarray(
        params.get("microns_per_voxel", [1.0, 1.0, 1.0]), dtype=np.float32
    )
    vertex_center_image = paint_vertex_center_image(vertex_positions, energy.shape)
    capture_started_at = time.monotonic()
    telemetry_points: list[dict[str, Any]] = []
    progress_artifacts: dict[str, Any] = {}

    def _record_progress(
        *,
        phase: str,
        detail: str,
        iteration_count: int = 0,
        candidate_count: int = 0,
    ) -> None:
        nonlocal progress_artifacts
        previous_record = telemetry_points[-1] if telemetry_points else None
        progress_record = _build_candidate_progress_record(
            phase=phase,
            detail=detail,
            started_at_monotonic=capture_started_at,
            iteration_count=iteration_count,
            candidate_count=candidate_count,
            previous_record=previous_record,
        )
        telemetry_points.append(progress_record)
        progress_artifacts = persist_candidate_progress_artifacts(dest_root, telemetry_points)
        write_capture_candidates_snapshot(
            dest_root,
            detail=detail,
            iteration_count=iteration_count,
            candidate_count=candidate_count,
            elapsed_seconds=float(progress_record["elapsed_seconds"]),
            telemetry_point_count=len(telemetry_points),
            progress_artifacts=progress_artifacts,
        )

    def _heartbeat(iteration_count: int, candidate_count: int) -> None:
        detail = (
            "Generating edge candidates through MATLAB-style frontier workflow "
            f"(iterations={iteration_count}, candidates={candidate_count})"
        )
        _record_progress(
            phase="heartbeat",
            detail=detail,
            iteration_count=iteration_count,
            candidate_count=candidate_count,
        )

    _record_progress(
        phase="started",
        detail="Generating edge candidates through MATLAB-style frontier workflow",
    )
    candidates = _generate_edge_candidates_matlab_frontier(
        energy,
        None if scale_indices is None else np.asarray(scale_indices, dtype=np.int16),
        vertex_positions,
        vertex_scales,
        lumen_radius_microns,
        microns_per_voxel,
        vertex_center_image,
        params,
        heartbeat=_heartbeat,
    )
    candidates = _finalize_matlab_parity_candidates(
        candidates,
        energy,
        None if scale_indices is None else np.asarray(scale_indices, dtype=np.int16),
        vertex_positions,
        float(energy_payload.get("energy_sign", params.get("energy_sign", -1.0))),
        params,
        microns_per_voxel,
    )
    _record_progress(
        phase="completed",
        detail=(
            "Completed edge candidate generation through MATLAB-style frontier workflow "
            f"(candidates={len(candidates.get('traces', []))})"
        ),
        iteration_count=int(telemetry_points[-1]["iteration_count"]) if telemetry_points else 0,
        candidate_count=len(candidates.get("traces", [])),
    )

    snapshot_payload = build_candidate_snapshot_payload(
        candidates,
        include_debug_maps=include_debug_maps,
    )
    _write_joblib_with_hash(dest_root / EDGE_CANDIDATE_CHECKPOINT_PATH, snapshot_payload)
    matlab_edges = load_normalized_matlab_vectors(source_surface.matlab_batch_dir, ("edges",))[
        "edges"
    ]
    _persist_normalized_payloads(
        dest_root,
        group_name="capture_candidates",
        payloads={
            "candidate_snapshot": snapshot_payload,
            "matlab_edges": matlab_edges,
        },
    )
    coverage_report = build_candidate_coverage_report(matlab_edges, snapshot_payload)
    coverage_report.update(
        {
            "source_run_root": str(source_surface.run_root),
            "dest_run_root": str(dest_root),
            "debug_maps_included": bool(include_debug_maps),
            "candidate_checkpoint_path": str(dest_root / EDGE_CANDIDATE_CHECKPOINT_PATH),
            "candidate_progress_point_count": len(telemetry_points),
            "candidate_progress_elapsed_seconds": (
                float(telemetry_points[-1]["elapsed_seconds"]) if telemetry_points else 0.0
            ),
        }
    )
    coverage_report.update(progress_artifacts)
    json_path = dest_root / CANDIDATE_COVERAGE_JSON_PATH
    text_path = dest_root / CANDIDATE_COVERAGE_TEXT_PATH
    persist_text_and_json_report(
        json_path,
        text_path,
        coverage_report,
        renderer=render_candidate_coverage_report,
    )
    _write_run_manifest(
        dest_root,
        run_kind="parity_run",
        status="passed" if bool(coverage_report.get("passed")) else "failed",
        command="capture-candidates",
        dataset_hash=_dataset_hash_from_run_root(source_run_root),
        oracle_surface=source_surface.oracle_surface,
        params_payload=params,
        extra={
            "candidate_checkpoint_path": str(dest_root / EDGE_CANDIDATE_CHECKPOINT_PATH),
            "candidate_report": str(json_path),
            "candidate_progress_jsonl": str(dest_root / CANDIDATE_PROGRESS_JSONL_PATH),
            "candidate_progress_plot": str(dest_root / CANDIDATE_PROGRESS_PLOT_PATH),
        },
    )
    persist_gap_diagnosis_report(
        dest_root,
        coverage_payload=coverage_report,
        audit_payload=cast("dict[str, Any]", snapshot_payload.get("diagnostics", {})),
    )
    persist_recording_tables(dest_root)
    return coverage_report, snapshot_payload, json_path, text_path


def _run_replay_edges(
    *,
    source_run_root: Path,
    dest_run_root: Path,
    oracle_root: Path | None = None,
) -> tuple[dict[str, Any], Path, Path]:
    source_surface = validate_exact_proof_source_surface(
        source_run_root,
        oracle_root=oracle_root,
    )
    params = load_exact_params_file(source_surface)
    energy_payload = _load_exact_energy_payload(source_surface)
    vertices_payload = _load_exact_vertices_payload(source_surface)

    dest_root = dest_run_root.expanduser().resolve()
    ensure_dest_run_layout(dest_root)
    candidate_checkpoint_path = dest_root / EDGE_CANDIDATE_CHECKPOINT_PATH
    if not candidate_checkpoint_path.is_file():
        raise ValueError(f"missing candidate checkpoint for replay: {candidate_checkpoint_path}")
    candidates = _expect_mapping(
        safe_load(candidate_checkpoint_path), str(candidate_checkpoint_path)
    )

    energy = np.asarray(energy_payload["energy"], dtype=np.float32)
    scale_indices = energy_payload.get("scale_indices")
    lumen_radius_microns = np.asarray(energy_payload["lumen_radius_microns"], dtype=np.float32)
    microns_per_voxel = np.asarray(
        params.get("microns_per_voxel", [1.0, 1.0, 1.0]), dtype=np.float32
    )
    lumen_radius_pixels_axes = _lumen_radius_pixels_axes(energy_payload, params)
    vertex_positions = np.asarray(vertices_payload["positions"], dtype=np.float32)
    vertex_scales = np.asarray(vertices_payload["scales"], dtype=np.int16)

    chosen = choose_edges_for_workflow(
        candidates,
        vertex_positions,
        vertex_scales,
        lumen_radius_microns,
        lumen_radius_pixels_axes,
        tuple(int(value) for value in energy.shape),
        params,
    )
    chosen = add_vertices_to_edges_matlab_style(
        chosen,
        vertices_payload,
        energy=energy,
        scale_indices=None if scale_indices is None else np.asarray(scale_indices, dtype=np.int16),
        microns_per_voxel=microns_per_voxel,
        lumen_radius_microns=lumen_radius_microns,
        lumen_radius_pixels_axes=lumen_radius_pixels_axes,
        size_of_image=tuple(int(value) for value in energy.shape),
        params=params,
    )
    chosen = finalize_edges_matlab_style(
        chosen,
        lumen_radius_microns=lumen_radius_microns,
        microns_per_voxel=microns_per_voxel,
        size_of_image=tuple(int(value) for value in energy.shape),
    )
    chosen["lumen_radius_microns"] = lumen_radius_microns.astype(np.float32, copy=True)
    _write_joblib_with_hash(dest_root / CHECKPOINTS_DIR / "checkpoint_edges.pkl", chosen)

    matlab_edges = load_normalized_matlab_vectors(source_surface.matlab_batch_dir, ("edges",))
    python_edges = {
        "edges": normalize_python_stage_payload("edges", chosen),
    }
    _persist_param_storage(dest_root, params)
    _persist_normalized_payloads(
        dest_root,
        group_name="replay_edges",
        payloads={
            "matlab_edges": matlab_edges,
            "python_edges": python_edges,
        },
    )
    report_payload = compare_exact_artifacts(matlab_edges, python_edges, ("edges",))
    report_payload.update(
        {
            "source_run_root": str(source_surface.run_root),
            "dest_run_root": str(dest_root),
            "report_scope": "edge replay proof only",
        }
    )
    json_path = dest_root / EDGE_REPLAY_PROOF_JSON_PATH
    text_path = dest_root / EDGE_REPLAY_PROOF_TEXT_PATH
    persist_exact_proof_report(json_path, text_path, report_payload)
    _write_run_manifest(
        dest_root,
        run_kind="parity_run",
        status="passed" if bool(report_payload.get("passed")) else "failed",
        command="replay-edges",
        dataset_hash=_dataset_hash_from_run_root(source_run_root),
        oracle_surface=source_surface.oracle_surface,
        params_payload=params,
        extra={"edge_replay_report": str(json_path)},
    )
    return report_payload, json_path, text_path


def _handle_rerun_python(args: argparse.Namespace) -> None:
    source_surface = validate_source_run_surface(Path(args.source_run_root))
    dest_run_root = Path(args.dest_run_root).expanduser().resolve()
    input_file = resolve_input_file(source_surface, args.input)
    params = load_params_file(source_surface, args.params_file)
    copy_source_surface(source_surface, dest_run_root)
    _persist_param_storage(dest_run_root, params)
    oracle_surface: OracleSurface | None = None
    if bool(params.get("comparison_exact_network")):
        try:
            oracle_surface = _resolve_oracle_surface(source_surface.run_root, oracle_root=None)
        except ValueError:
            oracle_surface = None
    exact_vertex_sync = maybe_sync_exact_vertex_checkpoint(
        source_surface.run_root,
        dest_run_root,
        oracle_root=oracle_surface.oracle_root if oracle_surface is not None else None,
    )
    dataset_hash = fingerprint_file(input_file)
    _materialize_dataset_record(
        _resolve_experiment_root(dest_run_root),
        dataset_hash=dataset_hash,
        dataset_file=input_file,
    )

    _write_json_with_hash(
        dest_run_root / METADATA_DIR / "experiment_provenance.json",
        {
            "source_run_root": str(source_surface.run_root),
            "source_comparison_report": str(source_surface.comparison_report_path),
            "source_validated_params": str(source_surface.validated_params_path),
            "source_run_snapshot": (
                str(source_surface.run_snapshot_path)
                if source_surface.run_snapshot_path is not None
                else None
            ),
            "input_file": str(input_file),
            "dataset_hash": dataset_hash,
            "rerun_from": args.rerun_from,
            "exact_vertex_checkpoint_sync": exact_vertex_sync,
            "oracle_root": str(oracle_surface.oracle_root) if oracle_surface is not None else None,
            "oracle_id": oracle_surface.oracle_id if oracle_surface is not None else None,
        },
    )

    image = load_tiff_volume(input_file)
    processor = SLAVVProcessor()
    processor.process_image(
        image,
        params,
        run_dir=str(dest_run_root),
        force_rerun_from=args.rerun_from,
    )

    report_payload = load_json_dict(source_surface.comparison_report_path)
    if report_payload is None:
        raise ValueError(
            f"expected JSON object in comparison report: {source_surface.comparison_report_path}"
        )
    summary_payload = build_experiment_summary(
        source_run_root=source_surface.run_root,
        dest_run_root=dest_run_root,
        input_file=input_file,
        rerun_from=args.rerun_from,
        matlab_counts=extract_matlab_counts(report_payload),
        source_python_counts=extract_source_python_counts(report_payload),
        new_python_counts=read_python_counts_from_run(dest_run_root),
    )
    persist_experiment_summary(dest_run_root, summary_payload)
    _write_run_manifest(
        dest_run_root,
        run_kind="parity_run",
        status="completed",
        command="rerun-python",
        dataset_hash=dataset_hash,
        oracle_surface=oracle_surface,
        params_payload=params,
        extra={"rerun_from": args.rerun_from},
    )
    persist_recording_tables(dest_run_root)
    print(render_experiment_summary(summary_payload))


def _handle_summarize(args: argparse.Namespace) -> None:
    run_root = Path(args.run_root).expanduser().resolve()
    summary_text_path = run_root / SUMMARY_TEXT_PATH
    if summary_text_path.is_file():
        print(summary_text_path.read_text(encoding="utf-8"))
        return

    summary_payload = load_json_dict(run_root / SUMMARY_JSON_PATH)
    if summary_payload is None:
        raise ValueError(
            f"no experiment summary found under {run_root / SUMMARY_TEXT_PATH} or {run_root / SUMMARY_JSON_PATH}"
        )
    print(render_experiment_summary(summary_payload))


def _handle_normalize_recordings(args: argparse.Namespace) -> None:
    run_root = Path(args.run_root).expanduser().resolve()
    index_payload = persist_recording_tables(run_root)
    print(
        "\n".join(
            [
                f"Normalized recording tables written under {run_root / ANALYSIS_TABLES_DIR}",
                f"Table count: {index_payload['table_count']}",
                f"Total rows: {index_payload['row_count_total']}",
            ]
        )
    )


def _handle_diagnose_gaps(args: argparse.Namespace) -> None:
    run_root = Path(args.run_root).expanduser().resolve()
    report_payload = persist_gap_diagnosis_report(run_root, limit=max(1, int(args.limit)))
    persist_recording_tables(run_root)
    print(render_gap_diagnosis_report(report_payload))


def _handle_prove_exact(args: argparse.Namespace) -> None:
    source_surface = validate_exact_proof_source_surface(
        Path(args.source_run_root),
        oracle_root=Path(args.oracle_root) if getattr(args, "oracle_root", None) else None,
    )
    params = load_exact_params_file(source_surface)
    dest_run_root = Path(args.dest_run_root).expanduser().resolve()
    ensure_dest_run_layout(dest_run_root)
    _persist_param_storage(dest_run_root, params)
    checkpoints_dir = dest_run_root / CHECKPOINTS_DIR
    if not checkpoints_dir.is_dir():
        raise ValueError(f"destination run root is missing python checkpoints: {checkpoints_dir}")

    selected_stages = _selected_exact_stages(args.stage)
    matlab_artifacts = load_normalized_matlab_vectors(
        source_surface.matlab_batch_dir,
        selected_stages,
    )
    edge_checkpoint_path = checkpoints_dir / "checkpoint_edges.pkl"
    candidate_checkpoint_path = dest_run_root / EDGE_CANDIDATE_CHECKPOINT_PATH

    if (
        selected_stages == ("edges",)
        and not edge_checkpoint_path.is_file()
        and candidate_checkpoint_path.is_file()
    ):
        candidate_payload = _expect_mapping(
            safe_load(candidate_checkpoint_path),
            str(candidate_checkpoint_path),
        )
        report_payload = _build_candidate_exact_proof_report(
            matlab_artifacts["edges"],
            candidate_payload,
        )
        report_payload.update(
            {
                "candidate_checkpoint_path": str(candidate_checkpoint_path),
                "edge_checkpoint_path": str(edge_checkpoint_path),
            }
        )
        _persist_normalized_payloads(
            dest_run_root,
            group_name="prove_exact",
            payloads={
                "matlab_edges": matlab_artifacts["edges"],
                "candidate_edges": candidate_payload,
            },
        )
    else:
        python_artifacts = load_normalized_python_checkpoints(checkpoints_dir, selected_stages)
        _persist_normalized_payloads(
            dest_run_root,
            group_name="prove_exact",
            payloads={
                "matlab_artifacts": matlab_artifacts,
                "python_artifacts": python_artifacts,
            },
        )
        report_payload = compare_exact_artifacts(
            matlab_artifacts, python_artifacts, selected_stages
        )
    report_payload.update(
        {
            "source_run_root": str(source_surface.run_root),
            "dest_run_root": str(dest_run_root),
            "matlab_batch_dir": str(source_surface.matlab_batch_dir),
            "report_scope": str(
                report_payload.get("report_scope", "native-first exact route only")
            ),
            "exact_route_gate": exact_route_gate_description(),
        }
    )

    report_json_path, report_text_path = _resolve_exact_report_paths(
        dest_run_root,
        args.report_path,
    )
    persist_exact_proof_report(report_json_path, report_text_path, report_payload)
    _write_run_manifest(
        dest_run_root,
        run_kind="parity_run",
        status="passed" if bool(report_payload.get("passed")) else "failed",
        command="prove-exact",
        dataset_hash=_dataset_hash_from_run_root(source_surface.run_root),
        oracle_surface=source_surface.oracle_surface,
        params_payload=params,
        extra={"exact_report": str(report_json_path), "stage": args.stage},
    )

    rendered = render_exact_proof_report(report_payload)
    print(rendered)
    if not bool(report_payload.get("passed")):
        raise SystemExit(1)


def _handle_preflight_exact(args: argparse.Namespace) -> None:
    report_payload, _json_path, _text_path = _run_preflight_exact(
        source_run_root=Path(args.source_run_root),
        dest_run_root=Path(args.dest_run_root),
        oracle_root=Path(args.oracle_root) if args.oracle_root else None,
        memory_safety_fraction=float(args.memory_safety_fraction),
        force=bool(args.force),
    )
    rendered = render_exact_preflight_report(report_payload)
    print(rendered)
    if not bool(report_payload.get("passed")):
        raise SystemExit(1)


def _handle_prove_luts(args: argparse.Namespace) -> None:
    report_payload, _json_path, _text_path = _run_prove_luts(
        source_run_root=Path(args.source_run_root),
        dest_run_root=Path(args.dest_run_root),
        oracle_root=Path(args.oracle_root) if args.oracle_root else None,
    )
    rendered = render_lut_proof_report(report_payload)
    print(rendered)
    if not bool(report_payload.get("passed")):
        raise SystemExit(1)


def _handle_capture_candidates(args: argparse.Namespace) -> None:
    report_payload, _snapshot_payload, _json_path, _text_path = _run_capture_candidates(
        source_run_root=Path(args.source_run_root),
        dest_run_root=Path(args.dest_run_root),
        oracle_root=Path(args.oracle_root) if args.oracle_root else None,
        include_debug_maps=bool(args.debug_maps),
    )
    rendered = render_candidate_coverage_report(report_payload)
    print(rendered)
    if not bool(report_payload.get("passed")):
        raise SystemExit(1)


def _handle_replay_edges(args: argparse.Namespace) -> None:
    report_payload, _json_path, _text_path = _run_replay_edges(
        source_run_root=Path(args.source_run_root),
        dest_run_root=Path(args.dest_run_root),
        oracle_root=Path(args.oracle_root) if args.oracle_root else None,
    )
    rendered = render_exact_proof_report(report_payload)
    print(rendered)
    if not bool(report_payload.get("passed")):
        raise SystemExit(1)


def _handle_fail_fast(args: argparse.Namespace) -> None:
    source_run_root = Path(args.source_run_root)
    dest_run_root = Path(args.dest_run_root)
    oracle_root = Path(args.oracle_root) if args.oracle_root else None
    preflight_report, _json_path, _text_path = _run_preflight_exact(
        source_run_root=source_run_root,
        dest_run_root=dest_run_root,
        oracle_root=oracle_root,
        memory_safety_fraction=float(args.memory_safety_fraction),
        force=bool(args.force),
    )
    print(render_exact_preflight_report(preflight_report))
    if not bool(preflight_report.get("passed")):
        raise SystemExit(1)

    lut_report, _json_path, _text_path = _run_prove_luts(
        source_run_root=source_run_root,
        dest_run_root=dest_run_root,
        oracle_root=oracle_root,
    )
    print(render_lut_proof_report(lut_report))
    if not bool(lut_report.get("passed")):
        raise SystemExit(1)

    candidate_report, _snapshot_payload, _json_path, _text_path = _run_capture_candidates(
        source_run_root=source_run_root,
        dest_run_root=dest_run_root,
        oracle_root=oracle_root,
        include_debug_maps=bool(args.debug_maps),
    )
    print(render_candidate_coverage_report(candidate_report))
    if not bool(candidate_report.get("passed")):
        raise SystemExit(1)

    replay_report, _json_path, _text_path = _run_replay_edges(
        source_run_root=source_run_root,
        dest_run_root=dest_run_root,
        oracle_root=oracle_root,
    )
    print(render_exact_proof_report(replay_report))
    if not bool(replay_report.get("passed")):
        raise SystemExit(1)

    exact_args = argparse.Namespace(
        source_run_root=str(source_run_root),
        oracle_root=str(oracle_root) if oracle_root is not None else None,
        dest_run_root=str(dest_run_root),
        stage="all",
        report_path=None,
    )
    _handle_prove_exact(exact_args)


def _handle_promote_oracle(args: argparse.Namespace) -> None:
    matlab_batch_dir = Path(args.matlab_batch_dir).expanduser().resolve()
    oracle_root = Path(args.oracle_root).expanduser().resolve()
    if not matlab_batch_dir.is_dir():
        raise ValueError(f"MATLAB batch directory not found: {matlab_batch_dir}")
    if oracle_root.exists():
        raise ValueError(f"oracle root already exists: {oracle_root}")

    experiment_root = _resolve_experiment_root(oracle_root) or oracle_root.parent.parent
    _ensure_experiment_root_layout(experiment_root)
    dataset_file = Path(args.dataset_file).expanduser().resolve() if args.dataset_file else None
    dataset_hash = _materialize_dataset_record(
        experiment_root,
        dataset_hash=_string_or_none(args.dataset_hash),
        dataset_file=dataset_file,
    )
    _materialize_oracle_root(
        matlab_batch_dir=matlab_batch_dir,
        oracle_root=oracle_root,
        dataset_hash=dataset_hash,
        oracle_id=_string_or_none(args.oracle_id),
        matlab_source_version=_string_or_none(args.matlab_source_version),
    )
    print(str(oracle_root / ORACLE_MANIFEST_PATH))


def _handle_promote_dataset(args: argparse.Namespace) -> None:
    dataset_file = Path(args.dataset_file).expanduser().resolve()
    if not dataset_file.is_file():
        raise ValueError(f"dataset file not found: {dataset_file}")

    experiment_root = Path(args.experiment_root).expanduser().resolve()
    _ensure_experiment_root_layout(experiment_root)
    dataset_hash = _materialize_dataset_record(
        experiment_root,
        dataset_hash=None,
        dataset_file=dataset_file,
    )
    if dataset_hash is None:
        raise ValueError(f"could not fingerprint dataset file: {dataset_file}")
    print(str(experiment_root / "datasets" / dataset_hash / DATASET_MANIFEST_PATH))


def _handle_init_exact_run(args: argparse.Namespace) -> None:
    dataset_surface = _load_dataset_surface(Path(args.dataset_root))
    oracle_surface = _load_oracle_surface(Path(args.oracle_root))
    if (
        oracle_surface.dataset_hash is not None
        and oracle_surface.dataset_hash != dataset_surface.dataset_hash
    ):
        raise ValueError(
            "dataset and oracle hashes do not match: "
            f"{dataset_surface.dataset_hash} != {oracle_surface.dataset_hash}"
        )

    dest_run_root = Path(args.dest_run_root).expanduser().resolve()
    params, selected_settings_paths, selected_settings_payloads = derive_exact_params_from_oracle(
        oracle_surface
    )
    params["energy_storage_format"] = str(args.energy_storage_format).strip()
    resume_finalization_only = _resolve_existing_init_exact_run(
        dest_run_root=dest_run_root,
        dataset_surface=dataset_surface,
        oracle_surface=oracle_surface,
        stop_after=args.stop_after,
    )

    if resume_finalization_only:
        provenance_payload = load_json_dict(dest_run_root / EXPERIMENT_PROVENANCE_PATH) or {}
        raw_oracle_size = provenance_payload.get("oracle_size_of_image")
        oracle_size_of_image = (
            cast("tuple[int, int, int]", tuple(int(value) for value in raw_oracle_size))
            if isinstance(raw_oracle_size, list) and len(raw_oracle_size) == 3
            else _oracle_energy_size_of_image(oracle_surface)
        )
        raw_input_axis_permutation = provenance_payload.get("input_axis_permutation")
        input_axis_permutation = (
            cast(
                "tuple[int, int, int]",
                tuple(int(value) for value in raw_input_axis_permutation),
            )
            if isinstance(raw_input_axis_permutation, list) and len(raw_input_axis_permutation) == 3
            else None
        )
    else:
        image = load_tiff_volume(dataset_surface.input_file)
        image, oracle_size_of_image, input_axis_permutation = _reorient_exact_input_volume(
            image,
            oracle_surface,
        )
        ensure_dest_run_layout(dest_run_root)
        _persist_param_storage(dest_run_root, params)
        _copy_exact_bootstrap_refs(
            dest_run_root,
            dataset_surface=dataset_surface,
            oracle_surface=oracle_surface,
        )
        _write_json_with_hash(
            dest_run_root / EXPERIMENT_PROVENANCE_PATH,
            {
                "bootstrap_kind": "init-exact-run",
                "dataset_root": str(dataset_surface.dataset_root),
                "dataset_manifest": str(dataset_surface.manifest_path),
                "dataset_hash": dataset_surface.dataset_hash,
                "input_file": str(dataset_surface.input_file),
                "oracle_root": str(oracle_surface.oracle_root),
                "oracle_manifest": (
                    str(oracle_surface.manifest_path)
                    if oracle_surface.manifest_path is not None
                    else None
                ),
                "oracle_id": oracle_surface.oracle_id,
                "matlab_batch_dir": str(oracle_surface.matlab_batch_dir),
                "matlab_source_version": oracle_surface.matlab_source_version,
                "selected_settings_paths": selected_settings_paths,
                "selected_settings_payloads": selected_settings_payloads,
                "oracle_size_of_image": (
                    list(oracle_size_of_image) if oracle_size_of_image is not None else None
                ),
                "input_axis_permutation": (
                    list(input_axis_permutation) if input_axis_permutation is not None else None
                ),
                "stop_after": args.stop_after,
                "created_at": _now_iso(),
            },
        )

        processor = SLAVVProcessor()
        processor.process_image(
            image,
            params,
            run_dir=str(dest_run_root),
            stop_after=args.stop_after,
        )

    _finalize_init_exact_run(
        dest_run_root=dest_run_root,
        dataset_surface=dataset_surface,
        oracle_surface=oracle_surface,
        params=params,
        selected_settings_paths=selected_settings_paths,
        oracle_size_of_image=oracle_size_of_image,
        input_axis_permutation=input_axis_permutation,
        stop_after=args.stop_after,
    )
    print(str(dest_run_root / RUN_MANIFEST_PATH))


def _handle_promote_report(args: argparse.Namespace) -> None:
    run_root = Path(args.run_root).expanduser().resolve()
    if not run_root.is_dir():
        raise ValueError(f"run root not found: {run_root}")
    experiment_root = _resolve_experiment_root(run_root)
    if experiment_root is None:
        raise ValueError(f"run root is not under a structured experiment root: {run_root}")
    _ensure_experiment_root_layout(experiment_root)

    report_root = (
        Path(args.report_root).expanduser().resolve()
        if args.report_root
        else experiment_root / "reports" / _entity_id_from_path(run_root)
    )
    if report_root.exists():
        raise ValueError(f"report root already exists: {report_root}")

    report_root.mkdir(parents=True, exist_ok=False)
    for relative_dir in (EXPERIMENT_REFS_DIR, EXPERIMENT_PARAMS_DIR, ANALYSIS_DIR):
        source_dir = run_root / relative_dir
        if source_dir.is_dir():
            copytree(source_dir, report_root / relative_dir)
    (report_root / METADATA_DIR).mkdir(parents=True, exist_ok=True)

    source_manifest = load_json_dict(run_root / RUN_MANIFEST_PATH) or {}
    report_manifest = {
        "manifest_version": 1,
        "kind": "promoted_report",
        "report_id": _entity_id_from_path(report_root),
        "source_run_id": _string_or_none(source_manifest.get("run_id"))
        or _entity_id_from_path(run_root),
        "source_run_root": str(run_root),
        "report_root": str(report_root),
        "dataset_hash": source_manifest.get("dataset_hash"),
        "oracle_id": source_manifest.get("oracle_id"),
        "python_commit": source_manifest.get("python_commit") or _resolve_python_commit(),
        "matlab_source_version": source_manifest.get("matlab_source_version"),
        "status": "promoted",
        "retention": "promoted",
        "timestamps": {
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
        },
    }
    _write_json_with_hash(report_root / REPORT_MANIFEST_PATH, report_manifest)

    if source_manifest:
        source_manifest["promoted_report_root"] = str(report_root)
        source_manifest["promotion_state"] = "promoted"
        source_timestamps = cast("dict[str, Any]", dict(source_manifest.get("timestamps", {})))
        source_timestamps["updated_at"] = _now_iso()
        source_manifest["timestamps"] = source_timestamps
        _write_json_with_hash(run_root / RUN_MANIFEST_PATH, source_manifest)

    _upsert_index_record(
        experiment_root,
        {
            "kind": "promoted_report",
            "id": report_manifest["report_id"],
            "status": "promoted",
            "path": str(report_root),
            "dataset_hash": report_manifest.get("dataset_hash"),
            "oracle_id": report_manifest.get("oracle_id"),
            "updated_at": report_manifest["timestamps"]["updated_at"],
        },
    )
    print(str(report_root / REPORT_MANIFEST_PATH))


def _expect_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping payload for {label}")
    return cast("dict[str, Any]", value)


def _mapping_item(payload: dict[str, Any], key: str) -> dict[str, Any]:
    return _expect_mapping(payload.get(key), key)


def _coerce_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"expected integer value for {label}")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise ValueError(f"expected integer value for {label}")


def _payload_count(payload: dict[str, Any], *, preferred_keys: tuple[str, ...], label: str) -> int:
    for key in preferred_keys:
        if key not in payload:
            continue
        value = payload[key]
        if key == "count":
            return _coerce_int(value, label=f"{label}.{key}")
        try:
            return len(value)
        except TypeError as exc:
            raise ValueError(f"expected sized payload for {label}.{key}") from exc
    expected = ", ".join(preferred_keys)
    raise ValueError(f"could not determine count for {label}; expected one of: {expected}")


def _diff_counts(current: RunCounts, baseline: RunCounts) -> dict[str, int]:
    return {
        "vertices": current.vertices - baseline.vertices,
        "edges": current.edges - baseline.edges,
        "strands": current.strands - baseline.strands,
    }


def _format_delta(value: int) -> str:
    return f"{value:+d}"


def _format_delta_line(diff_payload: dict[str, Any]) -> str:
    return (
        f"vertices={_format_delta(_coerce_int(diff_payload.get('vertices'), label='delta vertices'))} "
        f"edges={_format_delta(_coerce_int(diff_payload.get('edges'), label='delta edges'))} "
        f"strands={_format_delta(_coerce_int(diff_payload.get('strands'), label='delta strands'))}"
    )


def _selected_exact_stages(stage_arg: str) -> tuple[str, ...]:
    if stage_arg == "all":
        return cast("tuple[str, ...]", EXACT_STAGE_ORDER)
    return (stage_arg,)


def _resolve_exact_report_paths(
    dest_run_root: Path,
    report_path_arg: str | None,
) -> tuple[Path, Path]:
    if report_path_arg is None:
        return dest_run_root / EXACT_PROOF_JSON_PATH, dest_run_root / EXACT_PROOF_TEXT_PATH

    base_path = Path(report_path_arg).expanduser().resolve()
    if base_path.suffix.lower() == ".json":
        return base_path, base_path.with_suffix(".txt")
    if base_path.suffix.lower() == ".txt":
        return base_path.with_suffix(".json"), base_path
    return base_path.with_suffix(".json"), base_path.with_suffix(".txt")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    if argv is not None and len(argv) == 0:
        parser.print_help()
        raise SystemExit(0)
    args = parser.parse_args(argv)
    try:
        args.handler(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
