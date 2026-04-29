"""Developer helpers for native-first MATLAB-oracle parity experiments."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from shutil import copy2, copytree
from typing import Any, cast

import numpy as np
import psutil

REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_DIR = REPO_ROOT / "source"
DEV_RUNS_ROOT = REPO_ROOT / "dev" / "runs"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

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
DATASET_INPUT_DIR = Path("01_Input")
COMPARISON_REPORT_PATH = ANALYSIS_DIR / "comparison_report.json"
EDGE_CANDIDATE_CHECKPOINT_PATH = CHECKPOINTS_DIR / "checkpoint_edge_candidates.pkl"
EDGE_REPLAY_PROOF_JSON_PATH = ANALYSIS_DIR / "edge_replay_proof.json"
EDGE_REPLAY_PROOF_TEXT_PATH = ANALYSIS_DIR / "edge_replay_proof.txt"
EXACT_PROOF_JSON_PATH = ANALYSIS_DIR / "exact_proof.json"
EXACT_PROOF_TEXT_PATH = ANALYSIS_DIR / "exact_proof.txt"
LUT_PROOF_JSON_PATH = ANALYSIS_DIR / "lut_proof.json"
LUT_PROOF_TEXT_PATH = ANALYSIS_DIR / "lut_proof.txt"
PREFLIGHT_EXACT_JSON_PATH = ANALYSIS_DIR / "preflight_exact.json"
PREFLIGHT_EXACT_TEXT_PATH = ANALYSIS_DIR / "preflight_exact.txt"
RUN_SNAPSHOT_PATH = METADATA_DIR / "run_snapshot.json"
SUMMARY_JSON_PATH = ANALYSIS_DIR / "experiment_summary.json"
SUMMARY_TEXT_PATH = ANALYSIS_DIR / "experiment_summary.txt"
VALIDATED_PARAMS_PATH = METADATA_DIR / "validated_params.json"
CANDIDATE_COVERAGE_JSON_PATH = ANALYSIS_DIR / "candidate_coverage.json"
CANDIDATE_COVERAGE_TEXT_PATH = ANALYSIS_DIR / "candidate_coverage.txt"
SHARED_PARAMS_PATH = EXPERIMENT_PARAMS_DIR / "shared_params.json"
PYTHON_DERIVED_PARAMS_PATH = EXPERIMENT_PARAMS_DIR / "python_derived_params.json"
PARAM_DIFF_PATH = EXPERIMENT_PARAMS_DIR / "param_diff.json"
HEARTBEAT_INTERVAL_ITERATIONS = 512
DEFAULT_MEMORY_SAFETY_FRACTION = 0.8
EXACT_SHARED_METHOD_PARAMETER_KEYS = frozenset(
    {
        "approximating_PSF",
        "bandpass_window",
        "direction_method",
        "discrete_tracing",
        "edge_method",
        "energy_method",
        "energy_projection_mode",
        "energy_sign",
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
        matlab_vector_paths=find_matlab_vector_paths(matlab_batch_dir),
        oracle_id=_string_or_none(manifest.get("oracle_id")) if manifest else None,
        matlab_source_version=(
            _string_or_none(manifest.get("matlab_source_version")) if manifest else None
        ),
        dataset_hash=_string_or_none(manifest.get("dataset_hash")) if manifest else None,
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
            input_bytes
            if input_bytes is not None
            else existing_manifest.get("input_bytes")
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

    vector_paths = find_matlab_vector_paths(raw_batch_dir)
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
    resolved_oracle_id = f"{matlab_batch_dir.name}_{fingerprint_file(find_matlab_vector_paths(matlab_batch_dir)['edges'])[:12]}"
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
    current_pid = int(psutil.Process().pid)
    collisions: list[dict[str, Any]] = []
    normalized_dest = str(dest_run_root.resolve()).lower()
    owner_commands = {"rerun-python", "capture-candidates", "replay-edges", "fail-fast"}
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        info = process.info
        pid = int(info.get("pid", -1))
        if pid == current_pid:
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
) -> None:
    """Persist lightweight run progress for long candidate-capture runs."""
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
            "artifacts": {
                "edge_candidate_iterations": str(int(iteration_count)),
                "edge_candidate_count": str(int(candidate_count)),
            },
            "errors": [],
            "provenance": {},
        },
    )


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
    report_payload = compare_lut_fixture_payload(
        load_builtin_lut_fixture(),
        size_of_image=size_of_image,
        microns_per_voxel=np.asarray(
            params.get("microns_per_voxel", [1.0, 1.0, 1.0]), dtype=np.float32
        ),
        lumen_radius_microns=np.asarray(energy_payload["lumen_radius_microns"], dtype=np.float32),
    )
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

    def _heartbeat(iteration_count: int, candidate_count: int) -> None:
        write_capture_candidates_snapshot(
            dest_root,
            detail=(
                "Generating edge candidates through MATLAB-style frontier workflow "
                f"(iterations={iteration_count}, candidates={candidate_count})"
            ),
            iteration_count=iteration_count,
            candidate_count=candidate_count,
        )

    write_capture_candidates_snapshot(
        dest_root,
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
        }
    )
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
        },
    )
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
