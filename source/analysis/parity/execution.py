"""Execution and parameter logic for native-first MATLAB-oracle parity experiments."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import psutil

from source import SLAVVProcessor
from source.io import load_tiff_volume
from source.io.matlab_exact_proof import find_single_matlab_batch_dir, find_matlab_vector_paths
from source.runtime.run_state import (
    atomic_write_json,
    atomic_write_text,
    fingerprint_jsonable,
    load_json_dict,
    stable_json_dumps,
)

from .constants import (
    ANALYSIS_DIR,
    CANDIDATE_PROGRESS_JSONL_PATH,
    CANDIDATE_PROGRESS_PLOT_PATH,
    CHECKPOINTS_DIR,
    EXPERIMENT_PARAMS_DIR,
    EXPERIMENT_PROVENANCE_PATH,
    EXPERIMENT_REFS_DIR,
    HASHES_DIR,
    MATLAB_EXACT_EDGE_SOURCE_CONSTANTS,
    METADATA_DIR,
    NORMALIZED_DIR,
    ORACLE_DISCOVERY_STAGES,
    PARAM_DIFF_PATH,
    PYTHON_DERIVED_PARAMS_PATH,
    RUN_MANIFEST_PATH,
    RUN_SNAPSHOT_PATH,
    SHARED_PARAMS_PATH,
    VALIDATED_PARAMS_PATH,
    EXACT_REQUIRED_PARAMETER_VALUES,
    EXACT_SHARED_METHOD_PARAMETER_KEYS,
    EXACT_ALLOWED_ORCHESTRATION_PARAMETER_KEYS,
)
from .index import ensure_experiment_root_layout, resolve_experiment_root, upsert_index_record
from .utils import (
    entity_id_from_path,
    now_iso,
    resolve_python_commit,
    string_or_none,
    write_json_with_hash,
)
from .models import DatasetSurface, OracleSurface, SourceRunSurface

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
            required_exact_mismatches.append({
                "key": key, "expected": normalized_expected, "found": actual_value
            })

    known_keys = (
        EXACT_SHARED_METHOD_PARAMETER_KEYS | 
        EXACT_ALLOWED_ORCHESTRATION_PARAMETER_KEYS | 
        set(EXACT_REQUIRED_PARAMETER_VALUES)
    )
    unclassified_keys = sorted(
        key for key in param_keys 
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
            key: _normalize_param_value(value) for key, value in EXACT_REQUIRED_PARAMETER_VALUES.items()
        },
        "required_exact_mismatches": required_exact_mismatches,
        "disallowed_python_only_keys": disallowed_python_only_keys,
        "unclassified_keys": unclassified_keys,
    }

def _normalize_param_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _normalize_param_value(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (tuple, list)):
        return [_normalize_param_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalize_param_value(item) for key, item in value.items()}
    return value

def persist_param_storage(dest_run_root: Path, params: dict[str, Any]) -> dict[str, Any]:
    """Persist structured parameters for exact-route runs."""
    audit = build_exact_params_audit(params)
    shared_params = cast("dict[str, Any]", dict(audit.get("shared_method_params", {})))
    derived_keys = sorted(set(params) - set(shared_params))
    
    orchestration_params = {
        key: _normalize_param_value(params[key])
        for key in derived_keys if key in EXACT_ALLOWED_ORCHESTRATION_PARAMETER_KEYS
    }
    python_only_params = {
        key: _normalize_param_value(params[key])
        for key in derived_keys if key.startswith("parity_")
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
        "required_exact_values": cast("dict[str, Any]", dict(audit.get("required_exact_values", {}))),
        "required_exact_mismatches": cast("list[dict[str, Any]]", list(audit.get("required_exact_mismatches", []))),
        "shared_params_hash": fingerprint_jsonable(shared_params),
        "python_derived_params_hash": fingerprint_jsonable(python_derived),
    }
    
    write_json_with_hash(dest_run_root / SHARED_PARAMS_PATH, shared_params)
    write_json_with_hash(dest_run_root / PYTHON_DERIVED_PARAMS_PATH, python_derived)
    write_json_with_hash(dest_run_root / PARAM_DIFF_PATH, param_diff)
    
    return {
        "shared_params": shared_params,
        "python_derived_params": python_derived,
        "param_diff": param_diff,
    }

def write_run_manifest(
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
    """Write the run manifest and update the central index."""
    src_manifest = load_json_dict(dest_run_root / RUN_MANIFEST_PATH) or {}
    run_id = string_or_none(src_manifest.get("run_id")) or entity_id_from_path(dest_run_root)
    exp_root = resolve_experiment_root(dest_run_root)
    
    ts = cast("dict[str, Any]", dict(src_manifest.get("timestamps", {})))
    created_at = string_or_none(ts.get("created_at")) or string_or_none(src_manifest.get("created_at")) or now_iso()
    updated_at = now_iso()
    
    manifest = {
        "manifest_version": 1,
        "kind": run_kind,
        "run_id": run_id,
        "command": command,
        "run_root": str(dest_run_root),
        "status": status,
        "dataset_hash": dataset_hash,
        "oracle_id": oracle_surface.oracle_id if oracle_surface else None,
        "oracle_root": str(oracle_surface.oracle_root) if oracle_surface else None,
        "python_commit": string_or_none(src_manifest.get("python_commit")) or resolve_python_commit(exp_root or Path.cwd()),
        "timestamps": {
            "created_at": created_at,
            "updated_at": updated_at,
            "completed_at": ts.get("completed_at"),
        },
        "retention": "disposable" if dest_run_root.parent.name == "runs" else "promoted",
        "params_path": str(dest_run_root / VALIDATED_PARAMS_PATH) if params_payload else None,
        "analysis_dir": str(dest_run_root / ANALYSIS_DIR),
        "normalized_dir": str(dest_run_root / NORMALIZED_DIR),
        "hashes_dir": str(dest_run_root / HASHES_DIR),
        "stage_metrics": src_manifest.get("stage_metrics", {}),
    }
    if extra:
        manifest.update(extra)
    
from .reports import persist_experiment_summary, persist_recording_tables

def resolve_input_file(
    source_surface: SourceRunSurface,
    input_arg: str | None,
    *,
    repo_root: Path,
) -> Path:
    """Resolve the input file either from the CLI or the source run snapshot provenance."""
    if input_arg:
        candidate = Path(input_arg).expanduser()
    else:
        if source_surface.run_snapshot_path is None:
            raise ValueError("source run root does not contain snapshot and no --input provided")
        snapshot = load_json_dict(source_surface.run_snapshot_path) or {}
        provenance = snapshot.get("provenance", {})
        raw_input = provenance.get("input_file")
        if not isinstance(raw_input, str):
            raise ValueError("source snapshot does not record input_file; pass --input")
        candidate = Path(raw_input)
        if not candidate.is_absolute():
            candidate = repo_root / candidate
    
    resolved = candidate.resolve()
    if not resolved.is_file():
        raise ValueError(f"input file not found: {resolved}")
    return resolved

def copy_source_surface(source_surface: SourceRunSurface, dest_run_root: Path) -> None:
    """Copy the reusable source checkpoints and reference metadata into a fresh destination root."""
    from shutil import copy2, copytree
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

    refs_dir = destination / EXPERIMENT_REFS_DIR
    copy2(source_surface.validated_params_path, refs_dir / "source_validated_params.json")
    if source_surface.comparison_report_path.is_file():
        copy2(source_surface.comparison_report_path, refs_dir / "source_comparison_report.json")
    if source_surface.run_snapshot_path and source_surface.run_snapshot_path.is_file():
        copy2(source_surface.run_snapshot_path, refs_dir / "source_run_snapshot.json")

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

def load_params_file(source_surface: SourceRunSurface, params_arg: str | None) -> dict[str, Any]:
    """Load the JSON parameters either from the CLI or the source run metadata."""
    if params_arg:
        path = Path(params_arg).expanduser().resolve()
    else:
        path = source_surface.validated_params_path
    
    payload = load_json_dict(path)
    if payload is None:
        raise ValueError(f"expected JSON object in params file: {path}")
    return payload

def persist_param_storage(dest_run_root: Path, params: dict[str, Any]) -> None:
    """Write the validated parameters into the destination run metadata."""
    meta_dir = dest_run_root / METADATA_DIR
    meta_dir.mkdir(parents=True, exist_ok=True)
    write_json_with_hash(dest_run_root / VALIDATED_PARAMS_PATH, params)

def maybe_sync_exact_vertex_checkpoint(
    source_run_root: Path,
    dest_run_root: Path,
    *,
    oracle_root: Path | None = None,
) -> str:
    """Sync the exact-route vertex checkpoint if available."""
    # Placeholder for sync logic
    return "skipped"
