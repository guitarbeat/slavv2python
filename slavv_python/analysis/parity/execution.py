"""Execution and parameter logic for native-first MATLAB-oracle parity experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy.io import loadmat
from slavv_python.io.matlab_exact_proof import (
    find_matlab_vector_paths,
    find_single_matlab_batch_dir,
)
from slavv_python.runtime.run_state import (
    fingerprint_file,
    fingerprint_jsonable,
    load_json_dict,
)

from .constants import (
    ANALYSIS_DIR,
    ANALYSIS_TABLES_DIR,
    CHECKPOINTS_DIR,
    COMPARISON_REPORT_PATH,
    DATASET_INPUT_DIR,
    DATASET_MANIFEST_PATH,
    EXACT_ALLOWED_ORCHESTRATION_PARAMETER_KEYS,
    EXACT_REQUIRED_PARAMETER_VALUES,
    EXACT_SHARED_METHOD_PARAMETER_KEYS,
    EXPERIMENT_PARAMS_DIR,
    EXPERIMENT_PROVENANCE_PATH,
    EXPERIMENT_REFS_DIR,
    HASHES_DIR,
    MATLAB_EXACT_EDGE_SOURCE_CONSTANTS,
    METADATA_DIR,
    NORMALIZED_DIR,
    ORACLE_DISCOVERY_STAGES,
    ORACLE_MANIFEST_PATH,
    PARAM_DIFF_PATH,
    PYTHON_DERIVED_PARAMS_PATH,
    RUN_MANIFEST_PATH,
    RUN_SNAPSHOT_PATH,
    SHARED_PARAMS_PATH,
    VALIDATED_PARAMS_PATH,
)
from .index import resolve_experiment_root, upsert_index_record
from .models import (
    DatasetSurface,
    ExactProofSourceSurface,
    OracleSurface,
    RunCounts,
    SourceRunSurface,
)
from .utils import (
    entity_id_from_path,
    now_iso,
    resolve_python_commit,
    string_or_none,
    write_json_with_hash,
)


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
                {"key": key, "expected": normalized_expected, "found": actual_value}
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
            "list[dict[str, Any]]", list(audit.get("required_exact_mismatches", []))
        ),
        "disallowed_python_only_keys": cast(
            "list[str]", list(audit.get("disallowed_python_only_keys", []))
        ),
        "shared_params_hash": fingerprint_jsonable(shared_params),
        "python_derived_params_hash": fingerprint_jsonable(python_derived),
    }

    write_json_with_hash(dest_run_root / SHARED_PARAMS_PATH, shared_params)
    write_json_with_hash(dest_run_root / PYTHON_DERIVED_PARAMS_PATH, python_derived)
    write_json_with_hash(dest_run_root / PARAM_DIFF_PATH, param_diff)
    write_json_with_hash(dest_run_root / VALIDATED_PARAMS_PATH, params)

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
    created_at = (
        string_or_none(ts.get("created_at"))
        or string_or_none(src_manifest.get("created_at"))
        or now_iso()
    )
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
        "python_commit": string_or_none(src_manifest.get("python_commit"))
        or resolve_python_commit(exp_root or Path.cwd()),
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
        "stages": src_manifest.get("stages", {}),
        "artifacts": src_manifest.get("artifacts", {}),
    }
    if extra:
        manifest.update(extra)

    write_json_with_hash(dest_run_root / RUN_MANIFEST_PATH, manifest)
    write_json_with_hash(dest_run_root / RUN_SNAPSHOT_PATH, manifest)
    upsert_index_record(exp_root, manifest)
    return manifest


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
            raise ValueError(
                "source run root does not contain snapshot and no --input was provided"
            )
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


def validate_exact_proof_source_surface(source_run_root: Path) -> ExactProofSourceSurface:
    """Validate the authority surface for an exact-route proof against a MATLAB oracle."""
    run_root = source_run_root.resolve()
    manifest = load_json_dict(run_root / RUN_MANIFEST_PATH)
    if not manifest:
        raise ValueError(f"source run root is missing manifest: {run_root / RUN_MANIFEST_PATH}")

    validated_params_path = run_root / VALIDATED_PARAMS_PATH
    if not validated_params_path.is_file():
        raise ValueError(f"source run root is missing validated params: {validated_params_path}")

    stored_oracle_root = string_or_none(manifest.get("oracle_root"))
    if stored_oracle_root:
        oracle_surface = load_oracle_surface(Path(stored_oracle_root))
    else:
        oracle_surface = load_oracle_surface(run_root)

    return ExactProofSourceSurface(
        run_root=run_root,
        checkpoints_dir=run_root / CHECKPOINTS_DIR,
        validated_params_path=validated_params_path,
        oracle_surface=oracle_surface,
        matlab_batch_dir=oracle_surface.matlab_batch_dir,
        matlab_vector_paths=oracle_surface.matlab_vector_paths,
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

    # Audit if exact route is requested
    if payload.get("comparison_exact_network") is True:
        audit = build_exact_params_audit(payload)
        if not audit["passed"]:
            mismatches = audit.get("required_exact_mismatches", [])
            disallowed = audit.get("disallowed_python_only_keys", [])
            msg = "disallowed exact parameters:"
            if mismatches:
                msg += f" mismatches={mismatches}"
            if disallowed:
                msg += f" disallowed Python-only parity keys={disallowed}"
            raise ValueError(msg)

    return payload


def ensure_dest_run_layout(dest_run_root: Path) -> None:
    """Ensure the destination run root has the required subdirectory structure."""
    for subdir in (
        ANALYSIS_DIR,
        ANALYSIS_TABLES_DIR,
        CHECKPOINTS_DIR,
        EXPERIMENT_REFS_DIR,
        EXPERIMENT_PARAMS_DIR,
        METADATA_DIR,
        HASHES_DIR,
        NORMALIZED_DIR,
    ):
        (dest_run_root / subdir).mkdir(parents=True, exist_ok=True)


def load_oracle_surface(oracle_root: Path | None) -> OracleSurface:
    """Load the authority surface for a preserved MATLAB truth package."""
    if oracle_root is None:
        return OracleSurface(
            oracle_root=None,
            manifest_path=None,
            matlab_batch_dir=None,
            matlab_vector_paths={},
            oracle_id=None,
            matlab_source_version=None,
            dataset_hash=None,
        )
    resolved_root = oracle_root.expanduser().resolve()
    manifest_path = resolved_root / ORACLE_MANIFEST_PATH
    manifest = load_json_dict(manifest_path)
    matlab_batch_dir = find_single_matlab_batch_dir(resolved_root)
    return OracleSurface(
        oracle_root=resolved_root,
        manifest_path=manifest_path if manifest_path.is_file() else None,
        matlab_batch_dir=matlab_batch_dir,
        matlab_vector_paths=find_matlab_vector_paths(matlab_batch_dir, ORACLE_DISCOVERY_STAGES),
        oracle_id=string_or_none(manifest.get("oracle_id")) if manifest else None,
        matlab_source_version=(
            string_or_none(manifest.get("matlab_source_version")) if manifest else None
        ),
        dataset_hash=string_or_none(manifest.get("dataset_hash")) if manifest else None,
    )


def load_dataset_surface(dataset_root: Path) -> DatasetSurface:
    """Load the authority surface for a preserved dataset package."""
    resolved_root = dataset_root.expanduser().resolve()
    manifest_path = resolved_root / DATASET_MANIFEST_PATH
    manifest = load_json_dict(manifest_path)
    if manifest is None:
        raise ValueError(f"missing dataset manifest: {manifest_path}")

    stored_input = string_or_none(manifest.get("stored_input_file"))
    input_filename = string_or_none(manifest.get("input_filename"))
    dataset_hash = string_or_none(manifest.get("dataset_hash"))

    if dataset_hash is None:
        raise ValueError(f"dataset manifest is missing dataset_hash: {manifest_path}")

    # Robust path resolution: try stored absolute path, then try relative to dataset_root
    input_file: Path | None = None
    if stored_input:
        input_file = Path(stored_input).expanduser().resolve()
        if not input_file.is_file():
            input_file = None

    if input_file is None and input_filename:
        # Fallback to standard 01_Input/<filename> layout
        input_file = resolved_root / DATASET_INPUT_DIR / input_filename
        if not input_file.is_file():
            input_file = None

    if input_file is None:
        raise ValueError(f"dataset input file not found for {dataset_hash} under {resolved_root}")

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
    """Derive Python parameters from MATLAB settings artifacts."""
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
        "microns_per_voxel": _normalize_param_value(energy_settings["microns_per_voxel"]),
        "radius_of_smallest_vessel_in_microns": _normalize_param_value(
            energy_settings["radius_of_smallest_vessel_in_microns"]
        ),
        "radius_of_largest_vessel_in_microns": _normalize_param_value(
            energy_settings["radius_of_largest_vessel_in_microns"]
        ),
        "sample_index_of_refraction": _normalize_param_value(
            energy_settings["sample_index_of_refraction"]
        ),
        "numerical_aperture": _normalize_param_value(energy_settings["numerical_aperture"]),
        "excitation_wavelength_in_microns": _normalize_param_value(
            energy_settings["excitation_wavelength_in_microns"]
        ),
        "scales_per_octave": _normalize_param_value(energy_settings["scales_per_octave"]),
        "max_voxels_per_node_energy": _normalize_param_value(
            energy_settings["max_voxels_per_node_energy"]
        ),
        "gaussian_to_ideal_ratio": _normalize_param_value(
            energy_settings["gaussian_to_ideal_ratio"]
        ),
        "spherical_to_annular_ratio": _normalize_param_value(
            energy_settings["spherical_to_annular_ratio"]
        ),
        "approximating_PSF": bool(_normalize_param_value(energy_settings["approximating_PSF"])),
        "space_strel_apothem": _normalize_param_value(vertex_settings["space_strel_apothem"]),
        "energy_upper_bound": _normalize_param_value(vertex_settings["energy_upper_bound"]),
        "max_voxels_per_node": _normalize_param_value(vertex_settings["max_voxels_per_node"]),
        "length_dilation_ratio": _normalize_param_value(vertex_settings["length_dilation_ratio"]),
        "max_edge_length_per_origin_radius": _normalize_param_value(
            edge_settings["max_edge_length_per_origin_radius"]
        ),
        "space_strel_apothem_edges": _normalize_param_value(
            edge_settings["space_strel_apothem_edges"]
        ),
        "number_of_edges_per_vertex": _normalize_param_value(
            edge_settings["number_of_edges_per_vertex"]
        ),
    }
    params.update(MATLAB_EXACT_EDGE_SOURCE_CONSTANTS)

    path_map = {stage: str(path) for stage, path in settings_paths.items()}
    normalized_payloads = {
        stage: _normalize_param_value(payload) for stage, payload in settings_payloads.items()
    }
    return params, path_map, normalized_payloads


def _oracle_energy_size_of_image(oracle_surface: OracleSurface) -> tuple[int, int, int] | None:
    energy_path = oracle_surface.matlab_vector_paths.get("energy")
    if energy_path is None:
        return None
    payload = loadmat(energy_path, squeeze_me=False, struct_as_record=False)
    energy = payload.get("energy")
    if energy is not None and hasattr(energy, "shape"):
        return cast("tuple[int, int, int]", tuple(int(value) for value in energy.shape))

    size_of_image = payload.get("size_of_image")
    if size_of_image is not None:
        size_array = np.atleast_1d(np.squeeze(size_of_image))
        if size_array.ndim == 1 and len(size_array) == 3:
            return cast("tuple[int, int, int]", tuple(int(v) for v in size_array))

    return None


def _reorient_exact_input_volume(
    image: np.ndarray,
    oracle_surface: OracleSurface,
) -> tuple[np.ndarray, tuple[int, int, int] | None, tuple[int, int, int] | None]:
    oracle_size = _oracle_energy_size_of_image(oracle_surface)
    if oracle_size is None:
        return image, None, None

    input_size = cast("tuple[int, int, int]", tuple(int(value) for value in image.shape))
    if input_size == oracle_size:
        return image, oracle_size, None

    import itertools

    for p in itertools.permutations((0, 1, 2)):
        reordered_size = (input_size[p[0]], input_size[p[1]], input_size[p[2]])
        if reordered_size == oracle_size:
            reordered_image = np.transpose(image, p)
            return reordered_image, oracle_size, p

    return image, oracle_size, None


def _copy_exact_bootstrap_refs(
    dest_run_root: Path,
    *,
    dataset_surface: DatasetSurface,
    oracle_surface: OracleSurface,
) -> None:
    from shutil import copy2, copytree

    refs_dir = dest_run_root / EXPERIMENT_REFS_DIR
    copy2(dataset_surface.input_file, refs_dir / dataset_surface.input_file.name)
    copy2(dataset_surface.manifest_path, refs_dir / "dataset_manifest.json")
    if oracle_surface.manifest_path:
        copy2(oracle_surface.manifest_path, refs_dir / "oracle_manifest.json")

    matlab_results_dir = dest_run_root / "01_Input" / "matlab_results"
    matlab_results_dir.mkdir(parents=True, exist_ok=True)
    copytree(
        oracle_surface.matlab_batch_dir, matlab_results_dir / oracle_surface.matlab_batch_dir.name
    )


def _resolve_existing_init_exact_run(
    dest_run_root: Path,
    dataset_surface: DatasetSurface,
    oracle_surface: OracleSurface,
    stop_after: str | None,
) -> bool:
    if not dest_run_root.is_dir():
        return False
    prov = load_json_dict(dest_run_root / EXPERIMENT_PROVENANCE_PATH)
    if prov is None:
        return False
    if (
        prov.get("dataset_hash") == dataset_surface.dataset_hash
        and prov.get("oracle_id") == oracle_surface.oracle_id
        and prov.get("stop_after") == stop_after
    ):
        from .constants import RUN_SNAPSHOT_PATH

        snapshot = load_json_dict(dest_run_root / RUN_SNAPSHOT_PATH) or {}
        if snapshot.get("status") == "running":
            import sys

            sys.exit("seed run is still active; stop it or use --force-rerun-from to override")
        return True
    return False


def _finalize_init_exact_run(
    dest_run_root: Path,
    dataset_surface: DatasetSurface,
    oracle_surface: OracleSurface,
    params: dict[str, Any],
    selected_settings_paths: dict[str, str],
    oracle_size_of_image: tuple[int, int, int] | None,
    input_axis_permutation: tuple[int, int, int] | None,
    stop_after: str | None,
) -> None:
    write_run_manifest(
        dest_run_root,
        run_kind="parity_source_run",
        status="completed",
        command="init-exact-run",
        dataset_hash=dataset_surface.dataset_hash,
        oracle_surface=oracle_surface,
        params_payload=params,
        extra={
            "stop_after": stop_after,
            "oracle_size_of_image": list(oracle_size_of_image) if oracle_size_of_image else None,
            "input_axis_permutation": list(input_axis_permutation)
            if input_axis_permutation
            else None,
        },
    )
    from .reports import persist_recording_tables

    persist_recording_tables(dest_run_root)

    from .constants import RUN_SNAPSHOT_PATH

    snapshot_path = dest_run_root / RUN_SNAPSHOT_PATH
    if snapshot_path.is_file():
        snapshot = load_json_dict(snapshot_path) or {}
        prov = snapshot.get("provenance", {})
        prov.update(
            {
                "input_file": str(dataset_surface.input_file),
                "oracle_id": oracle_surface.oracle_id,
            }
        )
        snapshot["provenance"] = prov
        from slavv_python.runtime.run_state import atomic_write_json

        atomic_write_json(snapshot_path, snapshot)


def maybe_sync_exact_vertex_checkpoint(
    source_run_root: Path,
    dest_run_root: Path,
    *,
    oracle_root: Path | None = None,
) -> str:
    """Sync the exact-route vertex checkpoint if available."""
    from slavv_python.io.matlab_exact_proof import sync_exact_vertex_checkpoint_from_matlab

    src_checkpoints = source_run_root / CHECKPOINTS_DIR
    dest_checkpoints = dest_run_root / CHECKPOINTS_DIR
    src_vertex = src_checkpoints / "checkpoint_vertices.pkl"
    dest_vertex = dest_checkpoints / "checkpoint_vertices.pkl"

    if not src_vertex.is_file():
        return False

    if oracle_root is None:
        # Try to find an oracle manifest or batch dir in the source run root
        if (source_run_root / ORACLE_MANIFEST_PATH).is_file():
            oracle_root = source_run_root
        else:
            try:
                # find_single_matlab_batch_dir will raise if multiple or none
                find_single_matlab_batch_dir(source_run_root)
                oracle_root = source_run_root
            except Exception:
                return False

    try:
        oracle_surface = load_oracle_surface(oracle_root)
        sync_exact_vertex_checkpoint_from_matlab(
            dest_vertex,
            oracle_surface.matlab_batch_dir,
        )
        return True
    except Exception:
        return False


def extract_matlab_counts(report_payload: dict[str, Any]) -> RunCounts:
    """Extract RunCounts from a comparison report for the MATLAB side."""
    # This logic was buried in the old script, let's re-implement it simply
    # or look for it in the old script.
    # It usually comes from report_payload['matlab_counts']
    counts = report_payload.get("matlab_counts", {})
    return RunCounts(
        vertices=int(counts.get("vertices", 0)),
        edges=int(counts.get("edges", 0)),
        strands=int(counts.get("strands", 0)),
    )


def extract_source_python_counts(report_payload: dict[str, Any]) -> RunCounts:
    """Extract RunCounts from a comparison report for the source Python side."""
    counts = report_payload.get("python_counts", {})
    return RunCounts(
        vertices=int(counts.get("vertices", 0)),
        edges=int(counts.get("edges", 0)),
        strands=int(counts.get("strands", 0)),
    )


def read_python_counts_from_run(run_root: Path) -> RunCounts:
    """Read RunCounts from the checkpoints in a run root."""
    from slavv_python.utils.safe_unpickle import safe_load

    checkpoints = run_root / CHECKPOINTS_DIR

    v_path = checkpoints / "checkpoint_vertices.pkl"
    e_path = checkpoints / "checkpoint_edges.pkl"
    n_path = checkpoints / "checkpoint_network.pkl"

    v_count = 0
    if v_path.is_file():
        v_payload = safe_load(v_path)
        if isinstance(v_payload, dict):
            v_count = len(v_payload.get("positions", []))

    e_count = 0
    if e_path.is_file():
        e_payload = safe_load(e_path)
        if isinstance(e_payload, dict):
            e_count = len(e_payload.get("connections", []))

    n_count = 0
    if n_path.is_file():
        n_payload = safe_load(n_path)
        if isinstance(n_payload, dict):
            n_count = len(n_payload.get("strands", []))

    return RunCounts(vertices=v_count, edges=e_count, strands=n_count)
