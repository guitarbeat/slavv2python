"""init-exact-run bootstrap: params derivation, refs copy, and resume classification."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy.io import loadmat

from slavv_python.analytics.parity.matlab_vector_loader import find_single_matlab_batch_dir
from slavv_python.analytics.parity.python_checkpoint_loader import sync_exact_vertex_checkpoint_from_matlab
from slavv_python.engine.state import atomic_write_json, load_json_dict

from .constants import (
    CHECKPOINTS_DIR,
    EXPERIMENT_PROVENANCE_PATH,
    EXPERIMENT_REFS_DIR,
    MATLAB_EXACT_EDGE_SOURCE_CONSTANTS,
    ORACLE_MANIFEST_PATH,
    RUN_SNAPSHOT_PATH,
)
from .models import DatasetSurface, OracleSurface
from .params_audit import normalize_param_value
from .surfaces import (
    load_oracle_surface,
    oracle_energy_size_of_image,
    write_run_manifest,
)

__all__ = [
    "_copy_exact_bootstrap_refs",
    "_finalize_init_exact_run",
    "_reorient_exact_input_volume",
    "_resolve_existing_init_exact_run",
    "derive_exact_params_from_oracle",
    "maybe_sync_exact_vertex_checkpoint",
]


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
    if oracle_surface.matlab_batch_dir is None:
        raise ValueError("oracle surface is missing matlab_batch_dir")
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
        "microns_per_voxel": normalize_param_value(energy_settings["microns_per_voxel"]),
        "radius_of_smallest_vessel_in_microns": normalize_param_value(
            energy_settings["radius_of_smallest_vessel_in_microns"]
        ),
        "radius_of_largest_vessel_in_microns": normalize_param_value(
            energy_settings["radius_of_largest_vessel_in_microns"]
        ),
        "sample_index_of_refraction": normalize_param_value(
            energy_settings["sample_index_of_refraction"]
        ),
        "numerical_aperture": normalize_param_value(energy_settings["numerical_aperture"]),
        "excitation_wavelength_in_microns": normalize_param_value(
            energy_settings["excitation_wavelength_in_microns"]
        ),
        "scales_per_octave": normalize_param_value(energy_settings["scales_per_octave"]),
        "max_voxels_per_node_energy": normalize_param_value(
            energy_settings["max_voxels_per_node_energy"]
        ),
        "gaussian_to_ideal_ratio": normalize_param_value(energy_settings["gaussian_to_ideal_ratio"]),
        "spherical_to_annular_ratio": normalize_param_value(
            energy_settings["spherical_to_annular_ratio"]
        ),
        "approximating_PSF": bool(normalize_param_value(energy_settings["approximating_PSF"])),
        "space_strel_apothem": normalize_param_value(vertex_settings["space_strel_apothem"]),
        "energy_upper_bound": normalize_param_value(vertex_settings["energy_upper_bound"]),
        "max_voxels_per_node": normalize_param_value(vertex_settings["max_voxels_per_node"]),
        "length_dilation_ratio": normalize_param_value(vertex_settings["length_dilation_ratio"]),
        "max_edge_length_per_origin_radius": normalize_param_value(
            edge_settings["max_edge_length_per_origin_radius"]
        ),
        "space_strel_apothem_edges": normalize_param_value(
            edge_settings["space_strel_apothem_edges"]
        ),
        "number_of_edges_per_vertex": normalize_param_value(
            edge_settings["number_of_edges_per_vertex"]
        ),
    }
    params.update(MATLAB_EXACT_EDGE_SOURCE_CONSTANTS)

    path_map = {stage: str(path) for stage, path in settings_paths.items()}
    normalized_payloads = {
        stage: normalize_param_value(payload) for stage, payload in settings_payloads.items()
    }
    return params, path_map, normalized_payloads


def _reorient_exact_input_volume(
    image: np.ndarray,
    oracle_surface: OracleSurface,
) -> tuple[np.ndarray, tuple[int, int, int] | None, tuple[int, int, int] | None]:
    oracle_size = oracle_energy_size_of_image(oracle_surface)
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
            return reordered_image, oracle_size, cast("tuple[int, int, int]", p)

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
    if oracle_surface.matlab_batch_dir is None:
        raise ValueError("oracle surface is missing matlab_batch_dir")
    copytree(
        oracle_surface.matlab_batch_dir, matlab_results_dir / oracle_surface.matlab_batch_dir.name
    )


def _resolve_existing_init_exact_run(
    dest_run_root: Path,
    dataset_surface: DatasetSurface,
    oracle_surface: OracleSurface,
    stop_after: str | None,
    *,
    allow_resume: bool = False,
) -> str:
    """Classify how init-exact-run should treat an existing destination directory."""
    if not dest_run_root.is_dir():
        return "fresh"
    prov = load_json_dict(dest_run_root / EXPERIMENT_PROVENANCE_PATH)
    if prov is None:
        return "fresh"
    if not (
        prov.get("dataset_hash") == dataset_surface.dataset_hash
        and prov.get("oracle_id") == oracle_surface.oracle_id
        and prov.get("stop_after") == stop_after
    ):
        return "fresh"

    snapshot = load_json_dict(dest_run_root / RUN_SNAPSHOT_PATH) or {}
    if snapshot.get("status") == "running":
        if allow_resume:
            return "resume_pipeline"
        import sys

        sys.exit(
            "seed run snapshot status is still 'running'; use resume-exact-run "
            "(or init-exact-run --resume) to continue, or remove the run directory"
        )
    return "finalize_only"


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
    del selected_settings_paths
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
        atomic_write_json(snapshot_path, snapshot)


def maybe_sync_exact_vertex_checkpoint(
    source_run_root: Path,
    dest_run_root: Path,
    *,
    oracle_root: Path | None = None,
) -> bool:
    """Sync the exact-route vertex checkpoint if available."""
    src_checkpoints = source_run_root / CHECKPOINTS_DIR
    dest_checkpoints = dest_run_root / CHECKPOINTS_DIR
    src_vertex = src_checkpoints / "checkpoint_vertices.pkl"
    dest_vertex = dest_checkpoints / "checkpoint_vertices.pkl"

    if not src_vertex.is_file():
        return False

    if oracle_root is None:
        if (source_run_root / ORACLE_MANIFEST_PATH).is_file():
            oracle_root = source_run_root
        else:
            try:
                find_single_matlab_batch_dir(source_run_root)
                oracle_root = source_run_root
            except Exception:
                return False

    try:
        oracle_surface = load_oracle_surface(oracle_root)
        if oracle_surface.matlab_batch_dir is None:
            return False
        sync_exact_vertex_checkpoint_from_matlab(
            dest_vertex,
            oracle_surface.matlab_batch_dir,
        )
        return True
    except Exception:
        return False
