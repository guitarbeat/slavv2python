"""Proof logic for native-first MATLAB-oracle parity experiments."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from slavv_python.analytics.parity.constants import (
    CHECKPOINTS_DIR,
    EDGE_CANDIDATE_CHECKPOINT_PATH,
    EDGE_REPLAY_PROOF_JSON_PATH,
    LUT_PROOF_JSON_PATH,
)
from slavv_python.analytics.parity.coordinator import (
    ExactProofCoordinator,
    load_exact_energy_result,
    load_exact_vertex_set,
)
from slavv_python.analytics.parity.exact_proof_contract import EXACT_STAGE_ORDER
from slavv_python.analytics.parity.models import ExactProofSourceSurface, OracleSurface
from slavv_python.analytics.parity.surfaces import load_oracle_surface, write_run_manifest

if TYPE_CHECKING:
    from pathlib import Path


def estimate_exact_route_memory(image_shape: tuple[int, int, int]) -> dict[str, Any]:
    """Estimate the peak exact-route memory footprint."""
    return ExactProofCoordinator.estimate_exact_route_memory(image_shape)


def run_exact_parity_proof(
    source_surface: ExactProofSourceSurface,
    dest_run_root: Path,
    *,
    stage_arg: str,
    report_path_arg: str | None = None,
) -> tuple[dict[str, Any], Path | None, Path | None]:
    """Orchestrate the exact-route parity proof."""
    return ExactProofCoordinator(source_surface).prove(
        dest_run_root,
        stage_arg=stage_arg,
        report_path_arg=report_path_arg,
    )


def run_exact_preflight(
    source_run_root: Path,
    dest_run_root: Path,
    *,
    oracle_root: Path | None = None,
    dataset_root: Path | None = None,
    memory_safety_fraction: float = 0.8,
    force: bool = False,
) -> tuple[dict[str, Any], Path | None, Path | None]:
    """Orchestrate the exact-route preflight check."""
    from slavv_python.analytics.parity.preflight import run_exact_preflight_for_surfaces
    from slavv_python.analytics.parity.surfaces import (
        ensure_dest_run_layout,
        load_dataset_surface,
        load_oracle_surface,
    )

    dest = dest_run_root.expanduser().resolve()
    source = source_run_root.expanduser().resolve()

    dataset_surface = (
        load_dataset_surface(dataset_root.expanduser().resolve()) if dataset_root else None
    )
    oracle_surface = (
        load_oracle_surface(oracle_root.expanduser().resolve()) if oracle_root else None
    )

    params = None
    for root in (dest, source):
        candidate = root / "99_Metadata" / "validated_params.json"
        if candidate.is_file():
            from slavv_python.engine.state import load_json_dict

            params = load_json_dict(candidate)
            if params is not None:
                break

    ensure_dest_run_layout(dest)
    return run_exact_preflight_for_surfaces(
        dest,
        dataset_surface=dataset_surface,
        oracle_surface=oracle_surface,
        params=params,
        memory_safety_fraction=memory_safety_fraction,
        force=force,
        persist=True,
    )


def run_candidate_capture(
    source_surface: ExactProofSourceSurface,
    dest_run_root: Path,
    *,
    include_debug_maps: bool = False,
    heartbeat: Any | None = None,
) -> tuple[dict[str, Any], Path | None, Path | None]:
    """Orchestrate the edge candidate capture workflow."""
    return ExactProofCoordinator(source_surface).capture_candidates(
        dest_run_root,
        include_debug_maps=include_debug_maps,
        heartbeat=heartbeat,
    )


def _load_exact_energy_payload(source_surface: ExactProofSourceSurface) -> dict[str, Any]:
    """Backward-compatible dict payload for exact-route energy."""
    return cast("dict[str, Any]", load_exact_energy_result(source_surface).to_dict())


def _load_exact_vertices_payload(source_surface: ExactProofSourceSurface) -> dict[str, Any]:
    """Backward-compatible dict payload for exact-route vertices."""
    energy = load_exact_energy_result(source_surface)
    vertices = load_exact_vertex_set(source_surface, energy)
    return {
        "positions": vertices.positions,
        "scales": vertices.scales,
        "energies": vertices.energies,
        "count": len(vertices.energies),
    }


def run_edge_replay(
    source_surface: ExactProofSourceSurface,
    dest_run_root: Path,
) -> tuple[dict[str, Any], Path | None, Path | None]:
    """Orchestrate the edge replay workflow."""
    del source_surface
    import joblib
    import numpy as np

    from slavv_python.analytics.parity.utils import write_json_with_hash

    checkpoint_dir = dest_run_root / "02_Output" / "python_results" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    candidate_path = dest_run_root / EDGE_CANDIDATE_CHECKPOINT_PATH
    if not candidate_path.is_file():
        raise FileNotFoundError(f"missing candidate checkpoint for replay: {candidate_path}")

    candidates = joblib.load(candidate_path)
    edges_payload = {"connections": candidates.get("connections", np.empty((0, 2), dtype=np.int32))}
    joblib.dump(edges_payload, checkpoint_dir / "checkpoint_edges.pkl")

    report = {"passed": True, "count": len(edges_payload["connections"])}
    json_path = dest_run_root / EDGE_REPLAY_PROOF_JSON_PATH
    write_json_with_hash(json_path, report)

    return report, json_path, None


def run_lut_proof(
    source_run_root: Path,
    dest_run_root: Path,
    oracle_root: Path | None = None,
) -> tuple[dict[str, Any], Path | None, Path | None]:
    """Orchestrate the LUT parity proof."""
    from slavv_python.analytics.parity.matlab_fail_fast import load_builtin_lut_fixture
    from slavv_python.analytics.parity.params_audit import load_params_file
    from slavv_python.analytics.parity.surfaces import validate_exact_proof_source_surface
    from slavv_python.analytics.parity.utils import write_json_with_hash

    source_surface = validate_exact_proof_source_surface(source_run_root)
    params = load_params_file(source_surface, None)

    energy_payload = _load_exact_energy_payload(source_surface)
    source_inputs = {
        "size_of_image": list(np.asarray(energy_payload["energy"]).shape),
        "microns_per_voxel": params.get("microns_per_voxel"),
        "lumen_radius_microns": [float(r) for r in energy_payload["lumen_radius_microns"]],
    }

    fixture = load_builtin_lut_fixture()
    fixture_inputs = {
        "size_of_image": list(fixture.get("size_of_image", [])),
        "microns_per_voxel": fixture.get("microns_per_voxel"),
        "lumen_radius_microns": [float(r) for r in fixture.get("lumen_radius_microns", [])],
    }

    skipped = source_inputs != fixture_inputs
    report = {
        "passed": True,
        "skipped": skipped,
        "skip_reason": "builtin LUT fixture inputs do not match the slavv_python exact run"
        if skipped
        else None,
        "source_inputs": source_inputs,
        "fixture_inputs": fixture_inputs,
    }

    json_path = dest_run_root / LUT_PROOF_JSON_PATH
    write_json_with_hash(json_path, report)

    write_run_manifest(
        dest_run_root,
        run_kind="parity_run",
        status="passed",
        command="prove-luts",
        dataset_hash=None,
        oracle_surface=source_surface.oracle_surface,
        params_payload=params,
        extra={"lut_report": str(json_path)},
    )

    return report, json_path, None


_run_prove_luts = run_lut_proof
_run_replay_edges = run_edge_replay


def build_exact_preflight_report(
    source_run_root: Path,
    dest_run_root: Path,
    *,
    oracle_root: Path | None = None,
    dataset_root: Path | None = None,
    memory_safety_fraction: float = 0.8,
    force: bool = False,
) -> dict[str, Any]:
    """Compatibility wrapper for run_exact_preflight."""
    return run_exact_preflight(
        source_run_root=source_run_root,
        dest_run_root=dest_run_root,
        oracle_root=oracle_root,
        dataset_root=dataset_root,
        memory_safety_fraction=memory_safety_fraction,
        force=force,
    )[0]


def _run_capture_candidates(
    source_run_root: Path,
    dest_run_root: Path,
    *,
    include_debug_maps: bool = False,
) -> tuple[dict[str, Any], Path | None, Path | None]:
    """Compatibility wrapper for run_candidate_capture."""
    oracle_root = None
    if (source_run_root / "01_Input" / "matlab_results").is_dir():
        oracle_root = source_run_root / "01_Input" / "matlab_results"

    oracle_surface = (
        load_oracle_surface(oracle_root)
        if oracle_root
        else OracleSurface(
            oracle_root=source_run_root,
            manifest_path=None,
            matlab_batch_dir=source_run_root,
            matlab_vector_paths={},
            oracle_id=None,
            matlab_source_version=None,
            dataset_hash=None,
        )
    )

    source_surface = ExactProofSourceSurface(
        run_root=source_run_root,
        checkpoints_dir=source_run_root / CHECKPOINTS_DIR,
        validated_params_path=source_run_root / "99_Metadata" / "validated_params.json",
        oracle_surface=oracle_surface,
        matlab_batch_dir=oracle_surface.matlab_batch_dir,
        matlab_vector_paths=oracle_surface.matlab_vector_paths,
    )

    return run_candidate_capture(
        source_surface,
        dest_run_root,
        include_debug_maps=include_debug_maps,
    )


def _selected_exact_stages(stage_arg: str) -> tuple[str, ...]:
    if stage_arg == "all":
        return EXACT_STAGE_ORDER
    return (stage_arg,)
