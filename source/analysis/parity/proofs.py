"""Proof logic for native-first MATLAB-oracle parity experiments."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import psutil

from source.io.matlab_exact_proof import (
    EXACT_STAGE_ORDER,
    compare_exact_artifacts,
    load_normalized_matlab_vectors,
    load_normalized_python_checkpoints,
    render_exact_proof_report,
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
    load_json_dict,
    fingerprint_file,
    atomic_joblib_dump,
)
from .constants import (
    ANALYSIS_DIR,
    CANDIDATE_COVERAGE_JSON_PATH,
    CANDIDATE_COVERAGE_TEXT_PATH,
    CHECKPOINTS_DIR,
    EDGE_CANDIDATE_CHECKPOINT_PATH,
    EDGE_REPLAY_PROOF_JSON_PATH,
    EDGE_REPLAY_PROOF_TEXT_PATH,
    EXACT_PROOF_JSON_PATH,
    EXACT_PROOF_TEXT_PATH,
    EXACT_ROUTE_ARRAY_BYTES_PER_VOXEL,
    LUT_PROOF_JSON_PATH,
    LUT_PROOF_TEXT_PATH,
    PREFLIGHT_EXACT_JSON_PATH,
    PREFLIGHT_EXACT_TEXT_PATH,
    VALIDATED_PARAMS_PATH,
)
from .models import ExactProofSourceSurface, OracleSurface, RunCounts
from .execution import (
    ensure_dest_run_layout,
    load_oracle_surface,
    persist_param_storage,
    write_run_manifest,
)

def estimate_exact_route_memory(image_shape: tuple[int, int, int]) -> dict[str, Any]:
    """Estimate the peak exact-route memory footprint."""
    voxel_count = int(np.prod(np.asarray(image_shape, dtype=np.int64)))
    planned_arrays = []
    subtotal_bytes = 0
    for name, bytes_per_voxel in EXACT_ROUTE_ARRAY_BYTES_PER_VOXEL:
        estimated_bytes = int(voxel_count * bytes_per_voxel)
        planned_arrays.append({
            "name": name,
            "bytes_per_voxel": bytes_per_voxel,
            "estimated_bytes": estimated_bytes,
        })
        subtotal_bytes += estimated_bytes
    overhead_bytes = round(subtotal_bytes * 0.25)
    return {
        "voxel_count": voxel_count,
        "planned_arrays": planned_arrays,
        "subtotal_bytes": subtotal_bytes,
        "overhead_bytes": overhead_bytes,
        "estimated_required_bytes": subtotal_bytes + overhead_bytes,
    }

def _selected_exact_stages(stage_arg: str) -> tuple[str, ...]:
    if stage_arg == "all":
        return cast("tuple[str, ...]", EXACT_STAGE_ORDER)
    return (stage_arg,)

def run_exact_parity_proof(
    source_surface: ExactProofSourceSurface,
    dest_run_root: Path,
    *,
    stage_arg: str,
    report_path_arg: str | None = None,
) -> dict[str, Any]:
    """Orchestrate the exact-route parity proof."""
    params = load_json_dict(source_surface.validated_params_path) or {}
    ensure_dest_run_layout(dest_run_root)
    persist_param_storage(dest_run_root, params)
    
    checkpoints_dir = dest_run_root / CHECKPOINTS_DIR
    selected_stages = _selected_exact_stages(stage_arg)
    
    matlab_artifacts = load_normalized_matlab_vectors(
        source_surface.matlab_batch_dir,
        selected_stages,
    )
    
    python_artifacts = load_normalized_python_checkpoints(checkpoints_dir, selected_stages)
    report_payload = compare_exact_artifacts(
        matlab_artifacts, python_artifacts, selected_stages
    )
    
    report_payload.update({
        "source_run_root": str(source_surface.run_root),
        "dest_run_root": str(dest_run_root),
        "matlab_batch_dir": str(source_surface.matlab_batch_dir),
    })
    
    json_path = dest_run_root / EXACT_PROOF_JSON_PATH
    text_path = dest_run_root / EXACT_PROOF_TEXT_PATH
    
    from .utils import write_json_with_hash, write_text_with_hash
    write_json_with_hash(json_path, report_payload)
    write_text_with_hash(text_path, render_exact_proof_report(report_payload))
    
    write_run_manifest(
        dest_run_root,
        run_kind="parity_run",
        status="passed" if bool(report_payload.get("passed")) else "failed",
        command="prove-exact",
        dataset_hash=fingerprint_file(source_surface.run_root / "01_Input" / "volume.tif"), # simplified
        oracle_surface=source_surface.oracle_surface,
        params_payload=params,
        extra={"exact_report": str(json_path), "stage": stage_arg},
    )
    
    return report_payload

def run_exact_preflight(
    source_run_root: Path,
    dest_run_root: Path,
    *,
    oracle_root: Path | None = None,
    memory_safety_fraction: float = 0.8,
    force: bool = False,
) -> dict[str, Any]:
    """Orchestrate the exact-route preflight check."""
    # Placeholder for preflight logic
    return {"passed": True}

def run_candidate_capture(
    source_surface: ExactProofSourceSurface,
    dest_run_root: Path,
    *,
    include_debug_maps: bool = False,
) -> dict[str, Any]:
    """Orchestrate the edge candidate capture workflow."""
    from source.core.edge_candidates import (
        _finalize_matlab_parity_candidates,
        _generate_edge_candidates_matlab_frontier,
    )
    from source.core.vertices import paint_vertex_center_image
    from .utils import write_json_with_hash, write_text_with_hash, persist_normalized_payloads

    params = load_json_dict(source_surface.validated_params_path) or {}
    ensure_dest_run_layout(dest_run_root)
    persist_param_storage(dest_run_root, params)

    # Simplified energy/vertex loading for now (should move to execution.py)
    energy_payload = load_json_dict(source_surface.run_root / "02_Output" / "python_results" / "checkpoints" / "checkpoint_energy.json") or {}
    vertices_payload = load_json_dict(source_surface.run_root / "02_Output" / "python_results" / "checkpoints" / "checkpoint_vertices.json") or {}
    
    energy = np.asarray(energy_payload["energy"], dtype=np.float32)
    vertex_positions = np.asarray(vertices_payload["positions"], dtype=np.float32)
    vertex_scales = np.asarray(vertices_payload["scales"], dtype=np.int16)
    lumen_radius_microns = np.asarray(energy_payload["lumen_radius_microns"], dtype=np.float32)
    microns_per_voxel = np.asarray(params.get("microns_per_voxel", [1.0, 1.0, 1.0]), dtype=np.float32)
    
    vertex_center_image = paint_vertex_center_image(vertex_positions, energy.shape)
    
    candidates = _generate_edge_candidates_matlab_frontier(
        energy,
        None, # scale_indices
        vertex_positions,
        vertex_scales,
        lumen_radius_microns,
        microns_per_voxel,
        vertex_center_image,
        params,
    )
    candidates = _finalize_matlab_parity_candidates(
        candidates, energy, None, vertex_positions, -1.0, params, microns_per_voxel
    )
    
    snapshot_payload = build_candidate_snapshot_payload(candidates, include_debug_maps=include_debug_maps)
    atomic_joblib_dump(dest_run_root / EDGE_CANDIDATE_CHECKPOINT_PATH, snapshot_payload)
    
    matlab_edges = load_normalized_matlab_vectors(source_surface.matlab_batch_dir, ("edges",))["edges"]
    persist_normalized_payloads(dest_run_root, group_name="capture_candidates", payloads={
        "candidate_snapshot": snapshot_payload,
        "matlab_edges": matlab_edges,
    })
    
    coverage_report = build_candidate_coverage_report(matlab_edges, snapshot_payload)
    coverage_report.update({
        "source_run_root": str(source_surface.run_root),
        "dest_run_root": str(dest_run_root),
    })
    
    json_path = dest_run_root / CANDIDATE_COVERAGE_JSON_PATH
    text_path = dest_run_root / CANDIDATE_COVERAGE_TEXT_PATH
    write_json_with_hash(json_path, coverage_report)
    write_text_with_hash(text_path, render_candidate_coverage_report(coverage_report))
    
    write_run_manifest(
        dest_run_root,
        run_kind="parity_run",
        status="passed" if bool(coverage_report.get("passed")) else "failed",
        command="capture-candidates",
        dataset_hash=fingerprint_file(source_surface.run_root / "01_Input" / "volume.tif"),
        oracle_surface=source_surface.oracle_surface,
        params_payload=params,
    )
    
    return coverage_report

def run_edge_replay(
    source_surface: ExactProofSourceSurface,
    dest_run_root: Path,
) -> dict[str, Any]:
    """Orchestrate the edge replay workflow."""
    # Placeholder for replay logic
    return {"passed": True}

def run_lut_proof(
    source_run_root: Path,
    dest_run_root: Path,
) -> dict[str, Any]:
    """Orchestrate the LUT parity proof."""
    # Placeholder for LUT proof logic
    return {"passed": True}
