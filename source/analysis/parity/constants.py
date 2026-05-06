"""Constants for native-first MATLAB-oracle parity experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from source.io.matlab_exact_proof import EXACT_STAGE_ORDER

# Directory structure constants
ANALYSIS_DIR = Path("03_Analysis")
ANALYSIS_TABLES_DIR = ANALYSIS_DIR / "tables"
CHECKPOINTS_DIR = Path("02_Output") / "python_results" / "checkpoints"
EXPERIMENT_REFS_DIR = Path("00_Refs")
EXPERIMENT_PARAMS_DIR = Path("01_Params")
HASHES_DIR = ANALYSIS_DIR / "hashes"
NORMALIZED_DIR = ANALYSIS_DIR / "normalized"
METADATA_DIR = Path("99_Metadata")

# Manifest and index paths
RUN_MANIFEST_PATH = METADATA_DIR / "run_manifest.json"
ORACLE_MANIFEST_PATH = METADATA_DIR / "oracle_manifest.json"
REPORT_MANIFEST_PATH = METADATA_DIR / "report_manifest.json"
DATASET_MANIFEST_PATH = METADATA_DIR / "dataset_manifest.json"
EXPERIMENT_INDEX_PATH = Path("index.jsonl")
EXPERIMENT_ROOT_SUBDIRS = ("datasets", "oracles", "reports", "runs")

# Discovery and input constants
ORACLE_DISCOVERY_STAGES = ("energy", *EXACT_STAGE_ORDER)
DATASET_INPUT_DIR = Path("01_Input")

# Report and proof paths
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

# Parameter paths
SHARED_PARAMS_PATH = EXPERIMENT_PARAMS_DIR / "shared_params.json"
PYTHON_DERIVED_PARAMS_PATH = EXPERIMENT_PARAMS_DIR / "python_derived_params.json"
PARAM_DIFF_PATH = EXPERIMENT_PARAMS_DIR / "param_diff.json"

# Operational constants
HEARTBEAT_INTERVAL_ITERATIONS = 512
DEFAULT_MEMORY_SAFETY_FRACTION = 0.8
EDGE_CANDIDATE_AUDIT_PATH = (
    Path("02_Output") / "python_results" / "stages" / "edges" / "candidate_audit.json"
)

# Parameter validation keys
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
        "comparison_exact_network_use_conflict_painting",
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
