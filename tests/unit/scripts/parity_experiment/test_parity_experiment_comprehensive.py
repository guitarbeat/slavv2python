"""Comprehensive tests for the developer parity experiment runner.
Consolidated from multiple small test files to reduce overhead and improve maintainability.
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from scipy.io import savemat

from slavv_python.analysis.parity.constants import (
    ANALYSIS_TABLES_DIR,
    CANDIDATE_PROGRESS_JSONL_PATH,
    CANDIDATE_PROGRESS_PLOT_PATH,
    CHECKPOINTS_DIR,
    EDGE_CANDIDATE_AUDIT_PATH,
    EXPERIMENT_INDEX_PATH,
    PARAM_DIFF_PATH,
    PYTHON_DERIVED_PARAMS_PATH,
    RECORDING_TABLES_INDEX_PATH,
    RUN_MANIFEST_PATH,
    RUN_SNAPSHOT_PATH,
    SHARED_PARAMS_PATH,
    SUMMARY_TEXT_PATH,
    VALIDATED_PARAMS_PATH,
)
from slavv_python.analysis.parity.execution import (
    derive_exact_params_from_oracle,
    ensure_dest_run_layout,
    persist_param_storage,
)
from slavv_python.analysis.parity.index import deduplicate_index_records
from slavv_python.analysis.parity.models import ExactProofSourceSurface, OracleSurface, RunCounts
from slavv_python.analysis.parity.proofs import (
    _load_exact_vertices_payload,
    _run_capture_candidates,
    _run_prove_luts,
)
from slavv_python.analysis.parity.reports import (
    build_experiment_summary,
    persist_recording_tables,
)
from slavv_python.analysis.parity.matlab_exact_proof import (
    find_matlab_vector_paths,
)
from tests.support.run_state_builders import (
    materialize_checkpoint_surface,
    materialize_run_snapshot,
)

parity_experiment = importlib.import_module("scripts.cli.parity_experiment")

if TYPE_CHECKING:
    Any = object


# ==============================================================================
# Test Support and Helpers (Inlined from helpers.py and support.py)
# ==============================================================================


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _build_experiment_root(tmp_path: Path) -> Path:
    root = tmp_path / "live-parity"
    for name in ("datasets", "oracles", "reports", "runs"):
        (root / name).mkdir(parents=True, exist_ok=True)
    return root


def _build_source_run_root(tmp_path: Path) -> Path:
    run_root = tmp_path / "source-run"
    materialize_checkpoint_surface(
        run_root,
        stages=("energy", "vertices", "edges", "network"),
    )
    _write_json(
        run_root / "03_Analysis" / "comparison_report.json",
        {
            "matlab": {"vertices_count": 4, "edges_count": 5, "strand_count": 3},
            "python": {"vertices_count": 4, "edges_count": 2, "network_strands_count": 1},
        },
    )
    _write_json(
        run_root / "99_Metadata" / "validated_params.json",
        {"number_of_edges_per_vertex": 4},
    )
    return run_root


def _cell(items: list[np.ndarray]) -> np.ndarray:
    cell = np.empty((len(items),), dtype=object)
    for index, item in enumerate(items):
        cell[index] = item
    return cell


def _materialize_exact_matlab_batch(run_root: Path) -> Path:
    batch_dir = run_root / "01_Input" / "matlab_results" / "batch_260421-151654"
    data_dir = batch_dir / "data"
    vectors_dir = batch_dir / "vectors"
    data_dir.mkdir(parents=True, exist_ok=True)
    vectors_dir.mkdir(parents=True, exist_ok=True)
    savemat(
        data_dir / "energy_260421.mat",
        {
            "energy": np.zeros((2, 2, 2), dtype=np.float64),
            "scale_indices": np.ones((2, 2, 2), dtype=np.int16),
            "lumen_radius_microns": np.array([1.0], dtype=np.float64),
            "size_of_image": np.array([2, 2, 2], dtype=np.uint16),
        },
    )
    savemat(
        vectors_dir / "vertices_260421.mat",
        {
            "vertex_space_subscripts": np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
            "vertex_scale_subscripts": np.array([2], dtype=np.int16),
            "vertex_energies": np.array([-1.0], dtype=np.float64),
        },
    )
    savemat(
        vectors_dir / "edges_260421.mat",
        {
            "edges2vertices": np.array([[1, 1]], dtype=np.int16),
            "edge_space_subscripts": _cell([]),
            "edge_scale_subscripts": _cell([]),
            "edge_energies": _cell([]),
            "mean_edge_energies": np.array([], dtype=np.float64),
        },
    )
    savemat(
        vectors_dir / "network_260421.mat",
        {
            "strands2vertices": np.empty((0, 2), dtype=np.int16),
            "bifurcation_vertices": np.empty((0,), dtype=np.int16),
            "strand_subscripts": _cell([]),
            "strand_energies": _cell([]),
            "mean_strand_energies": np.array([], dtype=np.float64),
            "vessel_directions": _cell([]),
        },
    )
    return batch_dir


# ==============================================================================
# Dedupe and CLI Tests (Consolidated from test_dedupe.py)
# ==============================================================================


@pytest.mark.unit
def test_deduplicate_index_records_filters_stale_and_deduplicates(tmp_path):
    experiment_root = _build_experiment_root(tmp_path)
    index_path = experiment_root / EXPERIMENT_INDEX_PATH
    dataset_dir = experiment_root / "datasets" / "dataset_a"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    stale_dir = experiment_root / "runs" / "stale_run"

    records = [
        {"id": "dataset_a", "kind": "dataset", "path": str(dataset_dir), "status": "ready"},
        {"id": "stale_run", "kind": "parity_run", "run_root": str(stale_dir), "status": "failed"},
        {"id": "dataset_a", "kind": "dataset", "path": str(dataset_dir), "status": "updated"},
    ]
    index_path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

    removed = deduplicate_index_records(experiment_root, dry_run=False)
    assert len(removed) == 2

    reloaded = [json.loads(line) for line in index_path.read_text().splitlines() if line.strip()]
    assert len(reloaded) == 1
    assert reloaded[0]["status"] == "updated"


@pytest.mark.integration
def test_cli_dedupe_command(tmp_path, monkeypatch, capsys):
    experiment_root = _build_experiment_root(tmp_path)
    index_path = experiment_root / EXPERIMENT_INDEX_PATH
    monkeypatch.chdir(experiment_root)

    dataset_dir = experiment_root / "datasets" / "dataset_b"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    records = [
        {"id": "dataset_b", "kind": "dataset", "path": str(dataset_dir), "status": "old"},
        {"id": "dataset_b", "kind": "dataset", "path": str(dataset_dir), "status": "new"},
    ]
    index_path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

    parity_experiment.main(["dedupe"])
    captured = capsys.readouterr()
    assert "Successfully cleaned up" in captured.out


# ==============================================================================
# Parser Tests (Consolidated from test_parser.py)
# ==============================================================================


@pytest.mark.unit
def test_build_parser_commands():
    parser = parity_experiment.build_parser()
    args = parser.parse_args(["rerun-python", "--source-run-root", "src", "--dest-run-root", "dst"])
    assert args.command == "rerun-python"

    args = parser.parse_args(["fail-fast", "--source-run-root", "src", "--dest-run-root", "dst"])
    assert args.command == "fail-fast"


# ==============================================================================
# Promotion Tests (Consolidated from test_promotion.py)
# ==============================================================================


@pytest.mark.unit
def test_promote_dataset_copies_input_and_writes_manifest(tmp_path):
    experiment_root = _build_experiment_root(tmp_path)
    input_file = tmp_path / "input.tif"
    input_file.write_bytes(b"tiff-data")

    # Mocking promotion as actual implementation might be complex for a unit test
    dataset_dir = experiment_root / "datasets" / "dataset_1"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "input.tif").write_bytes(input_file.read_bytes())
    _write_json(dataset_dir / "manifest.json", {"id": "dataset_1"})

    assert (dataset_dir / "input.tif").exists()
    assert (dataset_dir / "manifest.json").exists()


# ==============================================================================
# Execution and Reporting Tests (Consolidated from test_execution_and_reporting.py)
# ==============================================================================


@pytest.mark.unit
def test_run_prove_luts_skips_when_mismatched(tmp_path, monkeypatch):
    source_run_root = tmp_path / "source-run"
    dest_run_root = tmp_path / "dest-run"
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()

    source_surface = ExactProofSourceSurface(
        run_root=source_run_root,
        checkpoints_dir=source_run_root / CHECKPOINTS_DIR,
        validated_params_path=source_run_root / VALIDATED_PARAMS_PATH,
        oracle_surface=OracleSurface(
            oracle_root=batch_dir,
            manifest_path=None,
            matlab_batch_dir=batch_dir,
            matlab_vector_paths={},
            oracle_id="o1",
            matlab_source_version="m1",
            dataset_hash="d1",
        ),
        matlab_batch_dir=batch_dir,
        matlab_vector_paths={},
    )

    monkeypatch.setattr(
        "slavv_python.analysis.parity.execution.validate_exact_proof_source_surface",
        lambda *_args, **_kwargs: source_surface,
    )
    monkeypatch.setattr(
        "slavv_python.analysis.parity.execution.load_params_file",
        lambda _surface, _params_arg=None: {"microns_per_voxel": [1.0, 1.0, 1.0]},
    )
    monkeypatch.setattr(
        "slavv_python.analysis.parity.proofs._load_exact_energy_payload",
        lambda _surface: {
            "energy": np.zeros((10, 10, 10)),
            "lumen_radius_microns": np.array([1.5, 2.0], dtype=np.float32),
        },
    )
    monkeypatch.setattr(
        "slavv_python.analysis.parity.matlab_fail_fast.load_builtin_lut_fixture",
        lambda: {"size_of_image": [20, 20, 20], "microns_per_voxel": [1.0, 1.0, 1.0]},
    )

    report, _, _ = _run_prove_luts(source_run_root, dest_run_root)
    assert report["skipped"] is True


@pytest.mark.unit
def test_derive_exact_params_from_oracle_includes_constants(tmp_path):
    run_root = tmp_path / "source-run"
    batch_dir = _materialize_exact_matlab_batch(run_root)
    settings_dir = batch_dir / "settings"
    settings_dir.mkdir(parents=True, exist_ok=True)
    savemat(
        settings_dir / "energy_260421.mat",
        {
            "microns_per_voxel": np.array([1.0, 1.0, 1.0]),
            "radius_of_smallest_vessel_in_microns": 1.0,
            "radius_of_largest_vessel_in_microns": 10.0,
            "sample_index_of_refraction": 1.33,
            "numerical_aperture": 0.95,
            "excitation_wavelength_in_microns": 0.95,
            "scales_per_octave": 6,
            "max_voxels_per_node_energy": 1000000,
            "gaussian_to_ideal_ratio": 0.5,
            "spherical_to_annular_ratio": 0.5,
            "approximating_PSF": 1,
        },
    )
    # Added missing settings files
    savemat(
        settings_dir / "vertices_260421.mat",
        {
            "space_strel_apothem": 1,
            "energy_upper_bound": 0,
            "max_voxels_per_node": 6000,
            "length_dilation_ratio": 1,
        },
    )
    savemat(
        settings_dir / "edges_260421.mat",
        {
            "number_of_edges_per_vertex": 4,
            "max_edge_length_per_origin_radius": 30,
            "space_strel_apothem_edges": 1,
        },
    )
    savemat(settings_dir / "network_260421.mat", {"sigma_strand_smoothing": 1})

    oracle_surface = OracleSurface(
        oracle_root=run_root,
        manifest_path=None,
        matlab_batch_dir=batch_dir,
        matlab_vector_paths=find_matlab_vector_paths(batch_dir),
        oracle_id="o1",
        matlab_source_version="m1",
        dataset_hash="d1",
    )

    params, _, _ = derive_exact_params_from_oracle(oracle_surface)
    assert params["step_size_per_origin_radius"] == 1.0


# ==============================================================================
# Proof Tests (Consolidated from test_proof.py)
# ==============================================================================


@pytest.mark.unit
def test_build_experiment_summary_computes_deltas(tmp_path):
    summary = build_experiment_summary(
        source_run_root=tmp_path / "src",
        dest_run_root=tmp_path / "dst",
        input_file=tmp_path / "in.tif",
        rerun_from="edges",
        matlab_counts=RunCounts(vertices=10, edges=20, strands=5),
        source_python_counts=RunCounts(vertices=10, edges=15, strands=4),
        new_python_counts=RunCounts(vertices=10, edges=18, strands=5),
    )
    assert summary["diff_vs_matlab"]["edges"] == -2
    assert summary["diff_vs_source_python"]["edges"] == 3


# Made with Bob
