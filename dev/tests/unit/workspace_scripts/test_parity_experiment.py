"""Unit tests for the developer parity experiment runner."""

from __future__ import annotations

import importlib
import json
from typing import TYPE_CHECKING

import numpy as np
import pytest
from dev.tests.support.run_state_builders import (
    materialize_checkpoint_surface,
    materialize_run_snapshot,
)
from scipy.io import savemat

parity_experiment = importlib.import_module("dev.scripts.cli.parity_experiment")

if TYPE_CHECKING:
    from pathlib import Path


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _build_source_run_root(tmp_path: Path) -> Path:
    run_root = tmp_path / "source-run"
    materialize_checkpoint_surface(
        run_root,
        stages=("energy", "vertices", "edges", "network"),
    )
    _write_json(
        run_root / "03_Analysis" / "comparison_report.json",
        {
            "matlab": {
                "vertices_count": 4,
                "edges_count": 5,
                "strand_count": 3,
            },
            "python": {
                "vertices_count": 4,
                "edges_count": 2,
                "network_strands_count": 1,
            },
            "vertices": {
                "matlab_count": 4,
                "python_count": 4,
            },
            "edges": {
                "matlab_count": 5,
                "python_count": 2,
            },
            "network": {
                "matlab_strand_count": 3,
                "python_strand_count": 1,
            },
        },
    )
    _write_json(
        run_root / "99_Metadata" / "validated_params.json",
        {"number_of_edges_per_vertex": 4},
    )
    return run_root


def _materialize_exact_matlab_batch(run_root: Path) -> Path:
    batch_dir = run_root / "01_Input" / "matlab_results" / "batch_260421-151654"
    vectors_dir = batch_dir / "vectors"
    vectors_dir.mkdir(parents=True, exist_ok=True)
    savemat(
        vectors_dir / "vertices_260421.mat",
        {
            "vertex_space_subscripts": [[1.0, 2.0, 3.0]],
            "vertex_scale_subscripts": [2],
            "vertex_energies": [-1.0],
        },
    )
    savemat(
        vectors_dir / "edges_260421.mat",
        {
            "edges2vertices": [[1, 1]],
            "edge_space_subscripts": [],
            "edge_scale_subscripts": [],
            "edge_energies": [],
            "mean_edge_energies": [],
        },
    )
    savemat(
        vectors_dir / "network_260421.mat",
        {
            "strands2vertices": [],
            "bifurcation_vertices": [],
            "strand_subscripts": [],
            "strand_energies": [],
            "mean_strand_energies": [],
            "vessel_directions": [],
        },
    )
    return batch_dir


def test_build_parser_rerun_python_defaults():
    parser = parity_experiment.build_parser()

    args = parser.parse_args(
        [
            "rerun-python",
            "--source-run-root",
            "source-run",
            "--dest-run-root",
            "dest-run",
        ]
    )

    assert args.command == "rerun-python"
    assert args.rerun_from == "edges"
    assert args.params_file is None
    assert args.input is None


def test_build_parser_prove_exact_defaults():
    parser = parity_experiment.build_parser()

    args = parser.parse_args(
        [
            "prove-exact",
            "--source-run-root",
            "source-run",
            "--dest-run-root",
            "dest-run",
        ]
    )

    assert args.command == "prove-exact"
    assert args.stage == "all"
    assert args.report_path is None


def test_build_parser_fail_fast_defaults():
    parser = parity_experiment.build_parser()

    args = parser.parse_args(
        [
            "fail-fast",
            "--source-run-root",
            "source-run",
            "--dest-run-root",
            "dest-run",
        ]
    )

    assert args.command == "fail-fast"
    assert args.force is False
    assert args.debug_maps is False


def test_validate_source_run_surface_accepts_required_artifacts(tmp_path):
    run_root = _build_source_run_root(tmp_path)

    surface = parity_experiment.validate_source_run_surface(run_root)

    assert surface.run_root == run_root.resolve()
    assert surface.checkpoints_dir == run_root.resolve() / parity_experiment.CHECKPOINTS_DIR
    assert surface.comparison_report_path.is_file()
    assert surface.validated_params_path.is_file()
    assert surface.run_snapshot_path is None


def test_validate_source_run_surface_reports_missing_artifacts(tmp_path):
    run_root = tmp_path / "source-run"
    run_root.mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValueError, match="missing required artifacts"):
        parity_experiment.validate_source_run_surface(run_root)


def test_resolve_input_file_uses_run_snapshot_provenance(tmp_path):
    repo_root = tmp_path / "repo"
    input_file = repo_root / "data" / "slavv_test_volume.tif"
    input_file.parent.mkdir(parents=True, exist_ok=True)
    input_file.write_bytes(b"tiff")

    run_root = _build_source_run_root(tmp_path)
    materialize_run_snapshot(
        run_root,
        {
            "run_id": "run-1",
            "provenance": {
                "input_file": "data/slavv_test_volume.tif",
            },
        },
    )
    surface = parity_experiment.validate_source_run_surface(run_root)

    resolved = parity_experiment.resolve_input_file(
        surface,
        None,
        repo_root=repo_root,
    )

    assert resolved == input_file.resolve()


def test_resolve_input_file_requires_snapshot_or_explicit_input(tmp_path):
    run_root = _build_source_run_root(tmp_path)
    surface = parity_experiment.validate_source_run_surface(run_root)

    with pytest.raises(ValueError, match="no --input was provided"):
        parity_experiment.resolve_input_file(surface, None)


def test_load_params_file_uses_source_default_and_override(tmp_path):
    run_root = _build_source_run_root(tmp_path)
    surface = parity_experiment.validate_source_run_surface(run_root)
    override = _write_json(tmp_path / "override_params.json", {"edge_method": "tracing"})

    default_params = parity_experiment.load_params_file(surface, None)
    override_params = parity_experiment.load_params_file(surface, str(override))

    assert default_params == {"number_of_edges_per_vertex": 4}
    assert override_params == {"edge_method": "tracing"}


def test_validate_exact_proof_source_surface_accepts_required_artifacts(tmp_path):
    run_root = _build_source_run_root(tmp_path)
    _materialize_exact_matlab_batch(run_root)
    _write_json(
        run_root / "99_Metadata" / "validated_params.json",
        {"comparison_exact_network": True},
    )
    materialize_checkpoint_surface(
        run_root,
        stages=("energy",),
        payloads={"energy": {"energy_origin": "matlab_batch_hdf5"}},
    )

    surface = parity_experiment.validate_exact_proof_source_surface(run_root)

    assert surface.run_root == run_root.resolve()
    assert surface.matlab_batch_dir.name == "batch_260421-151654"
    assert set(surface.matlab_vector_paths) == {"vertices", "edges", "network"}


def test_validate_exact_proof_source_surface_requires_exact_route_gate(tmp_path):
    run_root = _build_source_run_root(tmp_path)
    _materialize_exact_matlab_batch(run_root)
    materialize_checkpoint_surface(
        run_root,
        stages=("energy",),
        payloads={"energy": {"energy_origin": "matlab_batch_hdf5"}},
    )

    with pytest.raises(ValueError, match="comparison_exact_network"):
        parity_experiment.validate_exact_proof_source_surface(run_root)


def test_validate_exact_proof_source_surface_requires_matlab_batch_hdf5(tmp_path):
    run_root = _build_source_run_root(tmp_path)
    _materialize_exact_matlab_batch(run_root)
    _write_json(
        run_root / "99_Metadata" / "validated_params.json",
        {"comparison_exact_network": True},
    )
    materialize_checkpoint_surface(
        run_root,
        stages=("energy",),
        payloads={"energy": {"energy_origin": "python_native"}},
    )

    with pytest.raises(ValueError, match="matlab_batch_hdf5"):
        parity_experiment.validate_exact_proof_source_surface(run_root)


def test_build_exact_preflight_report_refuses_when_memory_budget_is_too_large(
    tmp_path,
    monkeypatch,
):
    run_root = _build_source_run_root(tmp_path)
    _materialize_exact_matlab_batch(run_root)
    _write_json(
        run_root / "99_Metadata" / "validated_params.json",
        {
            "comparison_exact_network": True,
            "microns_per_voxel": [1.0, 1.0, 1.0],
        },
    )
    materialize_checkpoint_surface(
        run_root,
        stages=("energy",),
        payloads={
            "energy": {
                "energy_origin": "matlab_batch_hdf5",
                "energy": np.zeros((10, 10, 10), dtype=np.float32),
                "lumen_radius_microns": np.array([1.0], dtype=np.float32),
            }
        },
    )
    monkeypatch.setattr(parity_experiment, "find_parity_process_collisions", lambda _path: [])

    class _FakeMemory:
        available = 64

    monkeypatch.setattr(parity_experiment.psutil, "virtual_memory", lambda: _FakeMemory())

    report = parity_experiment.build_exact_preflight_report(
        run_root,
        tmp_path / "dest-run",
        memory_safety_fraction=0.8,
        force=False,
    )

    assert report["passed"] is False
    assert report["collision_count"] == 0


def test_build_exact_preflight_report_refuses_on_destination_collision(
    tmp_path,
    monkeypatch,
):
    run_root = _build_source_run_root(tmp_path)
    _materialize_exact_matlab_batch(run_root)
    _write_json(
        run_root / "99_Metadata" / "validated_params.json",
        {
            "comparison_exact_network": True,
            "microns_per_voxel": [1.0, 1.0, 1.0],
        },
    )
    materialize_checkpoint_surface(
        run_root,
        stages=("energy",),
        payloads={
            "energy": {
                "energy_origin": "matlab_batch_hdf5",
                "energy": np.zeros((2, 2, 2), dtype=np.float32),
                "lumen_radius_microns": np.array([1.0], dtype=np.float32),
            }
        },
    )

    monkeypatch.setattr(
        parity_experiment,
        "find_parity_process_collisions",
        lambda _path: [
            {"pid": 1234, "name": "python", "cmdline": ["python", "parity_experiment.py"]}
        ],
    )

    class _FakeMemory:
        available = 2_000_000_000

    monkeypatch.setattr(parity_experiment.psutil, "virtual_memory", lambda: _FakeMemory())

    report = parity_experiment.build_exact_preflight_report(
        run_root,
        tmp_path / "dest-run",
        memory_safety_fraction=0.8,
        force=False,
    )

    assert report["passed"] is False
    assert report["collision_count"] == 1


def test_capture_candidates_persists_heartbeat_detail(tmp_path, monkeypatch):
    source_run_root = tmp_path / "source-run"
    dest_run_root = tmp_path / "dest-run"
    matlab_batch_dir = tmp_path / "matlab-batch"
    matlab_batch_dir.mkdir()
    source_surface = parity_experiment.ExactProofSourceSurface(
        run_root=source_run_root,
        checkpoints_dir=source_run_root / parity_experiment.CHECKPOINTS_DIR,
        validated_params_path=source_run_root / parity_experiment.VALIDATED_PARAMS_PATH,
        matlab_batch_dir=matlab_batch_dir,
        matlab_vector_paths={},
    )
    candidates = {
        "traces": [np.array([[0.0, 0.0, 0.0]], dtype=np.float32)],
        "connections": np.array([[0, 0]], dtype=np.int32),
        "metrics": np.array([-1.0], dtype=np.float32),
        "energy_traces": [np.array([-1.0], dtype=np.float32)],
        "scale_traces": [np.array([0], dtype=np.int16)],
        "origin_indices": np.array([0], dtype=np.int32),
        "connection_sources": ["global_watershed"],
        "diagnostics": {},
    }

    def fake_generate(*_args, **kwargs):
        heartbeat = kwargs.get("heartbeat")
        if callable(heartbeat):
            heartbeat(512, 1)
        return candidates

    monkeypatch.setattr(
        parity_experiment,
        "validate_exact_proof_source_surface",
        lambda _run_root: source_surface,
    )
    monkeypatch.setattr(
        parity_experiment,
        "load_exact_params_file",
        lambda _surface: {"comparison_exact_network": True},
    )
    monkeypatch.setattr(
        parity_experiment,
        "_load_exact_energy_payload",
        lambda _surface: {
            "energy": np.zeros((2, 2, 2), dtype=np.float32),
            "scale_indices": np.zeros((2, 2, 2), dtype=np.int16),
            "lumen_radius_microns": np.array([1.0], dtype=np.float32),
            "energy_sign": -1.0,
        },
    )
    monkeypatch.setattr(
        parity_experiment,
        "_load_exact_vertices_payload",
        lambda _surface: {
            "positions": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            "scales": np.array([0], dtype=np.int16),
        },
    )
    monkeypatch.setattr(
        parity_experiment,
        "_generate_edge_candidates_matlab_frontier",
        fake_generate,
    )
    monkeypatch.setattr(
        parity_experiment,
        "_finalize_matlab_parity_candidates",
        lambda candidate_payload, *_args: candidate_payload,
    )
    monkeypatch.setattr(
        parity_experiment,
        "load_normalized_matlab_vectors",
        lambda *_args, **_kwargs: {
            "edges": {
                "connections": np.array([[0, 0]], dtype=np.int32),
            }
        },
    )
    monkeypatch.setattr(
        parity_experiment,
        "build_candidate_coverage_report",
        lambda *_args, **_kwargs: {"candidate_surface": {}},
    )

    parity_experiment._run_capture_candidates(
        source_run_root=source_run_root,
        dest_run_root=dest_run_root,
        include_debug_maps=False,
    )

    snapshot = json.loads(
        (dest_run_root / parity_experiment.RUN_SNAPSHOT_PATH).read_text(encoding="utf-8")
    )
    assert snapshot["current_stage"] == "edges"
    assert snapshot["current_detail"] == (
        "Generating edge candidates through MATLAB-style frontier workflow "
        "(iterations=512, candidates=1)"
    )
    assert snapshot["stages"]["edges"]["detail"] == snapshot["current_detail"]


def test_build_experiment_summary_computes_deltas(tmp_path):
    source_run_root = tmp_path / "source-run"
    dest_run_root = tmp_path / "dest-run"
    input_file = tmp_path / "input.tif"
    input_file.write_bytes(b"tiff")

    summary = parity_experiment.build_experiment_summary(
        source_run_root=source_run_root,
        dest_run_root=dest_run_root,
        input_file=input_file,
        rerun_from="edges",
        matlab_counts=parity_experiment.RunCounts(vertices=4, edges=5, strands=3),
        source_python_counts=parity_experiment.RunCounts(vertices=4, edges=2, strands=1),
        new_python_counts=parity_experiment.RunCounts(vertices=4, edges=3, strands=2),
    )

    assert summary["diff_vs_matlab"] == {"vertices": 0, "edges": -2, "strands": -1}
    assert summary["diff_vs_source_python"] == {"vertices": 0, "edges": 1, "strands": 1}


def test_summarize_command_prints_saved_summary(capsys, tmp_path):
    run_root = tmp_path / "dest-run"
    summary_path = run_root / parity_experiment.SUMMARY_TEXT_PATH
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("example summary", encoding="utf-8")

    parity_experiment.main(["summarize", "--run-root", str(run_root)])

    captured = capsys.readouterr()
    assert "example summary" in captured.out
