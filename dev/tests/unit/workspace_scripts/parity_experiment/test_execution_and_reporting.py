"Unit tests for the developer parity experiment runner execution and reporting."

from __future__ import annotations

import importlib
import json

import numpy as np
import pytest
from dev.tests.support.run_state_builders import (
    materialize_checkpoint_surface,
    materialize_run_snapshot,
)
from scipy.io import savemat

from .helpers import (
    _build_experiment_root,
    _build_source_run_root,
    _materialize_exact_matlab_batch,
    _write_json,
)

parity_experiment = importlib.import_module("dev.scripts.cli.parity_experiment")


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
                "energy_origin": "python_native_hessian",
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
        oracle_root=None,
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
                "energy_origin": "python_native_hessian",
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
        oracle_root=None,
        memory_safety_fraction=0.8,
        force=False,
    )

    assert report["passed"] is False
    assert report["collision_count"] == 1


def test_build_exact_preflight_report_flags_unfair_exact_params(tmp_path, monkeypatch):
    run_root = _build_source_run_root(tmp_path)
    _materialize_exact_matlab_batch(run_root)
    _write_json(
        run_root / "99_Metadata" / "validated_params.json",
        {
            "comparison_exact_network": True,
            "direction_method": "hessian",
            "discrete_tracing": False,
            "edge_method": "tracing",
            "energy_method": "hessian",
            "energy_projection_mode": "matlab",
            "parity_candidate_salvage_mode": "auto",
        },
    )
    materialize_checkpoint_surface(
        run_root,
        stages=("energy",),
        payloads={
            "energy": {
                "energy_origin": "python_native_hessian",
                "energy": np.zeros((2, 2, 2), dtype=np.float32),
                "lumen_radius_microns": np.array([1.0], dtype=np.float32),
            }
        },
    )
    monkeypatch.setattr(parity_experiment, "find_parity_process_collisions", lambda _path: [])

    class _FakeMemory:
        available = 2_000_000_000

    monkeypatch.setattr(parity_experiment.psutil, "virtual_memory", lambda: _FakeMemory())

    report = parity_experiment.build_exact_preflight_report(
        run_root,
        tmp_path / "dest-run",
        oracle_root=None,
        memory_safety_fraction=0.8,
        force=False,
    )

    assert report["passed"] is False
    assert report["params_audit"]["passed"] is False
    assert report["params_audit"]["disallowed_python_only_keys"] == [
        "parity_candidate_salvage_mode"
    ]


def test_find_parity_process_collisions_ignores_current_process_ancestry(monkeypatch, tmp_path):
    dest_run_root = tmp_path / "dest-run"
    dest_run_root.mkdir()

    class _FakeProcess:
        def __init__(self, pid: int, cmdline: list[str] | None = None) -> None:
            self.pid = pid
            self.info = {
                "pid": pid,
                "name": "python.exe",
                "cmdline": cmdline or [],
            }

        def parents(self) -> list[object]:
            return [_FakeProcess(22), _FakeProcess(11)]

    normalized_dest = str(dest_run_root.resolve())
    matching_cmdline = [
        "python.exe",
        "dev/scripts/cli/parity_experiment.py",
        "fail-fast",
        "--dest-run-root",
        normalized_dest,
    ]
    monkeypatch.setattr(parity_experiment.psutil, "Process", lambda: _FakeProcess(33))
    monkeypatch.setattr(
        parity_experiment.psutil,
        "process_iter",
        lambda _attrs: iter(
            [
                _FakeProcess(33, matching_cmdline),
                _FakeProcess(22, matching_cmdline),
                _FakeProcess(11, matching_cmdline),
                _FakeProcess(44, matching_cmdline),
            ]
        ),
    )

    collisions = parity_experiment.find_parity_process_collisions(dest_run_root)

    assert collisions == [
        {
            "pid": 44,
            "name": "python.exe",
            "cmdline": matching_cmdline,
        }
    ]


def test_run_prove_luts_skips_when_builtin_fixture_inputs_do_not_match_source_run(
    tmp_path, monkeypatch
):
    source_run_root = tmp_path / "source-run"
    dest_run_root = tmp_path / "dest-run"
    matlab_batch_dir = tmp_path / "matlab-batch"
    matlab_batch_dir.mkdir()
    source_surface = parity_experiment.ExactProofSourceSurface(
        run_root=source_run_root,
        checkpoints_dir=source_run_root / parity_experiment.CHECKPOINTS_DIR,
        validated_params_path=source_run_root / parity_experiment.VALIDATED_PARAMS_PATH,
        oracle_surface=parity_experiment.OracleSurface(
            oracle_root=matlab_batch_dir,
            manifest_path=None,
            matlab_batch_dir=matlab_batch_dir,
            matlab_vector_paths={},
            oracle_id="oracle-a",
            matlab_source_version="matlab-a",
            dataset_hash="dataset-a",
        ),
        matlab_batch_dir=matlab_batch_dir,
        matlab_vector_paths={},
    )

    monkeypatch.setattr(
        parity_experiment,
        "validate_exact_proof_source_surface",
        lambda *_args, **_kwargs: source_surface,
    )
    monkeypatch.setattr(
        parity_experiment,
        "load_exact_params_file",
        lambda _surface: {"microns_per_voxel": [0.916, 0.916, 1.99688]},
    )
    monkeypatch.setattr(
        parity_experiment,
        "_load_exact_energy_payload",
        lambda _surface: {
            "energy": np.zeros((64, 512, 512), dtype=np.float32),
            "lumen_radius_microns": np.array([1.5, 2.0], dtype=np.float32),
        },
    )
    monkeypatch.setattr(
        parity_experiment,
        "load_builtin_lut_fixture",
        lambda: {
            "size_of_image": [121, 512, 512],
            "microns_per_voxel": [1.0, 1.0, 1.0],
            "lumen_radius_microns": [1.0, 2.0],
            "scales": {"0": {}},
        },
    )

    report, _json_path, _text_path = parity_experiment._run_prove_luts(
        source_run_root=source_run_root,
        dest_run_root=dest_run_root,
        oracle_root=None,
    )

    assert report["passed"] is True
    assert report["skipped"] is True
    assert report["skip_reason"] == "builtin LUT fixture inputs do not match the source exact run"
    assert report["source_inputs"]["size_of_image"] == [64, 512, 512]
    assert report["fixture_inputs"]["size_of_image"] == [121, 512, 512]


def test_load_exact_vertices_payload_uses_curated_vertex_surface(tmp_path):
    source_run_root = tmp_path / "source-run"
    materialize_checkpoint_surface(source_run_root, stages=("vertices",))

    batch_dir = tmp_path / "batch"
    vectors_dir = batch_dir / "vectors"
    vectors_dir.mkdir(parents=True, exist_ok=True)
    savemat(
        vectors_dir / "curated_vertices_1.mat",
        {
            "vertex_space_subscripts": np.array([[4.0, 5.0, 6.0]], dtype=np.float64),
            "vertex_scale_subscripts": np.array([3.0], dtype=np.float64),
            "vertex_energies": np.array([-9.0], dtype=np.float64),
        },
    )
    savemat(
        vectors_dir / "edges_1.mat",
        {
            "edges2vertices": np.empty((0, 2), dtype=np.int16),
            "edge_space_subscripts": np.empty((0,), dtype=object),
            "edge_scale_subscripts": np.empty((0,), dtype=object),
            "edge_energies": np.empty((0,), dtype=object),
            "mean_edge_energies": np.empty((0,), dtype=np.float64),
            "vertex_space_subscripts": np.array([[100.0, 200.0, 10.0]], dtype=np.float64),
            "vertex_scale_subscripts": np.array([9.0], dtype=np.float64),
            "vertex_energies": np.array([-1.0], dtype=np.float64),
        },
    )

    source_surface = parity_experiment.ExactProofSourceSurface(
        run_root=source_run_root,
        checkpoints_dir=source_run_root / parity_experiment.CHECKPOINTS_DIR,
        validated_params_path=source_run_root / parity_experiment.VALIDATED_PARAMS_PATH,
        oracle_surface=parity_experiment.OracleSurface(
            oracle_root=batch_dir,
            manifest_path=None,
            matlab_batch_dir=batch_dir,
            matlab_vector_paths={},
            oracle_id=None,
            matlab_source_version=None,
            dataset_hash=None,
        ),
        matlab_batch_dir=batch_dir,
        matlab_vector_paths={},
    )

    payload = parity_experiment._load_exact_vertices_payload(source_surface)

    np.testing.assert_array_equal(
        payload["positions"],
        np.array([[3.0, 4.0, 5.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(payload["scales"], np.array([2], dtype=np.int16))
    np.testing.assert_array_equal(payload["energies"], np.array([-9.0], dtype=np.float32))
    assert payload["count"] == 1


def test_capture_candidates_persists_heartbeat_detail(tmp_path, monkeypatch):
    source_run_root = tmp_path / "source-run"
    dest_run_root = tmp_path / "dest-run"
    matlab_batch_dir = tmp_path / "matlab-batch"
    matlab_batch_dir.mkdir()
    source_surface = parity_experiment.ExactProofSourceSurface(
        run_root=source_run_root,
        checkpoints_dir=source_run_root / parity_experiment.CHECKPOINTS_DIR,
        validated_params_path=source_run_root / parity_experiment.VALIDATED_PARAMS_PATH,
        oracle_surface=parity_experiment.OracleSurface(
            oracle_root=matlab_batch_dir,
            manifest_path=None,
            matlab_batch_dir=matlab_batch_dir,
            matlab_vector_paths={},
            oracle_id=None,
            matlab_source_version=None,
            dataset_hash=None,
        ),
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
        lambda _run_root, oracle_root=None: source_surface,
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
        "Completed edge candidate generation through MATLAB-style frontier workflow (candidates=1)"
    )
    assert snapshot["stages"]["edges"]["detail"] == snapshot["current_detail"]
    assert snapshot["artifacts"]["edge_candidate_iterations"] == "512"
    assert snapshot["artifacts"]["edge_candidate_count"] == "1"
    assert snapshot["artifacts"]["candidate_progress_point_count"] == "3"

    progress_jsonl = dest_run_root / parity_experiment.CANDIDATE_PROGRESS_JSONL_PATH
    progress_plot = dest_run_root / parity_experiment.CANDIDATE_PROGRESS_PLOT_PATH
    recording_index = dest_run_root / parity_experiment.RECORDING_TABLES_INDEX_PATH
    candidate_progress_csv = (
        dest_run_root / parity_experiment.ANALYSIS_TABLES_DIR / "candidate_progress.csv"
    )
    candidate_coverage_summary_jsonl = (
        dest_run_root / parity_experiment.ANALYSIS_TABLES_DIR / "candidate_coverage_summary.jsonl"
    )
    assert progress_jsonl.is_file()
    assert progress_plot.is_file()
    assert recording_index.is_file()
    assert candidate_progress_csv.is_file()
    assert candidate_coverage_summary_jsonl.is_file()

    progress_lines = progress_jsonl.read_text(encoding="utf-8").splitlines()
    assert len(progress_lines) == 3
    first_point = json.loads(progress_lines[0])
    middle_point = json.loads(progress_lines[1])
    last_point = json.loads(progress_lines[-1])
    assert first_point["phase"] == "started"
    assert middle_point["phase"] == "heartbeat"
    assert middle_point["detail"] == (
        "Generating edge candidates through MATLAB-style frontier workflow "
        "(iterations=512, candidates=1)"
    )
    assert last_point["phase"] == "completed"

    recording_tables = json.loads(recording_index.read_text(encoding="utf-8"))
    table_names = {entry["name"] for entry in recording_tables["tables"]}
    assert "candidate_progress" in table_names
    assert "candidate_coverage_summary" in table_names


def test_persist_recording_tables_flattens_existing_run_artifacts(tmp_path):
    run_root = tmp_path / "recorded-run"
    materialize_run_snapshot(
        run_root,
        {
            "run_id": "run-42",
            "status": "completed",
            "target_stage": "network",
            "current_stage": "network",
            "overall_progress": 1.0,
            "stages": {
                "edges": {
                    "name": "edges",
                    "status": "completed",
                    "progress": 1.0,
                    "detail": "Edges extracted",
                    "artifacts": {
                        "candidate_audit.json": "02_Output/python_results/stages/edges/candidate_audit.json"
                    },
                }
            },
            "optional_tasks": {
                "exports": {
                    "name": "exports",
                    "status": "completed",
                    "progress": 1.0,
                    "detail": "Exported json",
                    "artifacts": {"json": "output/network.json"},
                }
            },
            "artifacts": {
                "edges.candidate_audit.json": "02_Output/python_results/stages/edges/candidate_audit.json"
            },
            "errors": [],
            "provenance": {"source": "test-builder"},
            "last_event": "Run completed",
        },
    )
    _write_json(
        run_root / parity_experiment.RUN_MANIFEST_PATH,
        {
            "run_id": "run-42",
            "kind": "slavv_run",
            "status": "completed",
            "stage_metrics": {
                "edges": {
                    "status": "completed",
                    "elapsed_seconds": 1.5,
                    "peak_memory_bytes": 1024,
                }
            },
        },
    )
    _write_json(
        run_root / parity_experiment.EDGE_CANDIDATE_AUDIT_PATH,
        {
            "candidate_connection_count": 2,
            "candidate_origin_count": 1,
            "candidate_traces": 2,
            "diagnostic_counters": {"watershed_total_pairs": 1},
            "pair_source_breakdown": {"fallback_only_pair_count": 1},
            "frontier_per_origin_candidate_counts": {"7": 2},
            "per_origin_summary": [
                {
                    "origin_index": 7,
                    "candidate_connection_count": 2,
                    "fallback_candidate_count": 1,
                    "watershed_candidate_count": 1,
                }
            ],
        },
    )
    progress_records = [
        {"phase": "started", "iteration_count": 0, "candidate_count": 0},
        {"phase": "completed", "iteration_count": 10, "candidate_count": 2},
    ]
    (run_root / parity_experiment.CANDIDATE_PROGRESS_JSONL_PATH).parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    (run_root / parity_experiment.CANDIDATE_PROGRESS_JSONL_PATH).write_text(
        ""
        .join(f"{json.dumps(record)}\n" for record in progress_records),
        encoding="utf-8",
    )

    index_payload = parity_experiment.persist_recording_tables(run_root)

    assert index_payload["table_count"] >= 6
    assert (
        run_root / parity_experiment.RECORDING_TABLES_INDEX_PATH
    ).is_file()
    assert (
        run_root / parity_experiment.ANALYSIS_TABLES_DIR / "run_snapshot_stages.jsonl"
    ).is_file()
    assert (
        run_root / parity_experiment.ANALYSIS_TABLES_DIR / "candidate_audit_per_origin.csv"
    ).is_file()
    assert (
        run_root / parity_experiment.ANALYSIS_TABLES_DIR / "candidate_progress.csv"
    ).is_file()

    stage_rows = (
        (run_root / parity_experiment.ANALYSIS_TABLES_DIR / "run_snapshot_stages.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert len(stage_rows) == 1
    stage_row = json.loads(stage_rows[0])
    assert stage_row["stage_key"] == "edges"
    assert stage_row["artifact_count"] == 1

    per_origin_rows = (
        (run_root / parity_experiment.ANALYSIS_TABLES_DIR / "candidate_audit_per_origin.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert len(per_origin_rows) == 1
    per_origin_row = json.loads(per_origin_rows[0])
    assert per_origin_row["origin_index"] == 7

    metric_rows = (
        (run_root / parity_experiment.ANALYSIS_TABLES_DIR / "candidate_audit_origin_metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert len(metric_rows) == 1
    metric_row = json.loads(metric_rows[0])
    assert metric_row["metric_name"] == "frontier_per_origin_candidate_counts"


def test_persist_param_storage_writes_split_param_files(tmp_path):
    dest_run_root = (_build_experiment_root(tmp_path) / "runs" / "dest-run").resolve()
    parity_experiment.ensure_dest_run_layout(dest_run_root)

    parity_experiment._persist_param_storage(
        dest_run_root,
        {
            "comparison_exact_network": True,
            "direction_method": "hessian",
            "discrete_tracing": False,
            "edge_method": "tracing",
            "energy_method": "hessian",
            "energy_projection_mode": "matlab",
            "energy_storage_format": "joblib",
            "parity_candidate_salvage_mode": "legacy",
        },
    )

    shared_params = json.loads(
        (dest_run_root / parity_experiment.SHARED_PARAMS_PATH).read_text(encoding="utf-8")
    )
    python_derived = json.loads(
        (dest_run_root / parity_experiment.PYTHON_DERIVED_PARAMS_PATH).read_text(encoding="utf-8")
    )
    param_diff = json.loads(
        (dest_run_root / parity_experiment.PARAM_DIFF_PATH).read_text(encoding="utf-8")
    )

    assert shared_params["energy_projection_mode"] == "matlab"
    assert python_derived["orchestration_params"]["comparison_exact_network"] is True
    assert python_derived["python_only_params"]["parity_candidate_salvage_mode"] == "legacy"
    assert "parity_candidate_salvage_mode" in param_diff["disallowed_python_only_keys"]


def test_derive_exact_params_from_oracle_includes_released_matlab_edge_constants(tmp_path):
    run_root = tmp_path / "source-run"
    matlab_batch_dir = _materialize_exact_matlab_batch(run_root)
    settings_dir = matlab_batch_dir / "settings"
    settings_dir.mkdir(parents=True, exist_ok=True)
    savemat(
        settings_dir / "energy_260421.mat",
        {
            "microns_per_voxel": np.array([0.5, 0.5, 1.0], dtype=np.float64),
            "radius_of_smallest_vessel_in_microns": 1.5,
            "radius_of_largest_vessel_in_microns": 40.0,
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
            "max_edge_length_per_origin_radius": 30,
            "space_strel_apothem_edges": 1,
            "number_of_edges_per_vertex": 4,
        },
    )
    savemat(settings_dir / "network_260421.mat", {"sigma_strand_smoothing": 1})
    oracle_surface = parity_experiment.OracleSurface(
        oracle_root=run_root,
        manifest_path=None,
        matlab_batch_dir=matlab_batch_dir,
        matlab_vector_paths=parity_experiment.find_matlab_vector_paths(matlab_batch_dir),
        oracle_id="oracle-a",
        matlab_source_version="matlab-a",
        dataset_hash="dataset-a",
    )

    params, _settings_paths, _settings_payloads = parity_experiment.derive_exact_params_from_oracle(
        oracle_surface
    )

    assert params["step_size_per_origin_radius"] == 1.0
    assert params["max_edge_energy"] == 0.0
    assert params["edge_number_tolerance"] == 2
    assert params["distance_tolerance_per_origin_radius"] == 3.0
    assert params["distance_tolerance"] == 3.0
    assert params["radius_tolerance"] == 0.5
    assert params["energy_tolerance"] == 1.0
    assert params["direction_tolerance"] == 1.0


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
