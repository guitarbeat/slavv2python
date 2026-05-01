"""Integration coverage for the developer parity experiment runner."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pytest
from dev.tests.support.payload_builders import (
    build_edges_payload,
    build_energy_result,
    build_network_payload,
    build_vertices_payload,
)
from dev.tests.support.run_state_builders import (
    materialize_checkpoint_surface,
    materialize_run_snapshot,
)
from scipy.io import savemat

parity_experiment = importlib.import_module("dev.scripts.cli.parity_experiment")


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _build_experiment_root(tmp_path: Path) -> Path:
    root = tmp_path / "live-parity"
    for name in ("datasets", "oracles", "reports", "runs"):
        (root / name).mkdir(parents=True, exist_ok=True)
    return root


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
            "size_of_image": np.array([2, 2, 2], dtype=np.uint16),
            "intensity_limits": np.array([0, 1], dtype=np.uint16),
            "energy_runtime_in_seconds": np.array([1.0], dtype=np.float64),
        },
    )
    savemat(
        vectors_dir / "vertices_260421.mat",
        {
            "vertex_space_subscripts": np.array(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64
            ),
            "vertex_scale_subscripts": np.array([2, 3], dtype=np.int16),
            "vertex_energies": np.array([-2.0, -1.0], dtype=np.float64),
        },
    )
    savemat(
        vectors_dir / "edges_260421.mat",
        {
            "edges2vertices": np.array([[1, 2]], dtype=np.int16),
            "edge_space_subscripts": _cell(
                [
                    np.array(
                        [
                            [1.0, 2.0, 3.0],
                            [4.0, 5.0, 6.0],
                        ],
                        dtype=np.float64,
                    )
                ]
            ),
            "edge_scale_subscripts": _cell([np.array([2.0, 2.5], dtype=np.float64)]),
            "edge_energies": _cell([np.array([-4.0, -3.0], dtype=np.float64)]),
            "mean_edge_energies": np.array([-3.5], dtype=np.float64),
        },
    )
    savemat(
        vectors_dir / "network_260421.mat",
        {
            "strands2vertices": np.array([[1, 2]], dtype=np.int16),
            "bifurcation_vertices": np.empty((0,), dtype=np.int16),
            "strand_subscripts": _cell(
                [
                    np.array(
                        [
                            [1.0, 2.0, 3.0, 2.0],
                            [4.0, 5.0, 6.0, 2.5],
                        ],
                        dtype=np.float64,
                    )
                ]
            ),
            "strand_energies": _cell([np.array([-4.0, -3.0], dtype=np.float64)]),
            "mean_strand_energies": np.array([-3.5], dtype=np.float64),
            "vessel_directions": _cell(
                [np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)]
            ),
        },
    )
    return batch_dir


def _exact_vertex_payload() -> dict[str, object]:
    return {
        "positions": np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32),
        "scales": np.array([1, 2], dtype=np.int16),
        "energies": np.array([-2.0, -1.0], dtype=np.float32),
    }


def _exact_edge_payload(*, energies: np.ndarray | None = None) -> dict[str, object]:
    return {
        "connections": np.array([[0, 1]], dtype=np.int32),
        "traces": [np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32)],
        "scale_traces": [np.array([1.0, 1.5], dtype=np.float32)],
        "energy_traces": [np.array([-4.0, -3.0], dtype=np.float32)],
        "energies": np.array([-3.5], dtype=np.float32) if energies is None else energies,
        "bridge_vertex_positions": np.empty((0, 3), dtype=np.float32),
        "bridge_vertex_scales": np.empty((0,), dtype=np.int16),
        "bridge_vertex_energies": np.empty((0,), dtype=np.float32),
        "bridge_edges": {
            "connections": np.empty((0, 2), dtype=np.int32),
            "traces": [],
            "scale_traces": [],
            "energy_traces": [],
            "energies": np.empty((0,), dtype=np.float32),
        },
    }


def _exact_network_payload() -> dict[str, object]:
    return {
        "strands": [[0, 1]],
        "bifurcations": np.empty((0,), dtype=np.int32),
        "strand_subscripts": [
            np.array(
                [
                    [0.0, 1.0, 2.0, 1.0],
                    [3.0, 4.0, 5.0, 1.5],
                ],
                dtype=np.float32,
            )
        ],
        "strand_energy_traces": [np.array([-4.0, -3.0], dtype=np.float32)],
        "mean_strand_energies": np.array([-3.5], dtype=np.float32),
        "vessel_directions": [np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)],
    }


def _pass_report() -> dict[str, object]:
    return {"passed": True, "first_failure": None}


def _fail_report(field_path: str) -> dict[str, object]:
    return {
        "passed": False,
        "first_failure": {"field_path": field_path},
    }


def _exact_validated_params(**overrides: object) -> dict[str, object]:
    params: dict[str, object] = {
        "comparison_exact_network": True,
        "direction_method": "hessian",
        "discrete_tracing": False,
        "edge_method": "tracing",
        "energy_method": "hessian",
        "energy_projection_mode": "matlab",
    }
    params.update(overrides)
    return params


@pytest.mark.integration
def test_rerun_python_creates_fresh_dest_root_and_writes_summary(tmp_path, monkeypatch):
    experiment_root = _build_experiment_root(tmp_path)
    source_run_root = experiment_root / "runs" / "source-run"
    dest_run_root = experiment_root / "runs" / "dest-run"
    input_file = tmp_path / "input.tif"
    input_file.write_bytes(b"placeholder-tiff")

    materialize_checkpoint_surface(
        source_run_root,
        stages=("energy", "vertices", "edges", "network"),
        payloads={
            "energy": build_energy_result(),
            "vertices": build_vertices_payload(
                positions=[
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                ]
            ),
            "edges": build_edges_payload(
                traces=[
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                    [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                ],
                connections=[[0, 1], [1, 2]],
            ),
            "network": build_network_payload(strands=[[0, 1, 2]]),
        },
    )
    _write_json(
        source_run_root / "03_Analysis" / "comparison_report.json",
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
        source_run_root / "99_Metadata" / "validated_params.json",
        {"number_of_edges_per_vertex": 4},
    )
    materialize_run_snapshot(
        source_run_root,
        {
            "run_id": "run-1",
            "provenance": {
                "input_file": str(input_file),
            },
        },
    )
    source_edges_checkpoint = (
        source_run_root / parity_experiment.CHECKPOINTS_DIR / "checkpoint_edges.pkl"
    )
    source_edges_bytes = source_edges_checkpoint.read_bytes()

    calls: list[dict[str, object]] = []

    class FakeProcessor:
        def process_image(
            self,
            image,
            parameters,
            *,
            run_dir=None,
            force_rerun_from=None,
            **_kwargs,
        ):
            calls.append(
                {
                    "shape": tuple(image.shape),
                    "parameters": dict(parameters),
                    "run_dir": run_dir,
                    "force_rerun_from": force_rerun_from,
                }
            )
            checkpoint_dir = Path(run_dir) / parity_experiment.CHECKPOINTS_DIR
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            from joblib import dump

            dump(
                build_edges_payload(
                    traces=[
                        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                        [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                        [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
                    ],
                    connections=[[0, 1], [1, 2], [2, 3]],
                ),
                checkpoint_dir / "checkpoint_edges.pkl",
            )
            dump(
                build_network_payload(strands=[[0, 1, 2], [2, 3]]),
                checkpoint_dir / "checkpoint_network.pkl",
            )
            return {}

    monkeypatch.setattr(parity_experiment, "load_tiff_volume", lambda _path: np.ones((2, 2, 2)))
    monkeypatch.setattr(parity_experiment, "SLAVVProcessor", FakeProcessor)

    parity_experiment.main(
        [
            "rerun-python",
            "--source-run-root",
            str(source_run_root),
            "--dest-run-root",
            str(dest_run_root),
            "--rerun-from",
            "edges",
        ]
    )

    assert source_edges_checkpoint.read_bytes() == source_edges_bytes
    assert calls == [
        {
            "shape": (2, 2, 2),
            "parameters": {"number_of_edges_per_vertex": 4},
            "run_dir": str(dest_run_root.resolve()),
            "force_rerun_from": "edges",
        }
    ]

    summary_payload = json.loads(
        (dest_run_root / parity_experiment.SUMMARY_JSON_PATH).read_text(encoding="utf-8")
    )
    assert summary_payload["matlab_counts"] == {"vertices": 4, "edges": 5, "strands": 3}
    assert summary_payload["source_python_counts"] == {"vertices": 4, "edges": 2, "strands": 1}
    assert summary_payload["new_python_counts"] == {"vertices": 4, "edges": 3, "strands": 2}
    assert summary_payload["diff_vs_matlab"] == {"vertices": 0, "edges": -2, "strands": -1}
    assert summary_payload["diff_vs_source_python"] == {"vertices": 0, "edges": 1, "strands": 1}
    assert (dest_run_root / "00_Refs" / "source_comparison_report.json").is_file()
    assert (dest_run_root / "00_Refs" / "source_validated_params.json").is_file()
    assert (dest_run_root / parity_experiment.SHARED_PARAMS_PATH).is_file()
    assert (dest_run_root / parity_experiment.PYTHON_DERIVED_PARAMS_PATH).is_file()
    run_manifest = json.loads(
        (dest_run_root / parity_experiment.RUN_MANIFEST_PATH).read_text(encoding="utf-8")
    )
    assert run_manifest["dataset_hash"] == parity_experiment.fingerprint_file(input_file)
    index_lines = (
        (experiment_root / parity_experiment.EXPERIMENT_INDEX_PATH)
        .read_text(encoding="utf-8")
        .strip()
        .splitlines()
    )
    assert any('"command":"rerun-python"' in line for line in index_lines)


@pytest.mark.integration
def test_rerun_python_syncs_exact_vertex_checkpoint_from_matlab(tmp_path, monkeypatch):
    source_run_root = tmp_path / "source-run"
    dest_run_root = tmp_path / "dest-run"
    input_file = tmp_path / "input.tif"
    input_file.write_bytes(b"placeholder-tiff")
    _materialize_exact_matlab_batch(source_run_root)

    materialize_checkpoint_surface(
        source_run_root,
        stages=("energy", "vertices", "edges", "network"),
        payloads={
            "energy": {"energy_origin": "python_native_hessian"},
            "vertices": {
                "positions": np.zeros((2, 3), dtype=np.float32),
                "scales": np.zeros((2,), dtype=np.int16),
                "energies": np.zeros((2,), dtype=np.float32),
                "radii_microns": np.array([1.0, 1.0], dtype=np.float32),
                "count": 2,
            },
            "edges": build_edges_payload(
                traces=[[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]],
                connections=[[0, 1]],
            ),
            "network": build_network_payload(strands=[[0, 1]]),
        },
    )
    _write_json(
        source_run_root / "03_Analysis" / "comparison_report.json",
        {
            "matlab": {"vertices_count": 2, "edges_count": 1, "strand_count": 1},
            "python": {"vertices_count": 2, "edges_count": 1, "network_strands_count": 1},
            "vertices": {"matlab_count": 2, "python_count": 2},
            "edges": {"matlab_count": 1, "python_count": 1},
            "network": {"matlab_strand_count": 1, "python_strand_count": 1},
        },
    )
    _write_json(
        source_run_root / "99_Metadata" / "validated_params.json",
        _exact_validated_params(),
    )
    materialize_run_snapshot(
        source_run_root,
        {"run_id": "run-1", "provenance": {"input_file": str(input_file)}},
    )

    class FakeProcessor:
        def process_image(
            self, image, parameters, *, run_dir=None, force_rerun_from=None, **_kwargs
        ):
            from joblib import load

            checkpoint_vertices = load(
                Path(run_dir) / parity_experiment.CHECKPOINTS_DIR / "checkpoint_vertices.pkl"
            )
            np.testing.assert_array_equal(
                checkpoint_vertices["positions"],
                np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32),
            )
            np.testing.assert_array_equal(
                checkpoint_vertices["scales"],
                np.array([1, 2], dtype=np.int16),
            )
            np.testing.assert_array_equal(
                checkpoint_vertices["energies"],
                np.array([-2.0, -1.0], dtype=np.float32),
            )
            from joblib import dump

            checkpoint_dir = Path(run_dir) / parity_experiment.CHECKPOINTS_DIR
            dump(
                build_edges_payload(
                    traces=[[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]], connections=[[0, 1]]
                ),
                checkpoint_dir / "checkpoint_edges.pkl",
            )
            dump(build_network_payload(strands=[[0, 1]]), checkpoint_dir / "checkpoint_network.pkl")
            return {}

    monkeypatch.setattr(parity_experiment, "load_tiff_volume", lambda _path: np.ones((2, 2, 2)))
    monkeypatch.setattr(parity_experiment, "SLAVVProcessor", FakeProcessor)

    parity_experiment.main(
        [
            "rerun-python",
            "--source-run-root",
            str(source_run_root),
            "--dest-run-root",
            str(dest_run_root),
            "--rerun-from",
            "edges",
        ]
    )

    provenance = json.loads(
        (dest_run_root / "99_Metadata" / "experiment_provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["exact_vertex_checkpoint_sync"] is True


@pytest.mark.integration
def test_prove_exact_writes_pass_report_for_matching_artifacts(tmp_path):
    source_run_root = tmp_path / "source-run"
    dest_run_root = tmp_path / "dest-run"
    _materialize_exact_matlab_batch(source_run_root)

    vertex_payload = _exact_vertex_payload()
    edge_payload = _exact_edge_payload()
    network_payload = _exact_network_payload()

    materialize_checkpoint_surface(
        source_run_root,
        stages=("energy", "vertices", "edges", "network"),
        payloads={
            "energy": {"energy_origin": "python_native_hessian"},
            "vertices": vertex_payload,
            "edges": edge_payload,
            "network": network_payload,
        },
    )
    materialize_checkpoint_surface(
        dest_run_root,
        stages=("vertices", "edges", "network"),
        payloads={
            "vertices": vertex_payload,
            "edges": edge_payload,
            "network": network_payload,
        },
    )
    _write_json(
        source_run_root / "99_Metadata" / "validated_params.json",
        _exact_validated_params(),
    )

    parity_experiment.main(
        [
            "prove-exact",
            "--source-run-root",
            str(source_run_root),
            "--dest-run-root",
            str(dest_run_root),
        ]
    )

    report_payload = json.loads(
        (dest_run_root / parity_experiment.EXACT_PROOF_JSON_PATH).read_text(encoding="utf-8")
    )
    assert report_payload["passed"] is True
    assert report_payload["first_failure"] is None
    assert report_payload["exact_route_gate"] == (
        "comparison_exact_network + python_native_hessian energy provenance"
    )


@pytest.mark.integration
def test_prove_exact_reports_first_edge_mismatch(tmp_path):
    source_run_root = tmp_path / "source-run"
    dest_run_root = tmp_path / "dest-run"
    _materialize_exact_matlab_batch(source_run_root)

    materialize_checkpoint_surface(
        source_run_root,
        stages=("energy", "vertices", "edges", "network"),
        payloads={
            "energy": {"energy_origin": "python_native_hessian"},
        },
    )
    materialize_checkpoint_surface(
        dest_run_root,
        stages=("vertices", "edges", "network"),
        payloads={
            "vertices": _exact_vertex_payload(),
            "edges": _exact_edge_payload(energies=np.array([-9.0], dtype=np.float32)),
            "network": _exact_network_payload(),
        },
    )
    _write_json(
        source_run_root / "99_Metadata" / "validated_params.json",
        _exact_validated_params(),
    )

    with pytest.raises(SystemExit, match="1"):
        parity_experiment.main(
            [
                "prove-exact",
                "--source-run-root",
                str(source_run_root),
                "--dest-run-root",
                str(dest_run_root),
            ]
        )

    report_payload = json.loads(
        (dest_run_root / parity_experiment.EXACT_PROOF_JSON_PATH).read_text(encoding="utf-8")
    )
    assert report_payload["passed"] is False
    assert report_payload["first_failing_stage"] == "edges"
    assert report_payload["first_failing_field_path"] == "edges.energies"


@pytest.mark.integration
def test_prove_exact_falls_back_to_candidate_checkpoint_when_edges_checkpoint_missing(tmp_path):
    source_run_root = tmp_path / "source-run"
    dest_run_root = tmp_path / "dest-run"
    _materialize_exact_matlab_batch(source_run_root)

    materialize_checkpoint_surface(
        source_run_root,
        stages=("energy", "vertices", "edges", "network"),
        payloads={
            "energy": {"energy_origin": "python_native_hessian"},
        },
    )
    materialize_checkpoint_surface(
        dest_run_root,
        stages=("vertices", "edge_candidates"),
        payloads={
            "vertices": _exact_vertex_payload(),
            "edge_candidates": {
                "connections": np.array([[0, 1]], dtype=np.int32),
                "traces": [np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32)],
                "scale_traces": [np.array([1.0, 1.5], dtype=np.float32)],
                "energy_traces": [np.array([-4.0, -3.0], dtype=np.float32)],
                "metrics": np.array([-3.5], dtype=np.float32),
                "origin_indices": np.array([0], dtype=np.int32),
                "connection_sources": ["global_watershed"],
                "diagnostics": {},
                "candidate_source": "global_watershed",
                "matlab_global_watershed_exact": True,
            },
        },
    )
    _write_json(
        source_run_root / "99_Metadata" / "validated_params.json",
        _exact_validated_params(),
    )

    parity_experiment.main(
        [
            "prove-exact",
            "--source-run-root",
            str(source_run_root),
            "--dest-run-root",
            str(dest_run_root),
            "--stage",
            "edges",
        ]
    )

    report_payload = json.loads(
        (dest_run_root / parity_experiment.EXACT_PROOF_JSON_PATH).read_text(encoding="utf-8")
    )
    assert report_payload["passed"] is True
    assert report_payload["report_scope"] == "candidate boundary fallback (edges.connections only)"
    assert report_payload["candidate_surface"]["matlab_pair_count"] == 1
    assert report_payload["candidate_surface"]["python_pair_count"] == 1
    assert report_payload["candidate_checkpoint_path"].endswith("checkpoint_edge_candidates.pkl")
    assert report_payload["edge_checkpoint_path"].endswith("checkpoint_edges.pkl")


@pytest.mark.integration
def test_replay_edges_consumes_candidate_checkpoint_and_writes_proof(tmp_path, monkeypatch):
    source_run_root = tmp_path / "source-run"
    dest_run_root = tmp_path / "dest-run"
    _materialize_exact_matlab_batch(source_run_root)
    materialize_checkpoint_surface(
        source_run_root,
        stages=("energy", "vertices"),
        payloads={
            "energy": {
                "energy_origin": "python_native_hessian",
                "energy": np.zeros((4, 4, 4), dtype=np.float32),
                "scale_indices": np.zeros((4, 4, 4), dtype=np.int16),
                "lumen_radius_microns": np.array([1.0], dtype=np.float32),
                "lumen_radius_pixels_axes": np.ones((1, 3), dtype=np.float32),
            },
            "vertices": {
                "positions": np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32),
                "scales": np.array([1, 2], dtype=np.int16),
                "energies": np.array([-2.0, -1.0], dtype=np.float32),
                "count": 2,
            },
        },
    )
    _write_json(
        source_run_root / "99_Metadata" / "validated_params.json",
        _exact_validated_params(microns_per_voxel=[1.0, 1.0, 1.0]),
    )
    materialize_checkpoint_surface(
        dest_run_root,
        stages=("edge_candidates",),
        payloads={
            "edge_candidates": {
                "connections": np.array([[0, 1]], dtype=np.int32),
                "traces": [np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32)],
                "scale_traces": [np.array([1.0, 1.5], dtype=np.float32)],
                "energy_traces": [np.array([-4.0, -3.0], dtype=np.float32)],
                "metrics": np.array([-3.5], dtype=np.float32),
                "origin_indices": np.array([0], dtype=np.int32),
                "connection_sources": ["global_watershed"],
                "diagnostics": {},
                "candidate_source": "global_watershed",
                "matlab_global_watershed_exact": True,
            }
        },
    )

    def _fake_choose(*_args, **_kwargs):
        return _exact_edge_payload()

    monkeypatch.setattr(parity_experiment, "choose_edges_for_workflow", _fake_choose)
    monkeypatch.setattr(
        parity_experiment,
        "add_vertices_to_edges_matlab_style",
        lambda chosen, *_args, **_kwargs: chosen,
    )
    monkeypatch.setattr(
        parity_experiment, "finalize_edges_matlab_style", lambda chosen, **_kwargs: chosen
    )

    parity_experiment.main(
        [
            "replay-edges",
            "--source-run-root",
            str(source_run_root),
            "--dest-run-root",
            str(dest_run_root),
        ]
    )

    assert (dest_run_root / parity_experiment.CHECKPOINTS_DIR / "checkpoint_edges.pkl").is_file()
    report_payload = json.loads(
        (dest_run_root / parity_experiment.EDGE_REPLAY_PROOF_JSON_PATH).read_text(encoding="utf-8")
    )
    assert report_payload["passed"] is True


@pytest.mark.integration
def test_promote_dataset_copies_input_and_writes_manifest(tmp_path):
    experiment_root = _build_experiment_root(tmp_path)
    dataset_file = tmp_path / "input.tif"
    dataset_file.write_bytes(b"tiff-payload")
    dataset_hash = parity_experiment.fingerprint_file(dataset_file)

    parity_experiment.main(
        [
            "promote-dataset",
            "--dataset-file",
            str(dataset_file),
            "--experiment-root",
            str(experiment_root),
        ]
    )

    dataset_root = experiment_root / "datasets" / dataset_hash
    manifest = json.loads(
        (dataset_root / parity_experiment.DATASET_MANIFEST_PATH).read_text(encoding="utf-8")
    )
    assert manifest["dataset_hash"] == dataset_hash
    assert manifest["stored_input_file"] == str(
        dataset_root / parity_experiment.DATASET_INPUT_DIR / dataset_file.name
    )
    assert (dataset_root / parity_experiment.DATASET_INPUT_DIR / dataset_file.name).read_bytes() == (
        b"tiff-payload"
    )
    assert (
        dataset_root / parity_experiment.DATASET_INPUT_DIR / f"{dataset_file.name}.sha256"
    ).is_file()
    index_lines = (
        (experiment_root / parity_experiment.EXPERIMENT_INDEX_PATH)
        .read_text(encoding="utf-8")
        .strip()
        .splitlines()
    )
    assert any(f'"id":"{dataset_hash}"' in line and '"kind":"dataset"' in line for line in index_lines)


@pytest.mark.integration
def test_init_exact_run_bootstraps_source_surface_from_dataset_and_oracle(
    tmp_path, monkeypatch
):
    experiment_root = _build_experiment_root(tmp_path)
    dataset_file = tmp_path / "input.tif"
    dataset_file.write_bytes(b"tiff-payload")
    dataset_hash = parity_experiment.fingerprint_file(dataset_file)
    dataset_root = experiment_root / "datasets" / dataset_hash

    parity_experiment.main(
        [
            "promote-dataset",
            "--dataset-file",
            str(dataset_file),
            "--experiment-root",
            str(experiment_root),
        ]
    )

    matlab_batch_dir = (
        tmp_path / "matlab-source" / "01_Input" / "matlab_results" / "batch_260421-151654"
    )
    _materialize_exact_matlab_batch(tmp_path / "matlab-source")
    settings_dir = matlab_batch_dir / "settings"
    settings_dir.mkdir(parents=True, exist_ok=True)
    savemat(
        settings_dir / "energy_260421-151654.mat",
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
    savemat(
        settings_dir / "network_260421.mat",
        {
            "sigma_strand_smoothing": 1,
        },
    )

    oracle_root = experiment_root / "oracles" / "oracle-a"
    parity_experiment.main(
        [
            "promote-oracle",
            "--matlab-batch-dir",
            str(matlab_batch_dir),
            "--oracle-root",
            str(oracle_root),
            "--dataset-file",
            str(dataset_file),
            "--oracle-id",
            "oracle-a",
        ]
    )

    def _fake_load_tiff(_path):
        return np.ones((2, 2, 2), dtype=np.uint16)

    class FakeProcessor:
        def process_image(self, image, parameters, *, run_dir=None, stop_after=None, **_kwargs):
            assert tuple(image.shape) == (2, 2, 2)
            assert parameters["comparison_exact_network"] is True
            checkpoint_dir = Path(run_dir) / parity_experiment.CHECKPOINTS_DIR
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            from joblib import dump

            dump(
                {
                    "energy_origin": "python_native_hessian",
                    "energy": np.zeros((2, 2, 2), dtype=np.float32),
                    "scale_indices": np.zeros((2, 2, 2), dtype=np.int16),
                    "lumen_radius_microns": np.array([1.0], dtype=np.float32),
                    "lumen_radius_pixels_axes": np.ones((1, 3), dtype=np.float32),
                },
                checkpoint_dir / "checkpoint_energy.pkl",
            )
            if stop_after == "vertices":
                dump(
                    {
                        "positions": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
                        "scales": np.array([0], dtype=np.int16),
                        "energies": np.array([-1.0], dtype=np.float32),
                        "count": 1,
                    },
                    checkpoint_dir / "checkpoint_vertices.pkl",
                )
            return {"parameters": parameters}

    monkeypatch.setattr(parity_experiment, "load_tiff_volume", _fake_load_tiff)
    monkeypatch.setattr(parity_experiment, "SLAVVProcessor", FakeProcessor)

    dest_run_root = experiment_root / "runs" / "seed-a"
    parity_experiment.main(
        [
            "init-exact-run",
            "--dataset-root",
            str(dataset_root),
            "--oracle-root",
            str(oracle_root),
            "--dest-run-root",
            str(dest_run_root),
        ]
    )

    params = json.loads(
        (dest_run_root / parity_experiment.VALIDATED_PARAMS_PATH).read_text(encoding="utf-8")
    )
    assert params["comparison_exact_network"] is True
    assert params["microns_per_voxel"] == [0.5, 0.5, 1.0]
    assert params["step_size_per_origin_radius"] == 1.0
    assert params["max_edge_energy"] == 0.0
    assert params["edge_number_tolerance"] == 2
    assert params["distance_tolerance"] == 3.0
    assert params["radius_tolerance"] == 0.5
    assert params["energy_tolerance"] == 1.0
    assert (dest_run_root / "00_Refs" / "dataset_manifest.json").is_file()
    assert (dest_run_root / "00_Refs" / "oracle_manifest.json").is_file()
    provenance = json.loads(
        (dest_run_root / parity_experiment.EXPERIMENT_PROVENANCE_PATH).read_text(
            encoding="utf-8"
        )
    )
    assert provenance["dataset_hash"] == dataset_hash
    assert provenance["oracle_id"] == "oracle-a"
    assert provenance["oracle_size_of_image"] == [2, 2, 2]
    assert provenance["input_axis_permutation"] is None
    run_manifest = json.loads(
        (dest_run_root / parity_experiment.RUN_MANIFEST_PATH).read_text(encoding="utf-8")
    )
    assert run_manifest["command"] == "init-exact-run"
    assert run_manifest["oracle_id"] == "oracle-a"
    surface = parity_experiment.validate_exact_proof_source_surface(dest_run_root)
    assert surface.oracle_surface.oracle_root == oracle_root.resolve()


@pytest.mark.integration
def test_init_exact_run_reorients_input_volume_to_match_oracle_energy_shape(
    tmp_path, monkeypatch
):
    experiment_root = _build_experiment_root(tmp_path)
    dataset_file = tmp_path / "input.tif"
    dataset_file.write_bytes(b"tiff")
    dataset_hash = parity_experiment._materialize_dataset_record(
        experiment_root,
        dataset_hash=None,
        dataset_file=dataset_file,
    )
    dataset_root = experiment_root / "datasets" / dataset_hash
    matlab_batch_dir = (
        tmp_path / "matlab-source" / "01_Input" / "matlab_results" / "batch_260421-151654"
    )
    matlab_batch_dir.parent.mkdir(parents=True, exist_ok=True)
    _materialize_exact_matlab_batch(tmp_path / "matlab-source")
    settings_dir = matlab_batch_dir / "settings"
    settings_dir.mkdir(parents=True, exist_ok=True)
    savemat(
        settings_dir / "energy_260421.mat",
        {
            "microns_per_voxel": np.array([0.5, 0.5, 1.0], dtype=np.float64),
            "radius_of_smallest_vessel_in_microns": 1.0,
            "radius_of_largest_vessel_in_microns": 2.0,
            "sample_index_of_refraction": 1.33,
            "numerical_aperture": 0.95,
            "scales_per_octave": 1.0,
            "max_voxels_per_node_energy": 1000000,
            "gaussian_to_ideal_ratio": 0.5,
            "spherical_to_annular_ratio": 0.5,
            "approximating_PSF": 0,
            "pixels_per_sigma_PSF": np.array([1.0, 1.0, 1.0], dtype=np.float64),
            "lumen_radius_in_microns_range": np.array([1.0], dtype=np.float64),
            "lumen_radius_in_pixels_range": np.array([[2.0, 2.0, 1.0]], dtype=np.float64),
            "excitation_wavelength_in_microns": 0.9,
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
    savemat(
        settings_dir / "network_260421.mat",
        {
            "sigma_strand_smoothing": 1,
        },
    )
    savemat(
        matlab_batch_dir / "data" / "energy_260421.mat",
        {
            "size_of_image": np.array([4, 5, 3], dtype=np.uint16),
            "intensity_limits": np.array([0, 1], dtype=np.uint16),
            "energy_runtime_in_seconds": np.array([1.0], dtype=np.float64),
        },
    )
    oracle_root = experiment_root / "oracles" / "oracle-a"
    parity_experiment.main(
        [
            "promote-oracle",
            "--matlab-batch-dir",
            str(matlab_batch_dir),
            "--oracle-root",
            str(oracle_root),
            "--dataset-file",
            str(dataset_file),
            "--oracle-id",
            "oracle-a",
        ]
    )

    def _fake_load_tiff(_path):
        return np.ones((3, 4, 5), dtype=np.uint16)

    class FakeProcessor:
        def process_image(self, image, parameters, *, run_dir=None, stop_after=None, **_kwargs):
            assert tuple(image.shape) == (4, 5, 3)
            checkpoint_dir = Path(run_dir) / parity_experiment.CHECKPOINTS_DIR
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            from joblib import dump

            dump(
                {
                    "energy_origin": "python_native_hessian",
                    "energy": np.zeros((4, 5, 3), dtype=np.float32),
                    "scale_indices": np.zeros((4, 5, 3), dtype=np.int16),
                    "lumen_radius_microns": np.array([1.0], dtype=np.float32),
                    "lumen_radius_pixels_axes": np.ones((1, 3), dtype=np.float32),
                },
                checkpoint_dir / "checkpoint_energy.pkl",
            )
            return {"parameters": parameters}

    monkeypatch.setattr(parity_experiment, "load_tiff_volume", _fake_load_tiff)
    monkeypatch.setattr(parity_experiment, "SLAVVProcessor", FakeProcessor)

    dest_run_root = experiment_root / "runs" / "seed-b"
    parity_experiment.main(
        [
            "init-exact-run",
            "--dataset-root",
            str(dataset_root),
            "--oracle-root",
            str(oracle_root),
            "--dest-run-root",
            str(dest_run_root),
            "--stop-after",
            "energy",
        ]
    )

    provenance = json.loads(
        (dest_run_root / parity_experiment.EXPERIMENT_PROVENANCE_PATH).read_text(
            encoding="utf-8"
        )
    )
    assert provenance["oracle_size_of_image"] == [4, 5, 3]
    assert provenance["input_axis_permutation"] == [1, 2, 0]


@pytest.mark.integration
def test_init_exact_run_can_finalize_existing_completed_seed(tmp_path, monkeypatch):
    experiment_root = _build_experiment_root(tmp_path)
    dataset_file = tmp_path / "input.tif"
    dataset_file.write_bytes(b"tiff")
    dataset_hash = parity_experiment._materialize_dataset_record(
        experiment_root,
        dataset_hash=None,
        dataset_file=dataset_file,
    )
    dataset_root = experiment_root / "datasets" / dataset_hash
    matlab_batch_dir = (
        tmp_path / "matlab-source" / "01_Input" / "matlab_results" / "batch_260421-151654"
    )
    matlab_batch_dir.parent.mkdir(parents=True, exist_ok=True)
    _materialize_exact_matlab_batch(tmp_path / "matlab-source")
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
    oracle_root = experiment_root / "oracles" / "oracle-a"
    parity_experiment.main(
        [
            "promote-oracle",
            "--matlab-batch-dir",
            str(matlab_batch_dir),
            "--oracle-root",
            str(oracle_root),
            "--dataset-file",
            str(dataset_file),
            "--oracle-id",
            "oracle-a",
        ]
    )

    def _fake_load_tiff(_path):
        return np.ones((2, 2, 2), dtype=np.uint16)

    class FirstProcessor:
        def process_image(self, image, parameters, *, run_dir=None, stop_after=None, **_kwargs):
            checkpoint_dir = Path(run_dir) / parity_experiment.CHECKPOINTS_DIR
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            from joblib import dump

            dump(
                {
                    "energy_origin": "python_native_hessian",
                    "energy": np.zeros((2, 2, 2), dtype=np.float32),
                    "scale_indices": np.zeros((2, 2, 2), dtype=np.int16),
                    "lumen_radius_microns": np.array([1.0], dtype=np.float32),
                    "lumen_radius_pixels_axes": np.ones((1, 3), dtype=np.float32),
                },
                checkpoint_dir / "checkpoint_energy.pkl",
            )
            snapshot_payload = {
                "status": "completed",
                "target_stage": stop_after,
                "current_stage": stop_after,
                "provenance": {
                    "source": "pipeline",
                    "layout": "structured",
                    "stop_after": stop_after,
                    "image_shape": list(image.shape),
                },
                "input_fingerprint": dataset_hash,
            }
            (Path(run_dir) / parity_experiment.METADATA_DIR).mkdir(parents=True, exist_ok=True)
            (Path(run_dir) / parity_experiment.RUN_SNAPSHOT_PATH).write_text(
                json.dumps(snapshot_payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            return {"parameters": parameters}

    monkeypatch.setattr(parity_experiment, "load_tiff_volume", _fake_load_tiff)
    monkeypatch.setattr(parity_experiment, "SLAVVProcessor", FirstProcessor)

    dest_run_root = experiment_root / "runs" / "seed-c"
    parity_experiment.main(
        [
            "init-exact-run",
            "--dataset-root",
            str(dataset_root),
            "--oracle-root",
            str(oracle_root),
            "--dest-run-root",
            str(dest_run_root),
            "--stop-after",
            "energy",
        ]
    )

    class SecondProcessor:
        def process_image(self, *_args, **_kwargs):
            raise AssertionError("completed seed should finalize without rerunning process_image")

    monkeypatch.setattr(parity_experiment, "SLAVVProcessor", SecondProcessor)
    parity_experiment.main(
        [
            "init-exact-run",
            "--dataset-root",
            str(dataset_root),
            "--oracle-root",
            str(oracle_root),
            "--dest-run-root",
            str(dest_run_root),
            "--stop-after",
            "energy",
        ]
    )

    run_manifest = json.loads(
        (dest_run_root / parity_experiment.RUN_MANIFEST_PATH).read_text(encoding="utf-8")
    )
    assert run_manifest["kind"] == "parity_source_run"
    assert run_manifest["command"] == "init-exact-run"
    snapshot_payload = json.loads(
        (dest_run_root / parity_experiment.RUN_SNAPSHOT_PATH).read_text(encoding="utf-8")
    )
    assert snapshot_payload["provenance"]["input_file"] == str(
        dataset_root / "01_Input" / dataset_file.name
    )
    assert snapshot_payload["provenance"]["oracle_id"] == "oracle-a"


@pytest.mark.integration
def test_init_exact_run_rejects_existing_active_seed(tmp_path):
    experiment_root = _build_experiment_root(tmp_path)
    dataset_file = tmp_path / "input.tif"
    dataset_file.write_bytes(b"tiff")
    dataset_hash = parity_experiment._materialize_dataset_record(
        experiment_root,
        dataset_hash=None,
        dataset_file=dataset_file,
    )
    dataset_root = experiment_root / "datasets" / dataset_hash
    matlab_batch_dir = (
        tmp_path / "matlab-source" / "01_Input" / "matlab_results" / "batch_260421-151654"
    )
    matlab_batch_dir.parent.mkdir(parents=True, exist_ok=True)
    _materialize_exact_matlab_batch(tmp_path / "matlab-source")
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
    oracle_root = experiment_root / "oracles" / "oracle-a"
    parity_experiment.main(
        [
            "promote-oracle",
            "--matlab-batch-dir",
            str(matlab_batch_dir),
            "--oracle-root",
            str(oracle_root),
            "--dataset-file",
            str(dataset_file),
            "--oracle-id",
            "oracle-a",
        ]
    )

    dest_run_root = experiment_root / "runs" / "seed-d"
    parity_experiment.ensure_dest_run_layout(dest_run_root)
    (dest_run_root / parity_experiment.EXPERIMENT_PROVENANCE_PATH).write_text(
        json.dumps(
            {
                "bootstrap_kind": "init-exact-run",
                "dataset_hash": dataset_hash,
                "oracle_id": "oracle-a",
                "stop_after": "vertices",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (dest_run_root / parity_experiment.RUN_SNAPSHOT_PATH).write_text(
        json.dumps({"status": "running"}, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="still active"):
        parity_experiment.main(
            [
                "init-exact-run",
                "--dataset-root",
                str(dataset_root),
                "--oracle-root",
                str(oracle_root),
                "--dest-run-root",
                str(dest_run_root),
            ]
        )


@pytest.mark.integration
def test_promote_oracle_writes_manifest_and_index(tmp_path):
    experiment_root = _build_experiment_root(tmp_path)
    matlab_batch_dir = (
        tmp_path / "matlab-source" / "01_Input" / "matlab_results" / "batch_260421-151654"
    )
    matlab_batch_dir.parent.mkdir(parents=True, exist_ok=True)
    _materialize_exact_matlab_batch(tmp_path / "matlab-source")
    oracle_root = experiment_root / "oracles" / "oracle-a"
    dataset_file = tmp_path / "input.tif"
    dataset_file.write_bytes(b"tiff")

    parity_experiment.main(
        [
            "promote-oracle",
            "--matlab-batch-dir",
            str(matlab_batch_dir),
            "--oracle-root",
            str(oracle_root),
            "--dataset-file",
            str(dataset_file),
            "--oracle-id",
            "oracle-a",
        ]
    )

    manifest = json.loads(
        (oracle_root / parity_experiment.ORACLE_MANIFEST_PATH).read_text(encoding="utf-8")
    )
    assert manifest["oracle_id"] == "oracle-a"
    assert manifest["dataset_hash"] == parity_experiment.fingerprint_file(dataset_file)
    assert (experiment_root / parity_experiment.EXPERIMENT_INDEX_PATH).is_file()


@pytest.mark.integration
def test_promote_report_copies_analysis_and_writes_manifest(tmp_path):
    experiment_root = _build_experiment_root(tmp_path)
    run_root = experiment_root / "runs" / "trial-a"
    parity_experiment.ensure_dest_run_layout(run_root)
    _write_json(run_root / parity_experiment.SUMMARY_JSON_PATH, {"passed": True})
    _write_json(run_root / parity_experiment.RUN_MANIFEST_PATH, {"run_id": "trial-a"})

    parity_experiment.main(
        [
            "promote-report",
            "--run-root",
            str(run_root),
        ]
    )

    report_root = experiment_root / "reports" / "trial-a"
    report_manifest = json.loads(
        (report_root / parity_experiment.REPORT_MANIFEST_PATH).read_text(encoding="utf-8")
    )
    assert report_manifest["source_run_id"] == "trial-a"
    assert (report_root / parity_experiment.SUMMARY_JSON_PATH).is_file()


@pytest.mark.integration
@pytest.mark.parametrize(
    ("failing_step", "expected_field"),
    [
        ("luts", "luts.scales[0].linear_offsets"),
        ("candidates", "candidate_coverage"),
        ("replay", "edges.connections"),
    ],
)
def test_fail_fast_stops_at_first_failing_gate(
    tmp_path,
    monkeypatch,
    failing_step,
    expected_field,
):
    source_run_root = tmp_path / "source-run"
    dest_run_root = tmp_path / "dest-run"
    order: list[str] = []
    monkeypatch.setattr(
        parity_experiment, "render_exact_preflight_report", lambda _report: "preflight"
    )
    monkeypatch.setattr(parity_experiment, "render_lut_proof_report", lambda _report: "luts")
    monkeypatch.setattr(
        parity_experiment, "render_candidate_coverage_report", lambda _report: "candidates"
    )
    monkeypatch.setattr(parity_experiment, "render_exact_proof_report", lambda _report: "replay")

    monkeypatch.setattr(
        parity_experiment,
        "_run_preflight_exact",
        lambda **_kwargs: (
            order.append("preflight")
            or (_pass_report(), dest_run_root / "a.json", dest_run_root / "a.txt")
        ),
    )
    monkeypatch.setattr(
        parity_experiment,
        "_run_prove_luts",
        lambda **_kwargs: (
            order.append("luts")
            or (
                (_fail_report(expected_field) if failing_step == "luts" else _pass_report()),
                dest_run_root / "b.json",
                dest_run_root / "b.txt",
            )
        ),
    )
    monkeypatch.setattr(
        parity_experiment,
        "_run_capture_candidates",
        lambda **_kwargs: (
            order.append("candidates")
            or (
                (_fail_report(expected_field) if failing_step == "candidates" else _pass_report()),
                {},
                dest_run_root / "c.json",
                dest_run_root / "c.txt",
            )
        ),
    )
    monkeypatch.setattr(
        parity_experiment,
        "_run_replay_edges",
        lambda **_kwargs: (
            order.append("replay")
            or (
                (_fail_report(expected_field) if failing_step == "replay" else _pass_report()),
                dest_run_root / "d.json",
                dest_run_root / "d.txt",
            )
        ),
    )
    monkeypatch.setattr(
        parity_experiment,
        "_handle_prove_exact",
        lambda _args: order.append("prove-exact"),
    )

    with pytest.raises(SystemExit, match="1"):
        parity_experiment.main(
            [
                "fail-fast",
                "--source-run-root",
                str(source_run_root),
                "--dest-run-root",
                str(dest_run_root),
            ]
        )

    if failing_step == "luts":
        assert order == ["preflight", "luts"]
    elif failing_step == "candidates":
        assert order == ["preflight", "luts", "candidates"]
    else:
        assert order == ["preflight", "luts", "candidates", "replay"]


@pytest.mark.integration
def test_fail_fast_reaches_final_exact_proof_when_all_gates_pass(tmp_path, monkeypatch):
    source_run_root = tmp_path / "source-run"
    dest_run_root = tmp_path / "dest-run"
    order: list[str] = []
    monkeypatch.setattr(
        parity_experiment, "render_exact_preflight_report", lambda _report: "preflight"
    )
    monkeypatch.setattr(parity_experiment, "render_lut_proof_report", lambda _report: "luts")
    monkeypatch.setattr(
        parity_experiment, "render_candidate_coverage_report", lambda _report: "candidates"
    )
    monkeypatch.setattr(parity_experiment, "render_exact_proof_report", lambda _report: "replay")

    monkeypatch.setattr(
        parity_experiment,
        "_run_preflight_exact",
        lambda **_kwargs: (
            order.append("preflight")
            or (_pass_report(), dest_run_root / "a.json", dest_run_root / "a.txt")
        ),
    )
    monkeypatch.setattr(
        parity_experiment,
        "_run_prove_luts",
        lambda **_kwargs: (
            order.append("luts")
            or (_pass_report(), dest_run_root / "b.json", dest_run_root / "b.txt")
        ),
    )
    monkeypatch.setattr(
        parity_experiment,
        "_run_capture_candidates",
        lambda **_kwargs: (
            order.append("candidates")
            or (_pass_report(), {}, dest_run_root / "c.json", dest_run_root / "c.txt")
        ),
    )
    monkeypatch.setattr(
        parity_experiment,
        "_run_replay_edges",
        lambda **_kwargs: (
            order.append("replay")
            or (_pass_report(), dest_run_root / "d.json", dest_run_root / "d.txt")
        ),
    )
    monkeypatch.setattr(
        parity_experiment,
        "_handle_prove_exact",
        lambda _args: order.append("prove-exact"),
    )

    parity_experiment.main(
        [
            "fail-fast",
            "--source-run-root",
            str(source_run_root),
            "--dest-run-root",
            str(dest_run_root),
        ]
    )

    assert order == ["preflight", "luts", "candidates", "replay", "prove-exact"]
