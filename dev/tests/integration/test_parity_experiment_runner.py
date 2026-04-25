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


def _cell(items: list[np.ndarray]) -> np.ndarray:
    cell = np.empty((len(items),), dtype=object)
    for index, item in enumerate(items):
        cell[index] = item
    return cell


def _materialize_exact_matlab_batch(run_root: Path) -> Path:
    batch_dir = run_root / "01_Input" / "matlab_results" / "batch_260421-151654"
    vectors_dir = batch_dir / "vectors"
    vectors_dir.mkdir(parents=True, exist_ok=True)
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
        "positions": np.array([[2.0, 1.0, 0.0], [5.0, 4.0, 3.0]], dtype=np.float32),
        "scales": np.array([1, 2], dtype=np.int16),
        "energies": np.array([-2.0, -1.0], dtype=np.float32),
    }


def _exact_edge_payload(*, energies: np.ndarray | None = None) -> dict[str, object]:
    return {
        "connections": np.array([[0, 1]], dtype=np.int32),
        "traces": [np.array([[2.0, 1.0, 0.0], [5.0, 4.0, 3.0]], dtype=np.float32)],
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
                    [2.0, 1.0, 0.0, 1.0],
                    [5.0, 4.0, 3.0, 1.5],
                ],
                dtype=np.float32,
            )
        ],
        "strand_energy_traces": [np.array([-4.0, -3.0], dtype=np.float32)],
        "mean_strand_energies": np.array([-3.5], dtype=np.float32),
        "vessel_directions": [np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32)],
    }


def _pass_report() -> dict[str, object]:
    return {"passed": True, "first_failure": None}


def _fail_report(field_path: str) -> dict[str, object]:
    return {
        "passed": False,
        "first_failure": {"field_path": field_path},
    }


@pytest.mark.integration
def test_rerun_python_creates_fresh_dest_root_and_writes_summary(tmp_path, monkeypatch):
    source_run_root = tmp_path / "source-run"
    dest_run_root = tmp_path / "dest-run"
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
    assert (dest_run_root / "99_Metadata" / "source_comparison_report.json").is_file()
    assert (dest_run_root / "99_Metadata" / "source_validated_params.json").is_file()


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
        {"comparison_exact_network": True},
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
                np.array([[2.0, 1.0, 0.0], [5.0, 4.0, 3.0]], dtype=np.float32),
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
                    traces=[[[2.0, 1.0, 0.0], [5.0, 4.0, 3.0]]], connections=[[0, 1]]
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
        {"comparison_exact_network": True},
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
    assert report_payload["exact_route_gate"].endswith("(canonical: python_native_hessian)")


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
        {"comparison_exact_network": True},
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
                "traces": [np.array([[2.0, 1.0, 0.0], [5.0, 4.0, 3.0]], dtype=np.float32)],
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
        {"comparison_exact_network": True},
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
                "positions": np.array([[2.0, 1.0, 0.0], [5.0, 4.0, 3.0]], dtype=np.float32),
                "scales": np.array([1, 2], dtype=np.int16),
                "energies": np.array([-2.0, -1.0], dtype=np.float32),
                "count": 2,
            },
        },
    )
    _write_json(
        source_run_root / "99_Metadata" / "validated_params.json",
        {"comparison_exact_network": True, "microns_per_voxel": [1.0, 1.0, 1.0]},
    )
    materialize_checkpoint_surface(
        dest_run_root,
        stages=("edge_candidates",),
        payloads={
            "edge_candidates": {
                "connections": np.array([[0, 1]], dtype=np.int32),
                "traces": [np.array([[2.0, 1.0, 0.0], [5.0, 4.0, 3.0]], dtype=np.float32)],
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
