"""Tests for the proof commands in the parity experiment runner."""

from __future__ import annotations

import importlib
import json

import numpy as np
import pytest
from slavv_python.analysis.parity import cli
from slavv_python.analysis.parity.constants import (
    CHECKPOINTS_DIR,
    EDGE_REPLAY_PROOF_JSON_PATH,
    EXACT_PROOF_JSON_PATH,
)
from workspace.tests.support.run_state_builders import materialize_checkpoint_surface

from .support import (
    _exact_edge_payload,
    _exact_energy_payload,
    _exact_network_payload,
    _exact_validated_params,
    _exact_vertex_payload,
    _fail_report,
    _materialize_exact_matlab_batch,
    _pass_report,
)

parity_experiment = importlib.import_module("workspace.scripts.cli.parity_experiment")


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
            "energy": _exact_energy_payload(),
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
    from .support import _write_json

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

    report_payload = json.loads((dest_run_root / EXACT_PROOF_JSON_PATH).read_text(encoding="utf-8"))
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
            "energy": _exact_energy_payload(),
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
    from .support import _write_json

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

    report_payload = json.loads((dest_run_root / EXACT_PROOF_JSON_PATH).read_text(encoding="utf-8"))
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
            "energy": _exact_energy_payload(),
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
    from .support import _write_json

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

    report_payload = json.loads((dest_run_root / EXACT_PROOF_JSON_PATH).read_text(encoding="utf-8"))
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
    from .support import _write_json

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

    assert (dest_run_root / CHECKPOINTS_DIR / "checkpoint_edges.pkl").is_file()
    report_payload = json.loads(
        (dest_run_root / EDGE_REPLAY_PROOF_JSON_PATH).read_text(encoding="utf-8")
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
        cli,
        "run_exact_preflight",
        lambda *args, **kwargs: (
            order.append("preflight")
            or (
                (_fail_report(expected_field) if failing_step == "preflight" else _pass_report()),
                dest_run_root / "a.json",
                dest_run_root / "a.txt",
            )
        ),
    )
    monkeypatch.setattr(
        cli,
        "run_lut_proof",
        lambda *args, **kwargs: (
            order.append("luts")
            or (
                (_fail_report(expected_field) if failing_step == "luts" else _pass_report()),
                dest_run_root / "b.json",
                dest_run_root / "b.txt",
            )
        ),
    )
    monkeypatch.setattr(
        cli,
        "run_candidate_capture",
        lambda *args, **kwargs: (
            order.append("candidates")
            or (
                (_fail_report(expected_field) if failing_step == "candidates" else _pass_report()),
                dest_run_root / "c.json",
                dest_run_root / "c.txt",
            )
        ),
    )
    monkeypatch.setattr(
        cli,
        "run_edge_replay",
        lambda *args, **kwargs: (
            order.append("replay")
            or (
                (_fail_report(expected_field) if failing_step == "replay" else _pass_report()),
                dest_run_root / "d.json",
                dest_run_root / "d.txt",
            )
        ),
    )
    monkeypatch.setattr(
        cli,
        "run_exact_parity_proof",
        lambda *args, **kwargs: (
            order.append("prove-exact")
            or (_pass_report(), dest_run_root / "e.json", dest_run_root / "e.txt")
        ),
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
        cli,
        "run_exact_preflight",
        lambda *args, **kwargs: (
            order.append("preflight")
            or (_pass_report(), dest_run_root / "a.json", dest_run_root / "a.txt")
        ),
    )
    monkeypatch.setattr(
        cli,
        "run_lut_proof",
        lambda *args, **kwargs: (
            order.append("luts")
            or (_pass_report(), dest_run_root / "b.json", dest_run_root / "b.txt")
        ),
    )
    monkeypatch.setattr(
        cli,
        "run_candidate_capture",
        lambda *args, **kwargs: (
            order.append("candidates")
            or (_pass_report(), dest_run_root / "c.json", dest_run_root / "c.txt")
        ),
    )
    monkeypatch.setattr(
        cli,
        "run_edge_replay",
        lambda *args, **kwargs: (
            order.append("replay")
            or (_pass_report(), dest_run_root / "d.json", dest_run_root / "d.txt")
        ),
    )
    monkeypatch.setattr(
        cli,
        "run_exact_parity_proof",
        lambda *args, **kwargs: (
            order.append("prove-exact")
            or (_pass_report(), dest_run_root / "e.json", dest_run_root / "e.txt")
        ),
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
