from __future__ import annotations

import json
from pathlib import Path

import pytest

from slavv.parity.proof_artifacts import (
    display_latest_proof_summary,
    generate_proof_artifact,
    maintain_proof_artifact_index,
)


def _write_network_gate_execution(run_root: Path, *, parity_achieved: bool = True) -> Path:
    metadata_dir = run_root / "99_Metadata"
    matlab_dir = run_root / "01_Input" / "matlab_results"
    python_checkpoints = run_root / "02_Output" / "python_results" / "checkpoints"
    metadata_dir.mkdir(parents=True)
    (matlab_dir / "batch_260415-180000").mkdir(parents=True)
    (matlab_dir / "energy.mat").write_bytes(b"energy")
    python_checkpoints.mkdir(parents=True)
    (python_checkpoints / "checkpoint_vertices.pkl").write_bytes(b"vertices")
    (python_checkpoints / "checkpoint_edges.pkl").write_bytes(b"edges")
    (python_checkpoints / "checkpoint_network.pkl").write_bytes(b"network")
    execution_path = metadata_dir / "network_gate_execution.json"
    payload = {
        "validation": {
            "matlab_edges_fingerprint": "sha256:matlab-edges",
            "matlab_vertices_fingerprint": "sha256:matlab-vertices",
        },
        "matlab_batch_folder": str(matlab_dir / "batch_260415-180000"),
        "started_at": "2026-04-15T18:00:00",
        "completed_at": "2026-04-15T18:02:00",
        "elapsed_seconds": 120.0,
        "comparison_exact_network_forced": True,
        "parity_achieved": parity_achieved,
        "vertices_match": True,
        "edges_match": True,
        "strands_match": True,
        "python_network_fingerprint": "sha256:python-network",
        "peak_memory_mb": 42.0,
        "cpu_time_seconds": 13.5,
    }
    execution_path.write_text(json.dumps(payload), encoding="utf-8")
    return execution_path


def test_generate_proof_artifact_persists_json_and_markdown(tmp_path: Path):
    run_root = tmp_path / "comparison_run"
    execution_path = _write_network_gate_execution(run_root)

    proof = generate_proof_artifact(execution_path, run_root=run_root)

    assert proof.overall_parity_achieved is True
    assert proof.matlab_batch_timestamp == "260415-180000"
    assert Path(proof.artifact_json_path).exists()
    assert Path(proof.artifact_markdown_path).exists()


def test_maintain_proof_artifact_index_rebuilds_from_existing_proofs(tmp_path: Path):
    run_root = tmp_path / "comparison_run"
    execution_path = _write_network_gate_execution(run_root)
    proof = generate_proof_artifact(execution_path, run_root=run_root)

    index = maintain_proof_artifact_index(run_root, proof)

    assert index.total_proofs == 1
    assert index.latest_proof is not None
    assert index.latest_proof["overall_parity_achieved"] is True
    index_path = run_root / "03_Analysis" / "proof_artifact_index.json"
    assert index_path.exists()


def test_display_latest_proof_summary_reports_known_artifacts(tmp_path: Path):
    run_root = tmp_path / "comparison_run"
    execution_path = _write_network_gate_execution(run_root)
    proof = generate_proof_artifact(execution_path, run_root=run_root)
    maintain_proof_artifact_index(run_root, proof)

    summary = display_latest_proof_summary(run_root)

    assert "Latest network-gate proof" in summary
    assert "Known proof artifacts" in summary
    assert proof.artifact_json_path in summary


def test_generate_proof_artifact_rejects_failed_network_gate(tmp_path: Path):
    run_root = tmp_path / "comparison_run"
    execution_path = _write_network_gate_execution(run_root, parity_achieved=False)

    with pytest.raises(ValueError, match="did not achieve exact parity"):
        generate_proof_artifact(execution_path, run_root=run_root)
