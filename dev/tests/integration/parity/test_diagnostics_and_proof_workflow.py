from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from slavv.parity.diagnostics import (
    format_shared_neighborhood_summary,
    generate_shared_neighborhood_diagnostics,
    load_shared_neighborhood_diagnostics,
    recommend_diagnostics_if_needed,
)
from slavv.parity.proof_artifacts import (
    display_latest_proof_summary,
    generate_proof_artifact,
    maintain_proof_artifact_index,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write_comparison_report(run_root: Path) -> None:
    analysis_dir = run_root / "03_Analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "matlab": {"edges_count": 5},
        "python": {"edges_count": 3},
        "edges": {
            "matlab_count": 5,
            "python_count": 3,
            "exact_match": False,
            "diagnostics": {
                "candidate_endpoint_coverage": {
                    "matlab_endpoint_pair_count": 5,
                    "matched_matlab_endpoint_pair_count": 2,
                    "missing_matlab_endpoint_pair_count": 3,
                    "candidate_endpoint_pair_count": 4,
                    "python_endpoint_pair_count": 3,
                    "extra_candidate_endpoint_pair_count": 1,
                },
                "shared_neighborhood_audit": {
                    "neighborhoods": [
                        {
                            "origin_index": 866,
                            "selection_sources": ["tracked_hotspot"],
                            "matlab_incident_endpoint_pair_count": 4,
                            "candidate_endpoint_pair_count": 1,
                            "final_chosen_endpoint_pair_count": 1,
                            "missing_matlab_incident_endpoint_pair_count": 3,
                            "extra_candidate_endpoint_pair_count": 0,
                            "missing_final_endpoint_pair_count": 3,
                            "missing_matlab_incident_endpoint_pair_samples": [[866, 10]],
                            "candidate_endpoint_pair_samples": [[866, 22]],
                            "first_divergence_stage": "pre_manifest_rejection",
                            "first_divergence_reason": "rejected_parent_has_child at terminal 1023",
                        },
                        {
                            "origin_index": 359,
                            "selection_sources": ["top_extra_seed_origin"],
                            "matlab_incident_endpoint_pair_count": 2,
                            "candidate_endpoint_pair_count": 2,
                            "final_chosen_endpoint_pair_count": 0,
                            "missing_matlab_incident_endpoint_pair_count": 0,
                            "extra_candidate_endpoint_pair_count": 1,
                            "missing_final_endpoint_pair_count": 2,
                            "missing_matlab_incident_endpoint_pair_samples": [],
                            "candidate_endpoint_pair_samples": [[359, 44]],
                            "first_divergence_stage": "final_cleanup_loss",
                            "first_divergence_reason": "emitted [359, 44] but it was not retained",
                        },
                    ]
                },
            },
        },
    }
    (analysis_dir / "comparison_report.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_network_gate_execution(run_root: Path, *, parity_achieved: bool = True) -> Path:
    metadata_dir = run_root / "99_Metadata"
    matlab_dir = run_root / "01_Input" / "matlab_results"
    python_checkpoints = run_root / "02_Output" / "python_results" / "checkpoints"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (matlab_dir / "batch_260415-180000").mkdir(parents=True, exist_ok=True)
    (matlab_dir / "energy.mat").write_bytes(b"energy")
    python_checkpoints.mkdir(parents=True, exist_ok=True)
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
    }
    execution_path.write_text(json.dumps(payload), encoding="utf-8")
    return execution_path


def test_diagnostics_integration_recommendation_and_persistence(tmp_path: Path):
    run_root = tmp_path / "comparison_run"
    _write_comparison_report(run_root)

    recommendation = recommend_diagnostics_if_needed(
        run_root=run_root,
        edges_parity_ok=False,
        network_gate_parity_ok=True,
    )
    assert recommendation is not None
    assert "shared-neighborhood diagnostics" in recommendation

    report = generate_shared_neighborhood_diagnostics(run_root)
    loaded = load_shared_neighborhood_diagnostics(run_root)

    assert report.branch_invalidation_differences >= 1
    assert loaded is not None
    assert "Top origins:" in format_shared_neighborhood_summary(loaded)
    assert (run_root / "03_Analysis" / "shared_neighborhood_diagnostics.json").exists()
    assert (run_root / "03_Analysis" / "shared_neighborhood_diagnostics.md").exists()


def test_proof_artifact_integration_success_and_failure_paths(tmp_path: Path):
    success_root = tmp_path / "success_run"
    success_execution = _write_network_gate_execution(success_root, parity_achieved=True)

    proof = generate_proof_artifact(success_execution, run_root=success_root)
    index = maintain_proof_artifact_index(success_root, proof)
    summary = display_latest_proof_summary(success_root)

    assert index.total_proofs == 1
    assert proof.artifact_json_path in summary
    assert (success_root / "03_Analysis" / "proof_artifact_index.json").exists()

    failed_root = tmp_path / "failed_run"
    failed_execution = _write_network_gate_execution(failed_root, parity_achieved=False)
    with pytest.raises(ValueError, match="did not achieve exact parity"):
        generate_proof_artifact(failed_execution, run_root=failed_root)


def test_workflow_level_artifacts_share_canonical_layout(tmp_path: Path):
    run_root = tmp_path / "workflow_run"
    _write_comparison_report(run_root)
    report = generate_shared_neighborhood_diagnostics(run_root)
    execution_path = _write_network_gate_execution(run_root, parity_achieved=True)
    proof = generate_proof_artifact(execution_path, run_root=run_root)
    maintain_proof_artifact_index(run_root, proof)

    assert report.run_root.endswith("workflow_run")
    assert (run_root / "03_Analysis" / "shared_neighborhood_diagnostics.json").exists()
    assert (run_root / "03_Analysis" / "proof_artifact_index.json").exists()
    assert (run_root / "99_Metadata" / "network_gate_execution.json").exists()
