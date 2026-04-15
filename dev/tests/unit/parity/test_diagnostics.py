from __future__ import annotations

import json
from typing import TYPE_CHECKING

from slavv.parity.diagnostics import (
    NeighborhoodDivergence,
    SharedNeighborhoodDiagnosticReport,
    format_shared_neighborhood_summary,
    generate_shared_neighborhood_diagnostics,
    load_shared_neighborhood_diagnostics,
    recommend_diagnostics_if_needed,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write_comparison_report(run_root: Path) -> None:
    analysis_dir = run_root / "03_Analysis"
    analysis_dir.mkdir(parents=True)
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
                        {
                            "origin_index": 64,
                            "selection_sources": ["top_missing_seed_origin"],
                            "matlab_incident_endpoint_pair_count": 1,
                            "candidate_endpoint_pair_count": 0,
                            "final_chosen_endpoint_pair_count": 0,
                            "missing_matlab_incident_endpoint_pair_count": 1,
                            "extra_candidate_endpoint_pair_count": 0,
                            "missing_final_endpoint_pair_count": 1,
                            "missing_matlab_incident_endpoint_pair_samples": [[64, 8]],
                            "candidate_endpoint_pair_samples": [],
                            "first_divergence_stage": "candidate_admission_gap",
                            "first_divergence_reason": "missing candidate pair [64, 8]",
                        },
                    ]
                },
            },
        },
    }
    (analysis_dir / "comparison_report.json").write_text(json.dumps(payload), encoding="utf-8")


def test_diagnostic_dataclasses_are_serialization_friendly():
    divergence = NeighborhoodDivergence(
        neighborhood_id=1, divergence_type="claim_ordering", severity="medium"
    )
    report = SharedNeighborhoodDiagnosticReport(
        run_root="run",
        generated_at="2026-04-15T12:00:00",
        matlab_edges_count=4,
        python_edges_count=3,
        edge_count_delta=-1,
        claim_ordering_differences=1,
        branch_invalidation_differences=0,
        partner_choice_differences=0,
        divergent_neighborhoods=[divergence],
    )

    payload = report.to_dict()

    assert payload["divergent_neighborhoods"][0]["divergence_type"] == "claim_ordering"
    assert payload["divergent_neighborhoods"][0]["severity"] == "medium"


def test_generate_shared_neighborhood_diagnostics_persists_report(tmp_path: Path):
    run_root = tmp_path / "comparison_run"
    _write_comparison_report(run_root)

    report = generate_shared_neighborhood_diagnostics(run_root)

    json_path = run_root / "03_Analysis" / "shared_neighborhood_diagnostics.json"
    markdown_path = run_root / "03_Analysis" / "shared_neighborhood_diagnostics.md"
    assert json_path.exists()
    assert markdown_path.exists()
    assert report.branch_invalidation_differences == 1
    assert report.partner_choice_differences == 1
    assert report.claim_ordering_differences == 1
    assert report.coverage_delta["candidate_minus_matlab_pairs"] == -1
    assert "branch invalidation" in markdown_path.read_text(encoding="utf-8").lower()


def test_load_and_summarize_shared_neighborhood_report(tmp_path: Path):
    run_root = tmp_path / "comparison_run"
    _write_comparison_report(run_root)
    generate_shared_neighborhood_diagnostics(run_root)

    loaded = load_shared_neighborhood_diagnostics(run_root)

    assert loaded is not None
    assert "Top origins:" in format_shared_neighborhood_summary(loaded)


def test_recommend_diagnostics_mentions_network_gate_isolation(tmp_path: Path):
    run_root = tmp_path / "comparison_run"
    recommendation = recommend_diagnostics_if_needed(
        run_root=run_root,
        edges_parity_ok=False,
        network_gate_parity_ok=True,
    )

    assert recommendation is not None
    assert "isolated to edge candidate generation/selection" in recommendation
