from __future__ import annotations

from slavv.core.edge_candidates import _build_frontier_candidate_lifecycle


def test_frontier_lifecycle_marks_manifest_survivor_as_cleanup_loss_when_alternate_edge_wins():
    lifecycle = _build_frontier_candidate_lifecycle(
        {
            "frontier_lifecycle_events": [
                {
                    "seed_origin_index": 1283,
                    "terminal_vertex_index": 1319,
                    "resolved_origin_index": 1283,
                    "resolved_terminal_index": 1319,
                    "emitted_endpoint_pair": [1283, 1319],
                    "resolution_reason": "accepted_seed_origin",
                    "rejection_reason": None,
                    "parent_child_outcome": None,
                    "bifurcation_choice": None,
                    "claim_reassigned": False,
                    "claim_reassignment_reason": None,
                    "survived_candidate_manifest": True,
                    "manifest_candidate_index": 0,
                    "chosen_final_edge": False,
                    "terminal_hit_sequence": 1,
                },
                {
                    "seed_origin_index": 1283,
                    "terminal_vertex_index": 1659,
                    "resolved_origin_index": 1283,
                    "resolved_terminal_index": 1659,
                    "emitted_endpoint_pair": [1283, 1659],
                    "resolution_reason": "accepted_seed_origin",
                    "rejection_reason": None,
                    "parent_child_outcome": None,
                    "bifurcation_choice": None,
                    "claim_reassigned": False,
                    "claim_reassignment_reason": None,
                    "survived_candidate_manifest": True,
                    "manifest_candidate_index": 1,
                    "chosen_final_edge": True,
                    "terminal_hit_sequence": 2,
                },
            ]
        },
        chosen_candidate_indices=[1],
    )

    events = lifecycle["events"]
    assert events[0]["emitted_endpoint_pair"] == [1283, 1319]
    assert events[0]["chosen_final_edge"] is False
    assert events[0]["final_survival_stage"] == "final_cleanup_dropped"

    assert events[1]["emitted_endpoint_pair"] == [1283, 1659]
    assert events[1]["chosen_final_edge"] is True
    assert events[1]["final_survival_stage"] == "final_edge_retained"

    summary = lifecycle["per_origin_summary"][0]
    assert summary["origin_index"] == 1283
    assert summary["terminal_hit_count"] == 2
    assert summary["emitted_candidate_count"] == 2
    assert summary["chosen_final_edge_count"] == 1
    assert summary["final_cleanup_loss_count"] == 1
    assert summary["emitted_endpoint_pair_samples"] == [[1283, 1319], [1283, 1659]]
