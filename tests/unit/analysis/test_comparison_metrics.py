"""Behavior-focused tests for evaluation metric comparisons."""

import importlib

import numpy as np

from slavv.evaluation.metrics import (
    compare_edges,
    compare_networks,
    compare_results,
    compare_vertices,
    match_vertices,
)


def _vertex_payload(positions, radii):
    positions = np.asarray(positions, dtype=float)
    radii = np.asarray(radii, dtype=float)
    return {"count": len(positions), "positions": positions, "radii": radii}


def test_evaluation_package_imports_without_missing_parser_dependency():
    evaluation = importlib.import_module("slavv.evaluation")
    assert evaluation.compare_results is compare_results


def test_match_vertices_handles_empty_inputs():
    matlab_idx, python_idx, distances = match_vertices(np.array([]), np.array([]))
    assert matlab_idx.size == 0
    assert python_idx.size == 0
    assert distances.size == 0


def test_match_vertices_uses_xyz_and_applies_threshold():
    matlab_pos = np.array(
        [
            [10.0, 10.0, 10.0, 1000.0],
            [100.0, 100.0, 100.0, 1.0],
        ]
    )
    python_pos = np.array(
        [
            [10.2, 10.2, 10.2, -1000.0],
            [150.0, 150.0, 150.0, 1.0],
        ]
    )

    matlab_idx, python_idx, distances = match_vertices(
        matlab_pos, python_pos, distance_threshold=1.0
    )

    assert matlab_idx.tolist() == [0]
    assert python_idx.tolist() == [0]
    assert distances[0] < 1.0


def test_compare_vertices_reports_matching_and_distance_stats():
    matlab = _vertex_payload(
        positions=[[10, 20, 30], [15, 25, 35], [20, 30, 40]],
        radii=[2.5, 3.0, 3.5],
    )
    matlab["scales"] = np.array([1, 2, 3])
    python = _vertex_payload(
        positions=[[10.5, 20.5, 30.5], [15.5, 25.5, 35.5], [20.5, 30.5, 40.5]],
        radii=[2.6, 3.1, 3.6],
    )
    python["scales"] = np.array([1, 2, 3])

    result = compare_vertices(matlab, python)

    expected_dist = float(np.sqrt(0.5**2 * 3))
    assert result["matched_vertices"] == 3
    assert np.isclose(result["position_mean_distance"], expected_dist, rtol=0.01)
    assert np.isclose(result["position_rmse"], expected_dist, rtol=0.01)
    assert result["radius_correlation"] is not None
    assert "rmse" in result["radius_stats"]


def test_compare_vertices_reports_unmatched_counts():
    matlab = _vertex_payload(
        positions=[[0, 0, 0], [1, 1, 1], [100, 100, 100]],
        radii=[1.0, 2.0, 3.0],
    )
    matlab["scales"] = np.array([1, 2, 3])
    python = _vertex_payload(
        positions=[[0.1, 0.1, 0.1], [1.1, 1.1, 1.1]],
        radii=[1.1, 2.1],
    )
    python["scales"] = np.array([1, 2])

    result = compare_vertices(matlab, python)

    assert result["matched_vertices"] == 2
    assert result["unmatched_matlab"] == 1
    assert result["unmatched_python"] == 0
    assert result["count_difference"] == 1


def test_compare_vertices_enforces_one_to_one_matching():
    matlab = _vertex_payload(
        positions=[[0, 0, 0], [0.2, 0.2, 0.2]],
        radii=[1.0, 3.0],
    )
    python = _vertex_payload(
        positions=[[0.1, 0.1, 0.1]],
        radii=[2.0],
    )

    result = compare_vertices(matlab, python)

    assert result["matched_vertices"] == 1
    assert result["unmatched_matlab"] == 1
    assert result["unmatched_python"] == 0
    assert result["radius_stats"]["rmse"] == 1.0


def test_compare_vertices_tolerates_nan_radii():
    matlab = _vertex_payload(
        positions=[[10, 10, 10], [20, 20, 20]],
        radii=[2.5, np.nan],
    )
    matlab["scales"] = np.array([1, 2])
    python = _vertex_payload(
        positions=[[10, 10, 10], [20, 20, 20]],
        radii=[2.5, 3.0],
    )
    python["scales"] = np.array([1, 2])

    result = compare_vertices(matlab, python)

    assert result["matched_vertices"] == 2
    assert "radius_stats" in result


def test_compare_edges_computes_python_total_length_and_percent_difference():
    matlab_edges = {"count": 2, "traces": [], "total_length": 10.0}
    python_edges = {
        "count": 2,
        "traces": [
            np.array([[0, 0, 0, 1], [3, 4, 0, 1]], dtype=float),  # length 5
            np.array([[0, 0, 0, 1], [0, 0, 5, 1]], dtype=float),  # length 5
        ],
    }

    result = compare_edges(matlab_edges, python_edges)

    assert result["count_difference"] == 0
    assert np.isclose(result["total_length"]["python"], 10.0)
    assert np.isclose(result["total_length"]["percent_difference"], 0.0)


def test_compare_edges_infers_count_and_length_from_json_style_traces():
    matlab_edges = {"count": 1, "total_length": 5.0}
    python_edges = {"traces": [[[0, 0, 0], [3, 4, 0]]]}

    result = compare_edges(matlab_edges, python_edges)

    assert result["python_count"] == 1
    assert np.isclose(result["total_length"]["python"], 5.0)


def test_compare_edges_reports_exact_endpoint_and_diagnostic_bundle():
    matlab_edges = {
        "connections": np.array([[0, 1], [1, 2]], dtype=np.int32),
        "traces": [
            np.array([[0, 0, 0], [0, 1, 0]], dtype=float),
            np.array([[0, 1, 0], [0, 2, 0]], dtype=float),
        ],
    }
    python_edges = {
        "connections": np.array([[0, 1], [1, 3]], dtype=np.int32),
        "traces": [
            np.array([[0, 0, 0], [0, 1, 0]], dtype=float),
            np.array([[0, 1, 0], [0, 3, 0]], dtype=float),
        ],
        "diagnostics": {"candidate_traced_edge_count": 7, "chosen_edge_count": 2},
    }

    result = compare_edges(matlab_edges, python_edges)

    assert not result["exact_endpoint_pairs_match"]
    assert result["matched_endpoint_pair_count"] == 1
    assert result["missing_endpoint_pair_count"] == 1
    assert result["extra_endpoint_pair_count"] == 1
    assert result["endpoint_pair_matlab_only_samples"]
    assert result["endpoint_pair_python_only_samples"]
    assert result["diagnostics"]["python"]["candidate_traced_edge_count"] == 7


def test_compare_edges_reports_candidate_endpoint_coverage():
    matlab_edges = {
        "connections": np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int32),
        "traces": [],
    }
    python_edges = {
        "connections": np.array([[0, 1], [1, 3]], dtype=np.int32),
        "traces": [
            np.array([[0, 0, 0], [0, 1, 0]], dtype=float),
            np.array([[0, 1, 0], [0, 2, 0], [0, 3, 0]], dtype=float),
        ],
        "energies": np.array([-10.0, -5.0], dtype=np.float32),
        "chosen_candidate_indices": np.array([0, 2], dtype=np.int32),
    }
    candidate_edges = {
        "connections": np.array([[0, 1], [1, 2], [4, 5]], dtype=np.int32),
        "connection_sources": ["frontier", "frontier", "watershed"],
        "traces": [],
    }

    candidate_audit = {
        "schema_version": 1,
        "vertex_count": 7,
        "use_frontier_tracer": True,
        "candidate_connection_count": 3,
        "candidate_origin_count": 4,
        "source_breakdown": {
            "frontier": {
                "candidate_connection_count": 1,
                "candidate_origin_count": 1,
                "candidate_endpoint_pair_count": 2,
                "candidate_endpoint_pair_samples": [(0, 1), (1, 2)],
            },
            "watershed": {
                "candidate_connection_count": 2,
                "candidate_origin_count": 1,
                "candidate_endpoint_pair_count": 1,
                "candidate_endpoint_pair_samples": [(4, 5)],
            },
            "fallback": {
                "candidate_connection_count": 0,
                "candidate_origin_count": 0,
                "candidate_endpoint_pair_count": 0,
                "candidate_endpoint_pair_samples": [],
            },
        },
        "frontier_per_origin_candidate_counts": {0: 1},
        "watershed_per_origin_candidate_counts": {4: 1, 5: 1},
        "per_origin_summary": [
            {
                "origin_index": 4,
                "frontier_candidate_count": 0,
                "watershed_candidate_count": 1,
                "fallback_candidate_count": 0,
                "candidate_connection_count": 1,
            },
            {
                "origin_index": 5,
                "frontier_candidate_count": 0,
                "watershed_candidate_count": 1,
                "fallback_candidate_count": 0,
                "candidate_connection_count": 1,
            },
            {
                "origin_index": 0,
                "frontier_candidate_count": 1,
                "watershed_candidate_count": 0,
                "fallback_candidate_count": 0,
                "candidate_connection_count": 1,
            },
        ],
        "diagnostic_counters": {
            "watershed_reachability_rejected": 3,
            "watershed_energy_rejected": 1,
            "watershed_cap_rejected": 0,
            "watershed_short_trace_rejected": 0,
            "watershed_accepted": 2,
        },
    }

    result = compare_edges(matlab_edges, python_edges, candidate_edges, candidate_audit)

    coverage = result["diagnostics"]["candidate_endpoint_coverage"]
    assert coverage["candidate_endpoint_pair_count"] == 3
    assert coverage["matlab_endpoint_pair_count"] == 3
    assert coverage["python_endpoint_pair_count"] == 2
    assert coverage["matched_matlab_endpoint_pair_count"] == 2
    assert coverage["missing_matlab_endpoint_pair_count"] == 1
    assert coverage["extra_candidate_endpoint_pair_count"] == 1
    assert not coverage["matlab_pairs_fully_covered"]
    assert coverage["missing_matlab_endpoint_pair_samples"] == [(2, 3)]
    assert coverage["extra_candidate_endpoint_pair_samples"] == [(4, 5)]
    assert coverage["frontier_only_candidate_endpoint_pair_count"] == 2
    assert coverage["watershed_only_candidate_endpoint_pair_count"] == 1
    assert coverage["fallback_only_candidate_endpoint_pair_count"] == 0
    assert coverage["watershed_only_candidate_endpoint_pair_samples"] == [(4, 5)]
    assert coverage["supplement_candidate_endpoint_pair_samples"] == [(4, 5)]
    assert coverage["extra_candidate_seed_origin_count"] == 1
    assert coverage["extra_candidate_seed_origin_samples"][0]["seed_origin_index"] == 4
    assert coverage["extra_candidate_seed_origin_samples"][0][
        "extra_candidate_endpoint_pair_samples"
    ] == [(4, 5)]
    chosen_sources = result["diagnostics"]["chosen_candidate_sources"]
    assert chosen_sources["counts"] == {"frontier": 1, "watershed": 1, "fallback": 0}
    assert chosen_sources["watershed_endpoint_pair_count"] == 1
    assert chosen_sources["watershed_matched_matlab_endpoint_pair_count"] == 0
    assert chosen_sources["watershed_extra_python_endpoint_pair_count"] == 1
    assert chosen_sources["source_breakdown"]["frontier"]["matched_matlab_edge_count"] == 1
    assert chosen_sources["source_breakdown"]["frontier"]["extra_python_edge_count"] == 0
    assert chosen_sources["source_breakdown"]["frontier"]["matched"] == {
        "edge_count": 1,
        "median_energy": -10.0,
        "median_length": 2.0,
    }
    assert chosen_sources["source_breakdown"]["watershed"]["matched_matlab_edge_count"] == 0
    assert chosen_sources["source_breakdown"]["watershed"]["extra_python_edge_count"] == 1
    assert chosen_sources["source_breakdown"]["watershed"]["extra"] == {
        "edge_count": 1,
        "median_energy": -5.0,
        "median_length": 3.0,
    }
    assert "candidate_audit" in result["diagnostics"]
    assert result["diagnostics"]["candidate_audit"]["schema_version"] == 1
    assert result["diagnostics"]["candidate_audit"]["candidate_connection_count"] == 3
    assert result["diagnostics"]["candidate_audit"]["candidate_origin_count"] == 4


def test_compare_edges_reports_missing_matlab_pairs_by_seed_origin():
    matlab_edges = {
        "connections": np.array([[0, 1], [0, 2], [0, 3], [4, 5]], dtype=np.int32),
        "traces": [],
    }
    python_edges = {
        "connections": np.array([[0, 1], [4, 5]], dtype=np.int32),
        "traces": [],
    }
    candidate_edges = {
        "connections": np.array([[0, 1], [0, 4], [4, 5]], dtype=np.int32),
        "origin_indices": np.array([0, 0, 4], dtype=np.int32),
        "connection_sources": ["frontier", "watershed", "fallback"],
        "traces": [],
    }

    result = compare_edges(matlab_edges, python_edges, candidate_edges)

    coverage = result["diagnostics"]["candidate_endpoint_coverage"]
    assert coverage["missing_matlab_seed_origin_count"] == 3
    top_seed_origin = coverage["missing_matlab_seed_origin_samples"][0]
    assert top_seed_origin["seed_origin_index"] == 0
    assert top_seed_origin["matlab_incident_endpoint_pair_count"] == 3
    assert top_seed_origin["missing_matlab_incident_endpoint_pair_count"] == 2
    assert top_seed_origin["matched_matlab_incident_endpoint_pair_count"] == 1
    assert top_seed_origin["candidate_endpoint_pair_count"] == 2
    assert top_seed_origin["extra_candidate_endpoint_pair_count"] == 1
    assert top_seed_origin["frontier_candidate_endpoint_pair_count"] == 1
    assert top_seed_origin["watershed_candidate_endpoint_pair_count"] == 1
    assert top_seed_origin["missing_matlab_incident_endpoint_pair_samples"] == [(0, 2), (0, 3)]
    assert top_seed_origin["candidate_endpoint_pair_samples"] == [(0, 1), (0, 4)]
    assert top_seed_origin["extra_candidate_endpoint_pair_samples"] == [(0, 4)]

    seed_origin_two = next(
        sample
        for sample in coverage["missing_matlab_seed_origin_samples"]
        if sample["seed_origin_index"] == 2
    )
    assert seed_origin_two["candidate_endpoint_pair_count"] == 0
    assert seed_origin_two["missing_matlab_incident_endpoint_pair_samples"] == [(0, 2)]


def test_compare_edges_reports_extra_frontier_missing_vertex_overlap():
    matlab_edges = {
        "connections": np.array([[0, 1], [0, 2], [3, 4], [5, 6]], dtype=np.int32),
        "traces": [],
    }
    python_edges = {
        "connections": np.array([[0, 1], [0, 7], [8, 9]], dtype=np.int32),
        "traces": [
            np.array([[0, 0, 0], [0, 1, 0]], dtype=float),
            np.array([[0, 0, 0], [0, 2, 0], [0, 3, 0]], dtype=float),
            np.array([[1, 1, 1], [1, 2, 2]], dtype=float),
        ],
        "energies": np.array([-10.0, -20.0, -5.0], dtype=np.float32),
        "connection_sources": ["frontier", "frontier", "watershed"],
    }
    candidate_edges = {
        "connections": np.array([[0, 1], [0, 7], [8, 9]], dtype=np.int32),
        "connection_sources": ["frontier", "frontier", "watershed"],
        "traces": [],
    }

    result = compare_edges(matlab_edges, python_edges, candidate_edges)

    overlap = result["diagnostics"]["extra_frontier_missing_vertex_overlap"]
    assert overlap["extra_frontier_edge_count"] == 1
    assert overlap["shared_missing_vertex_edge_count"] == 1
    assert overlap["missing_matlab_pair_count"] == 3
    assert overlap["shared_vertex_count"] == 1
    assert overlap["top_strength_overlap_counts"]["20"] == {
        "threshold": 20,
        "shared_missing_vertex_count": 1,
        "evaluated_edge_count": 1,
    }
    assert overlap["top_shared_vertices"][0]["vertex_index"] == 0
    assert overlap["top_shared_vertices"][0]["missing_matlab_endpoint_pair_count"] == 1
    assert overlap["top_shared_vertices"][0]["extra_frontier_endpoint_pair_count"] == 1
    assert overlap["top_shared_vertices"][0]["missing_matlab_pairs_present_in_candidates"] == 0
    assert overlap["top_shared_vertices"][0][
        "frontier_candidate_incident_endpoint_pair_count"
    ] == 2
    assert overlap["top_shared_vertices"][0]["chosen_frontier_incident_endpoint_pair_count"] == 2
    assert overlap["strongest_extra_frontier_samples"][0]["pair"] == (0, 7)
    assert overlap["strongest_extra_frontier_samples"][0]["shares_missing_vertex"] is True


def test_compare_edges_uses_candidate_audit_source_pair_counts_when_payload_lacks_sources():
    matlab_edges = {"connections": np.array([[0, 1], [1, 2]], dtype=np.int32), "traces": []}
    python_edges = {"connections": np.array([[0, 1]], dtype=np.int32), "traces": []}
    candidate_edges = {"connections": np.array([[0, 1], [4, 5]], dtype=np.int32), "traces": []}
    candidate_audit = {
        "schema_version": 1,
        "source_breakdown": {
            "frontier": {
                "candidate_endpoint_pair_count": 1,
                "candidate_endpoint_pair_samples": [(0, 1)],
            },
            "watershed": {
                "candidate_endpoint_pair_count": 1,
                "candidate_endpoint_pair_samples": [(4, 5)],
            },
            "fallback": {
                "candidate_endpoint_pair_count": 0,
                "candidate_endpoint_pair_samples": [],
            },
        },
    }

    result = compare_edges(matlab_edges, python_edges, candidate_edges, candidate_audit)

    coverage = result["diagnostics"]["candidate_endpoint_coverage"]
    assert coverage["frontier_candidate_endpoint_pair_count"] == 1
    assert coverage["supplement_candidate_endpoint_pair_count"] == 1
    assert coverage["supplement_candidate_endpoint_pair_samples"] == [(4, 5)]


def test_compare_networks_computes_strand_differences():
    matlab_stats = {"strand_count": 25}
    python_network = {"strands": [object() for _ in range(23)]}

    result = compare_networks(matlab_stats, python_network)

    assert result["matlab_strand_count"] == 25
    assert result["python_strand_count"] == 23
    assert result["strand_count_difference"] == 2


def test_compare_networks_infers_matlab_strands_without_stats_block():
    matlab_network = {"strands": [[0, 1], [1, 2]]}
    python_network = {"strands": [[0, 1], [1, 2]]}

    result = compare_networks(matlab_network, python_network)

    assert result["matlab_strand_count"] == 2
    assert result["python_strand_count"] == 2
    assert result["exact_match"]


def test_compare_results_infers_matlab_strand_count_from_network_without_stats():
    matlab_results = {"success": True, "elapsed_time": 5.0}
    python_results = {
        "success": True,
        "elapsed_time": 4.0,
        "results": {"network": {"strands": [[0, 1], [1, 2]]}},
    }
    matlab_parsed = {
        "network": {"strands": [[0, 1], [1, 2]]},
    }

    result = compare_results(matlab_results, python_results, matlab_parsed)

    assert result["matlab"]["strand_count"] == 2
    assert result["network"]["exact_match"]
    assert result["parity_gate"]["strands_exact"] is True


def test_compare_results_includes_performance_and_nested_metrics():
    matlab_results = {"success": True, "elapsed_time": 20.0, "output_dir": "m_out"}
    python_results = {
        "success": True,
        "elapsed_time": 10.0,
        "output_dir": "p_out",
        "vertices_count": 2,
        "edges_count": 1,
        "network_strands_count": 1,
        "results": {
            "vertices": {
                **_vertex_payload([[0, 0, 0], [1, 1, 1]], [1.0, 2.0]),
                "scales": np.array([1, 2]),
            },
            "edges": {
                "count": 1,
                "connections": np.array([[0, 1]]),
                "traces": [np.array([[0, 0, 0], [1, 0, 0]])],
            },
            "network": {"strands": [[0, 1]]},
        },
    }
    matlab_parsed = {
        "vertices": {
            **_vertex_payload([[0, 0, 0], [1, 1, 1]], [1.0, 2.0]),
            "scale_indices": np.array([1, 2]),
        },
        "edges": {"count": 1, "connections": np.array([[0, 1]]), "traces": [], "total_length": 1.0},
        "network": {"strands_to_vertices": [[0, 1]]},
        "network_stats": {"strand_count": 1},
    }

    result = compare_results(matlab_results, python_results, matlab_parsed)

    assert result["performance"]["speedup"] == 2.0
    assert result["performance"]["faster"] == "Python"
    assert result["matlab"]["vertices_count"] == 2
    assert result["matlab"]["edges_count"] == 1
    assert result["matlab"]["strand_count"] == 1
    assert "vertices" in result
    assert "edges" in result
    assert "network" in result
    assert result["parity_gate"]["passed"]


def test_compare_results_infers_missing_top_level_counts_from_payloads():
    matlab_results = {"success": True, "elapsed_time": 9.0, "output_dir": "m_out"}
    python_results = {
        "success": True,
        "elapsed_time": 3.0,
        "output_dir": "p_out",
        "results": {
            "vertices": {"positions": np.array([[0, 0, 0], [1, 1, 1]])},
            "edges": {"connections": [[0, 1], [1, 0]]},
            "network": {"strands": [[0, 1], [1, 2]]},
        },
    }
    matlab_parsed = {
        "vertices": {"positions": np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])},
        "edges": {"connections": np.array([[0, 1]])},
        "network": {"strands_to_vertices": [[0, 1], [1, 2], [2, 3], [3, 4]]},
        "network_stats": {"strand_count": 4},
    }

    result = compare_results(matlab_results, python_results, matlab_parsed)

    assert result["matlab"]["vertices_count"] == 3
    assert result["matlab"]["edges_count"] == 1
    assert result["matlab"]["strand_count"] == 4
    assert result["python"]["vertices_count"] == 2
    assert result["python"]["edges_count"] == 2
    assert result["python"]["network_strands_count"] == 2
    assert result["vertices"]["python_count"] == 2
    assert result["edges"]["python_count"] == 2


def test_compare_results_reports_exact_mismatch_samples():
    matlab_results = {"success": True, "elapsed_time": 5.0}
    python_results = {
        "success": True,
        "elapsed_time": 4.0,
        "results": {
            "vertices": {"positions": np.array([[0, 0, 0]]), "scales": np.array([1])},
            "edges": {
                "connections": np.array([[0, 1]]),
                "traces": [np.array([[0, 0, 0], [0, 0, 1]])],
            },
            "network": {"strands": [[0, 1]]},
        },
    }
    matlab_parsed = {
        "vertices": {"positions": np.array([[0, 0, 1]]), "scale_indices": np.array([1])},
        "edges": {
            "connections": np.array([[0, 1]]),
            "traces": [np.array([[0, 0, 0], [0, 1, 0]])],
        },
        "network": {"strands_to_vertices": [[0, 2]]},
        "network_stats": {"strand_count": 1},
    }

    result = compare_results(matlab_results, python_results, matlab_parsed)

    assert not result["parity_gate"]["passed"]
    assert result["vertices"]["matlab_only_samples"]
    assert result["edges"]["matlab_only_samples"]
    assert result["network"]["matlab_only_samples"]


def test_compare_results_surfaces_candidate_pair_coverage_diagnostics():
    matlab_results = {"success": True, "elapsed_time": 5.0}
    python_results = {
        "success": True,
        "elapsed_time": 4.0,
        "results": {
            "vertices": {"positions": np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])},
            "edges": {
                "connections": np.array([[0, 1], [1, 3]], dtype=np.int32),
                "traces": [],
            },
            "candidate_edges": {
                "connections": np.array([[0, 1], [1, 2], [4, 5]], dtype=np.int32),
                "traces": [],
            },
            "network": {"strands": []},
        },
    }
    matlab_parsed = {
        "vertices": {"positions": np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])},
        "edges": {
            "connections": np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int32),
            "traces": [],
        },
        "network": {"strands_to_vertices": []},
        "network_stats": {"strand_count": 0},
    }

    result = compare_results(matlab_results, python_results, matlab_parsed)

    coverage = result["edges"]["diagnostics"]["candidate_endpoint_coverage"]
    assert coverage["matched_matlab_endpoint_pair_count"] == 2
    assert coverage["missing_matlab_endpoint_pair_count"] == 1
    assert coverage["missing_matlab_endpoint_pair_samples"] == [(2, 3)]
    assert not coverage["matlab_pairs_fully_covered"]
    assert not result["parity_gate"]["passed"]


def test_compare_results_skips_nested_metrics_without_parsed_data():
    matlab_results = {"success": True, "elapsed_time": 5.0}
    python_results = {"success": True, "elapsed_time": 5.0, "results": {}}

    result = compare_results(matlab_results, python_results, matlab_parsed=None)

    assert "vertices" not in result
    assert "edges" not in result
    assert "network" not in result
    assert result["parity_gate"]["vertices_exact"] is None
    assert result["parity_gate"]["edges_exact"] is None
    assert result["parity_gate"]["strands_exact"] is None
    assert result["parity_gate"]["passed"] is None
