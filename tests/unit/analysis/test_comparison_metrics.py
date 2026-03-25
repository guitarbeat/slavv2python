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
        positions=[[10, 20, 30, 1], [15, 25, 35, 2], [20, 30, 40, 3]],
        radii=[2.5, 3.0, 3.5],
    )
    python = _vertex_payload(
        positions=[[10.5, 20.5, 30.5, 9], [15.5, 25.5, 35.5, 9], [20.5, 30.5, 40.5, 9]],
        radii=[2.6, 3.1, 3.6],
    )

    result = compare_vertices(matlab, python)

    expected_dist = float(np.sqrt(0.5**2 * 3))
    assert result["matched_vertices"] == 3
    assert np.isclose(result["position_mean_distance"], expected_dist, rtol=0.01)
    assert np.isclose(result["position_rmse"], expected_dist, rtol=0.01)
    assert result["radius_correlation"] is not None
    assert "rmse" in result["radius_stats"]


def test_compare_vertices_reports_unmatched_counts():
    matlab = _vertex_payload(
        positions=[[0, 0, 0, 1], [1, 1, 1, 2], [100, 100, 100, 3]],
        radii=[1.0, 2.0, 3.0],
    )
    python = _vertex_payload(
        positions=[[0.1, 0.1, 0.1, 1], [1.1, 1.1, 1.1, 2]],
        radii=[1.1, 2.1],
    )

    result = compare_vertices(matlab, python)

    assert result["matched_vertices"] == 2
    assert result["unmatched_matlab"] == 1
    assert result["unmatched_python"] == 0
    assert result["count_difference"] == 1


def test_compare_vertices_tolerates_nan_radii():
    matlab = _vertex_payload(
        positions=[[10, 10, 10, 1], [20, 20, 20, 2]],
        radii=[2.5, np.nan],
    )
    python = _vertex_payload(
        positions=[[10, 10, 10, 1], [20, 20, 20, 2]],
        radii=[2.5, 3.0],
    )

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


def test_compare_networks_computes_strand_differences():
    matlab_stats = {"strand_count": 25}
    python_network = {"strands": [object() for _ in range(23)]}

    result = compare_networks(matlab_stats, python_network)

    assert result["matlab_strand_count"] == 25
    assert result["python_strand_count"] == 23
    assert result["strand_count_difference"] == 2


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
            "vertices": _vertex_payload([[0, 0, 0, 1], [1, 1, 1, 2]], [1.0, 2.0]),
            "edges": {"count": 1, "traces": [np.array([[0, 0, 0], [1, 0, 0]])]},
            "network": {"strands": [[0, 1]]},
        },
    }
    matlab_parsed = {
        "vertices": _vertex_payload([[0, 0, 0, 1], [1, 1, 1, 2]], [1.0, 2.0]),
        "edges": {"count": 1, "traces": [], "total_length": 1.0},
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


def test_compare_results_skips_nested_metrics_without_parsed_data():
    matlab_results = {"success": True, "elapsed_time": 5.0}
    python_results = {"success": True, "elapsed_time": 5.0, "results": {}}

    result = compare_results(matlab_results, python_results, matlab_parsed=None)

    assert "vertices" not in result
    assert "edges" not in result
    assert "network" not in result
