from __future__ import annotations

from typing import Any

from .counts import _infer_edges_count, _infer_strand_count, _infer_vertices_count, _resolve_count
from .edges import compare_edges
from .network import compare_networks
from .vertices import compare_vertices


def compare_results(
    matlab_results: dict[str, Any],
    python_results: dict[str, Any],
    matlab_parsed: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare MATLAB and Python vectorization results."""
    python_data = python_results.get("results") or {}
    matlab_vertices_count = _resolve_count(
        matlab_results.get("vertices_count"),
        _infer_vertices_count((matlab_parsed or {}).get("vertices", {})),
    )
    matlab_edges_count = _resolve_count(
        matlab_results.get("edges_count"),
        _infer_edges_count((matlab_parsed or {}).get("edges", {})),
    )
    matlab_strands_count = _resolve_count(
        matlab_results.get("strand_count"),
        _resolve_count(
            matlab_results.get("network_strands_count"),
            _resolve_count(
                (matlab_parsed or {}).get("network_stats", {}).get("strand_count"),
                _infer_strand_count((matlab_parsed or {}).get("network", {})),
            ),
        ),
    )
    python_vertices_count = _resolve_count(
        python_results.get("vertices_count"),
        _infer_vertices_count(python_data.get("vertices", {})),
    )
    python_edges_count = _resolve_count(
        python_results.get("edges_count"),
        _infer_edges_count(python_data.get("edges", {})),
    )
    python_strands_count = _resolve_count(
        python_results.get("network_strands_count"),
        _infer_strand_count(python_data.get("network", {})),
    )

    comparison = {
        "matlab": {
            "success": matlab_results.get("success", False),
            "elapsed_time": matlab_results.get("elapsed_time", 0.0),
            "output_dir": matlab_results.get("output_dir", ""),
            "vertices_count": matlab_vertices_count,
            "edges_count": matlab_edges_count,
            "strand_count": matlab_strands_count,
        },
        "python": {
            "success": python_results.get("success", False),
            "elapsed_time": python_results.get("elapsed_time", 0.0),
            "output_dir": python_results.get("output_dir", ""),
            "vertices_count": python_vertices_count,
            "edges_count": python_edges_count,
            "network_strands_count": python_strands_count,
            "comparison_mode": python_results.get("comparison_mode", {}),
        },
        "performance": {},
    }

    matlab_time = matlab_results.get("elapsed_time", 0.0)
    python_time = python_results.get("elapsed_time", 0.0)
    if matlab_time > 0 and python_time > 0:
        speedup = matlab_time / python_time
        comparison["performance"] = {
            "matlab_time_seconds": matlab_time,
            "python_time_seconds": python_time,
            "speedup": speedup,
            "faster": "Python" if speedup > 1.0 else "MATLAB",
        }

    if matlab_parsed and python_data:
        if "vertices" in matlab_parsed and "vertices" in python_data:
            comparison["vertices"] = compare_vertices(matlab_parsed["vertices"], python_data["vertices"])

        if "edges" in matlab_parsed and "edges" in python_data:
            comparison["edges"] = compare_edges(
                matlab_parsed["edges"],
                python_data["edges"],
                python_results.get("candidate_edges") or python_data.get("candidate_edges"),
                python_results.get("candidate_audit") or python_data.get("candidate_audit"),
            )

        if "network" in matlab_parsed and "network" in python_data:
            comparison["network"] = compare_networks(
                matlab_parsed["network"],
                python_data["network"],
                matlab_parsed.get("network_stats"),
            )

    parity_gate = {
        "vertices_exact": comparison.get("vertices", {}).get("exact_positions_scales_match"),
        "edges_exact": comparison.get("edges", {}).get("exact_match"),
        "strands_exact": comparison.get("network", {}).get("exact_match"),
    }
    available_checks = [value for value in parity_gate.values() if value is not None]
    parity_gate["passed"] = all(available_checks) if available_checks else None
    comparison["parity_gate"] = parity_gate
    return comparison
