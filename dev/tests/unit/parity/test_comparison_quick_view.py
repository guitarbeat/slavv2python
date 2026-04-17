"""Tests for compact comparison sidecar artifacts."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from slavv.parity.comparison import (
    _build_comparison_quick_view,
    _write_comparison_quick_view,
    _write_comparison_report,
)

if TYPE_CHECKING:
    from pathlib import Path


def _sample_comparison() -> dict[str, object]:
    return {
        "matlab": {"elapsed_time": 20.0},
        "python": {"elapsed_time": 10.0},
        "performance": {"speedup": 2.0},
        "vertices": {"matlab_count": 100, "python_count": 90, "exact_match": False},
        "edges": {"matlab_count": 80, "python_count": 70, "exact_match": False},
        "network": {
            "matlab_strand_count": 50,
            "python_strand_count": 45,
            "exact_match": False,
        },
        "parity_gate": {"passed": False},
    }


def test_build_comparison_quick_view_uses_fallback_counts() -> None:
    quick = _build_comparison_quick_view(_sample_comparison())

    assert quick["vertices_matlab"] == 100
    assert quick["vertices_python"] == 90
    assert quick["vertices_diff"] == -10
    assert quick["edges_matlab"] == 80
    assert quick["edges_python"] == 70
    assert quick["edges_diff"] == -10
    assert quick["network_strands_matlab"] == 50
    assert quick["network_strands_python"] == 45
    assert quick["network_strands_diff"] == -5
    assert quick["matlab_elapsed_seconds"] == 20.0
    assert quick["python_elapsed_seconds"] == 10.0
    assert quick["python_vs_matlab_time_delta_seconds"] == -10.0
    assert quick["speedup"] == 2.0
    assert quick["parity_gate_passed"] is False


def test_write_comparison_quick_view_creates_sorted_json_and_tsv(tmp_path: Path) -> None:
    quick_json, quick_tsv = _write_comparison_quick_view(_sample_comparison(), tmp_path)

    assert quick_json.exists()
    assert quick_tsv.exists()

    quick_payload = json.loads(quick_json.read_text(encoding="utf-8"))
    assert quick_payload["speedup"] == 2.0

    tsv_lines = quick_tsv.read_text(encoding="utf-8").strip().splitlines()
    assert tsv_lines[0] == "metric\tvalue"
    metrics = [line.split("\t", 1)[0] for line in tsv_lines[1:]]
    assert metrics == sorted(metrics)


def test_write_comparison_report_serializes_top_level_keys_stably(tmp_path: Path) -> None:
    report_file = tmp_path / "comparison_report.json"
    _write_comparison_report(_sample_comparison(), report_file)

    report_text = report_file.read_text(encoding="utf-8")
    edges_pos = report_text.find('"edges"')
    matlab_pos = report_text.find('"matlab"')
    network_pos = report_text.find('"network"')
    parity_gate_pos = report_text.find('"parity_gate"')
    performance_pos = report_text.find('"performance"')
    python_pos = report_text.find('"python"')
    vertices_pos = report_text.find('"vertices"')

    assert -1 not in {
        edges_pos,
        matlab_pos,
        network_pos,
        parity_gate_pos,
        performance_pos,
        python_pos,
        vertices_pos,
    }
    assert edges_pos < matlab_pos < network_pos < parity_gate_pos < performance_pos < python_pos < vertices_pos
