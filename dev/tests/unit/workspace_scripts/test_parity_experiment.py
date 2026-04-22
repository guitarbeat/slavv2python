"""Unit tests for the developer parity experiment runner."""

from __future__ import annotations

import importlib
import json
from typing import TYPE_CHECKING

import pytest
from dev.tests.support.run_state_builders import (
    materialize_checkpoint_surface,
    materialize_run_snapshot,
)

parity_experiment = importlib.import_module("dev.scripts.cli.parity_experiment")

if TYPE_CHECKING:
    from pathlib import Path


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _build_source_run_root(tmp_path: Path) -> Path:
    run_root = tmp_path / "source-run"
    materialize_checkpoint_surface(
        run_root,
        stages=("energy", "vertices", "edges", "network"),
    )
    _write_json(
        run_root / "03_Analysis" / "comparison_report.json",
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
        run_root / "99_Metadata" / "validated_params.json",
        {"number_of_edges_per_vertex": 4},
    )
    return run_root


def test_build_parser_rerun_python_defaults():
    parser = parity_experiment.build_parser()

    args = parser.parse_args(
        [
            "rerun-python",
            "--source-run-root",
            "source-run",
            "--dest-run-root",
            "dest-run",
        ]
    )

    assert args.command == "rerun-python"
    assert args.rerun_from == "edges"
    assert args.params_file is None
    assert args.input is None


def test_validate_source_run_surface_accepts_required_artifacts(tmp_path):
    run_root = _build_source_run_root(tmp_path)

    surface = parity_experiment.validate_source_run_surface(run_root)

    assert surface.run_root == run_root.resolve()
    assert surface.checkpoints_dir == run_root.resolve() / parity_experiment.CHECKPOINTS_DIR
    assert surface.comparison_report_path.is_file()
    assert surface.validated_params_path.is_file()
    assert surface.run_snapshot_path is None


def test_validate_source_run_surface_reports_missing_artifacts(tmp_path):
    run_root = tmp_path / "source-run"
    run_root.mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValueError, match="missing required artifacts"):
        parity_experiment.validate_source_run_surface(run_root)


def test_resolve_input_file_uses_run_snapshot_provenance(tmp_path):
    repo_root = tmp_path / "repo"
    input_file = repo_root / "data" / "slavv_test_volume.tif"
    input_file.parent.mkdir(parents=True, exist_ok=True)
    input_file.write_bytes(b"tiff")

    run_root = _build_source_run_root(tmp_path)
    materialize_run_snapshot(
        run_root,
        {
            "run_id": "run-1",
            "provenance": {
                "input_file": "data/slavv_test_volume.tif",
            },
        },
    )
    surface = parity_experiment.validate_source_run_surface(run_root)

    resolved = parity_experiment.resolve_input_file(
        surface,
        None,
        repo_root=repo_root,
    )

    assert resolved == input_file.resolve()


def test_resolve_input_file_requires_snapshot_or_explicit_input(tmp_path):
    run_root = _build_source_run_root(tmp_path)
    surface = parity_experiment.validate_source_run_surface(run_root)

    with pytest.raises(ValueError, match="no --input was provided"):
        parity_experiment.resolve_input_file(surface, None)


def test_load_params_file_uses_source_default_and_override(tmp_path):
    run_root = _build_source_run_root(tmp_path)
    surface = parity_experiment.validate_source_run_surface(run_root)
    override = _write_json(tmp_path / "override_params.json", {"edge_method": "tracing"})

    default_params = parity_experiment.load_params_file(surface, None)
    override_params = parity_experiment.load_params_file(surface, str(override))

    assert default_params == {"number_of_edges_per_vertex": 4}
    assert override_params == {"edge_method": "tracing"}


def test_build_experiment_summary_computes_deltas(tmp_path):
    source_run_root = tmp_path / "source-run"
    dest_run_root = tmp_path / "dest-run"
    input_file = tmp_path / "input.tif"
    input_file.write_bytes(b"tiff")

    summary = parity_experiment.build_experiment_summary(
        source_run_root=source_run_root,
        dest_run_root=dest_run_root,
        input_file=input_file,
        rerun_from="edges",
        matlab_counts=parity_experiment.RunCounts(vertices=4, edges=5, strands=3),
        source_python_counts=parity_experiment.RunCounts(vertices=4, edges=2, strands=1),
        new_python_counts=parity_experiment.RunCounts(vertices=4, edges=3, strands=2),
    )

    assert summary["diff_vs_matlab"] == {"vertices": 0, "edges": -2, "strands": -1}
    assert summary["diff_vs_source_python"] == {"vertices": 0, "edges": 1, "strands": 1}


def test_summarize_command_prints_saved_summary(capsys, tmp_path):
    run_root = tmp_path / "dest-run"
    summary_path = run_root / parity_experiment.SUMMARY_TEXT_PATH
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("example summary", encoding="utf-8")

    parity_experiment.main(["summarize", "--run-root", str(run_root)])

    captured = capsys.readouterr()
    assert "example summary" in captured.out
