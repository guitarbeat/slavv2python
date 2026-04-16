"""Tests for ``dev/scripts/maintenance/comparison_layout_smoothing.py``."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_workspace_module(relative_path: str, module_name: str):
    repo_root = Path(__file__).resolve().parents[4]
    module_path = repo_root / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_report(run_root: Path, *, exact_match: bool = True) -> None:
    analysis_dir = run_root / "03_Analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.joinpath("comparison_report.json").write_text(
        json.dumps(
            {
                "vertices": {"matches_exactly": exact_match},
                "edges": {"matches_exactly": exact_match},
            }
        ),
        encoding="utf-8",
    )


def test_build_migration_report_discovers_grouped_and_aggregate_runs(tmp_path):
    module = _load_workspace_module(
        "dev/scripts/maintenance/comparison_layout_smoothing.py",
        "comparison_layout_smoothing_discovery_test",
    )
    comparisons_root = tmp_path / "slavv_comparisons"
    direct_run = comparisons_root / "20260413_release_verify"
    _write_report(direct_run)
    aggregate_child = comparisons_root / "20260328_python_consistency" / "run_edges"
    _write_report(aggregate_child)

    report = module.build_migration_report(comparisons_root, repo_root=tmp_path)

    assert len(report["runs"]) == 2
    assert len(report["aggregate_containers"]) == 1
    assert report["aggregate_containers"][0]["children"] == [str(aggregate_child)]


def test_build_migration_report_includes_conflicts_and_pointer_proposals(tmp_path):
    module = _load_workspace_module(
        "dev/scripts/maintenance/comparison_layout_smoothing.py",
        "comparison_layout_smoothing_conflict_test",
    )
    comparisons_root = tmp_path / "slavv_comparisons"
    run_root = comparisons_root / "20260413_release_verify"
    _write_report(run_root)
    conflict_target = comparisons_root / "experiments" / "release-verify" / "runs" / "20260413_release_verify"
    _write_report(conflict_target)

    report = module.build_migration_report(comparisons_root, repo_root=tmp_path)

    assert any(run["conflict"] for run in report["runs"])
    assert set(report["pointer_proposals"]) == {
        "latest_completed.txt",
        "canonical_acceptance.txt",
        "best_saved_batch.txt",
    }


def test_apply_migration_report_is_idempotent(tmp_path):
    module = _load_workspace_module(
        "dev/scripts/maintenance/comparison_layout_smoothing.py",
        "comparison_layout_smoothing_apply_test",
    )
    comparisons_root = tmp_path / "slavv_comparisons"
    run_root = comparisons_root / "20260413_release_verify"
    _write_report(run_root)

    first_report = module.build_migration_report(comparisons_root, repo_root=tmp_path)
    module.apply_migration_report(first_report, comparisons_root=comparisons_root, repo_root=tmp_path)
    second_report = module.build_migration_report(comparisons_root, repo_root=tmp_path)
    applied = module.apply_migration_report(second_report, comparisons_root=comparisons_root, repo_root=tmp_path)

    assert applied["mode"] == "apply"
    assert applied["applied_moves"] == []


def test_apply_migration_report_writes_single_line_pointer_files(tmp_path):
    module = _load_workspace_module(
        "dev/scripts/maintenance/comparison_layout_smoothing.py",
        "comparison_layout_smoothing_pointer_test",
    )
    comparisons_root = tmp_path / "slavv_comparisons"
    saved_batch_run = comparisons_root / "20260327_150656_saved_batch"
    _write_report(saved_batch_run)

    report = module.build_migration_report(comparisons_root, repo_root=tmp_path)
    module.apply_migration_report(report, comparisons_root=comparisons_root, repo_root=tmp_path)

    pointer_path = comparisons_root / "pointers" / "best_saved_batch.txt"
    pointer_value = pointer_path.read_text(encoding="utf-8").splitlines()
    assert len(pointer_value) == 1
    assert pointer_value[0].startswith("experiments/")


def test_apply_migration_report_requires_allow_list_for_non_empty_deletes(tmp_path):
    module = _load_workspace_module(
        "dev/scripts/maintenance/comparison_layout_smoothing.py",
        "comparison_layout_smoothing_delete_test",
    )
    comparisons_root = tmp_path / "slavv_comparisons"
    failed_run = comparisons_root / "20260413_failed_verify"
    _write_report(failed_run, exact_match=False)
    (failed_run / "99_Metadata").mkdir(parents=True, exist_ok=True)
    (failed_run / "99_Metadata" / "status.json").write_text(
        json.dumps(
            {
                "state": "failed",
                "retention": "eligible_for_cleanup",
                "quality_gate": "fail",
            }
        ),
        encoding="utf-8",
    )

    report = module.build_migration_report(comparisons_root, repo_root=tmp_path)
    applied = module.apply_migration_report(report, comparisons_root=comparisons_root, repo_root=tmp_path)

    assert any(item["path"] for item in applied["skipped_deletions"])
    grouped_target = comparisons_root / "experiments" / "failed-verify" / "runs" / "20260413_failed_verify"
    assert grouped_target.exists()


def test_build_migration_report_uses_parent_slug_and_unique_names_for_aggregate_children(tmp_path):
    module = _load_workspace_module(
        "dev/scripts/maintenance/comparison_layout_smoothing.py",
        "comparison_layout_smoothing_aggregate_slug_test",
    )
    comparisons_root = tmp_path / "slavv_comparisons"
    matlab_run = comparisons_root / "20260328_023500_matlab_consistency" / "run_01"
    python_run = comparisons_root / "20260328_142659_python_consistency" / "run_01"
    _write_report(matlab_run)
    _write_report(python_run)

    report = module.build_migration_report(comparisons_root, repo_root=tmp_path)

    by_source = {entry["source_relative_path"]: entry for entry in report["runs"]}
    matlab_entry = by_source["20260328_023500_matlab_consistency/run_01"]
    python_entry = by_source["20260328_142659_python_consistency/run_01"]

    assert matlab_entry["slug"] == "matlab-consistency"
    assert python_entry["slug"] == "python-consistency"
    assert matlab_entry["normalized_name"] == "20260328_023500_matlab_consistency_run_01"
    assert python_entry["normalized_name"] == "20260328_142659_python_consistency_run_01"
    assert matlab_entry["target_relative_path"] != python_entry["target_relative_path"]


def test_apply_migration_report_skips_missing_sources_when_target_exists(tmp_path):
    module = _load_workspace_module(
        "dev/scripts/maintenance/comparison_layout_smoothing.py",
        "comparison_layout_smoothing_missing_source_test",
    )
    comparisons_root = tmp_path / "slavv_comparisons"
    target_run = comparisons_root / "experiments" / "release-verify" / "runs" / "20260413_release_verify"
    _write_report(target_run)

    report = {
        "runs": [
            {
                "source_path": str(comparisons_root / "20260413_release_verify"),
                "target_path": str(target_run),
                "target_relative_path": "experiments/release-verify/runs/20260413_release_verify",
                "action": "move",
                "conflict": False,
            }
        ],
        "pointer_proposals": {
            "latest_completed.txt": "experiments/release-verify/runs/20260413_release_verify",
            "canonical_acceptance.txt": "experiments/release-verify/runs/20260413_release_verify",
            "best_saved_batch.txt": "experiments/release-verify/runs/20260413_release_verify",
        },
        "cleanup_candidates": [],
    }

    applied = module.apply_migration_report(report, comparisons_root=comparisons_root, repo_root=tmp_path)

    assert applied["applied_moves"] == []
    assert applied["skipped_moves"]
    assert applied["skipped_moves"][0]["reason"] == "source missing, target already exists"
