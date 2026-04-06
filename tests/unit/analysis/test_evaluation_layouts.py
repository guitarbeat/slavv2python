"""Focused tests for staged run layout support in evaluation tools."""

import json
from pathlib import Path

from slavv.evaluation.management import (
    collect_directory_inventory,
    generate_manifest,
    list_runs,
    load_run_info,
    resolve_run_layout,
)
from slavv.evaluation.reporting import generate_summary


def test_resolve_run_layout_prefers_staged_directories(tmp_path: Path):
    run_dir = tmp_path / "20260210_101213_manual_run"
    (run_dir / "01_Input" / "matlab_results").mkdir(parents=True)
    (run_dir / "02_Output" / "python_results").mkdir(parents=True)
    (run_dir / "03_Analysis").mkdir(parents=True)

    layout = resolve_run_layout(run_dir)

    assert layout["matlab_dir"] == run_dir / "01_Input" / "matlab_results"
    assert layout["python_dir"] == run_dir / "02_Output" / "python_results"
    assert layout["analysis_dir"] == run_dir / "03_Analysis"
    assert layout["report_file"] == run_dir / "03_Analysis" / "comparison_report.json"


def test_load_run_info_reads_report_from_analysis_stage(tmp_path: Path):
    run_dir = tmp_path / "20260209_173550_full_run"
    (run_dir / "01_Input" / "matlab_results").mkdir(parents=True)
    (run_dir / "02_Output" / "python_results").mkdir(parents=True)
    analysis_dir = run_dir / "03_Analysis"
    analysis_dir.mkdir(parents=True)
    report = {
        "matlab": {"elapsed_time": 20.0, "vertices_count": 10},
        "python": {"elapsed_time": 10.0, "vertices_count": 12},
        "performance": {"speedup": 2.0},
    }
    (analysis_dir / "comparison_report.json").write_text(json.dumps(report), encoding="utf-8")

    info = load_run_info(run_dir)

    assert info["has_matlab"] is True
    assert info["has_python"] is True
    assert info["has_report"] is True
    assert info["speedup"] == 2.0


def test_collect_directory_inventory_returns_size_and_type_buckets(tmp_path: Path):
    run_dir = tmp_path / "20260209_173550_full_run"
    (run_dir / "03_Analysis").mkdir(parents=True)
    json_file = run_dir / "03_Analysis" / "comparison_report.json"
    text_file = run_dir / "99_Metadata" / "notes.txt"
    (run_dir / "99_Metadata").mkdir(parents=True)
    json_file.write_text("{}", encoding="utf-8")
    text_file.write_text("hello", encoding="utf-8")

    inventory = collect_directory_inventory(run_dir)

    assert inventory["total_size"] == json_file.stat().st_size + text_file.stat().st_size
    assert inventory["inventory"]["json"] == [json_file]
    assert inventory["inventory"]["txt"] == [text_file]
    assert inventory["inventory"]["other"] == []


def test_load_run_info_normalizes_staged_input_path(tmp_path: Path):
    run_dir = tmp_path / "20260209_173550_full_run"
    analysis_dir = run_dir / "03_Analysis"
    analysis_dir.mkdir(parents=True)

    info = load_run_info(analysis_dir)

    assert info["name"] == "20260209_173550_full_run"
    assert info["path"] == run_dir


def test_resolve_run_layout_normalizes_result_subdirectories(tmp_path: Path):
    run_dir = tmp_path / "20260209_173550_full_run"
    matlab_dir = run_dir / "01_Input" / "matlab_results"
    python_dir = run_dir / "02_Output" / "python_results"
    matlab_dir.mkdir(parents=True)
    python_dir.mkdir(parents=True)

    matlab_layout = resolve_run_layout(matlab_dir)
    python_layout = resolve_run_layout(python_dir)

    assert matlab_layout["run_root"] == run_dir
    assert matlab_layout["matlab_dir"] == matlab_dir
    assert python_layout["run_root"] == run_dir
    assert python_layout["python_dir"] == python_dir


def test_list_runs_returns_run_roots_for_staged_layout(tmp_path: Path):
    run_with_report = tmp_path / "20260209_173134_full_run"
    (run_with_report / "03_Analysis").mkdir(parents=True)
    (run_with_report / "03_Analysis" / "comparison_report.json").write_text("{}", encoding="utf-8")

    run_with_python = tmp_path / "20260209_173027_full_run"
    (run_with_python / "02_Output" / "python_results").mkdir(parents=True)

    names = [run["name"] for run in list_runs(tmp_path)]

    assert "20260209_173134_full_run" in names
    assert "20260209_173027_full_run" in names
    assert "03_Analysis" not in names
    assert "02_Output" not in names


def test_generate_summary_uses_staged_result_paths(tmp_path: Path):
    run_dir = tmp_path / "20260210_100526_full_run"
    matlab_dir = run_dir / "01_Input" / "matlab_results"
    python_dir = run_dir / "02_Output" / "python_results"
    analysis_dir = run_dir / "03_Analysis"
    matlab_dir.mkdir(parents=True)
    python_dir.mkdir(parents=True)
    analysis_dir.mkdir(parents=True)

    # Ensure directories are non-empty so status checks mark them as present.
    (matlab_dir / "batch_260210-100526").mkdir(parents=True)
    (python_dir / "checkpoints").mkdir(parents=True)
    (analysis_dir / "comparison_report.json").write_text("{}", encoding="utf-8")

    output_file = analysis_dir / "summary.txt"
    generate_summary(run_dir, output_file)
    summary = output_file.read_text(encoding="utf-8")

    assert "MATLAB results: Present" in summary
    assert "Python results: Present" in summary


def test_generate_summary_normalizes_staged_run_name(tmp_path: Path):
    run_dir = tmp_path / "20260210_100526_full_run"
    analysis_dir = run_dir / "03_Analysis"
    (run_dir / "01_Input" / "matlab_results").mkdir(parents=True)
    (run_dir / "02_Output" / "python_results").mkdir(parents=True)
    analysis_dir.mkdir(parents=True)
    (analysis_dir / "comparison_report.json").write_text("{}", encoding="utf-8")

    output_file = analysis_dir / "summary.txt"
    generate_summary(analysis_dir, output_file)
    summary = output_file.read_text(encoding="utf-8")

    assert "Run: 20260210_100526_full_run" in summary
    assert "Run: 03_Analysis" not in summary
    assert "Date: 2026-02-10" in summary


def test_generate_summary_falls_back_to_nested_counts(tmp_path: Path):
    run_dir = tmp_path / "20260210_100526_full_run"
    matlab_dir = run_dir / "01_Input" / "matlab_results"
    python_dir = run_dir / "02_Output" / "python_results"
    analysis_dir = run_dir / "03_Analysis"
    matlab_dir.mkdir(parents=True)
    python_dir.mkdir(parents=True)
    analysis_dir.mkdir(parents=True)
    (matlab_dir / "batch_260210-100526").mkdir(parents=True)
    (python_dir / "network.json").write_text("{}", encoding="utf-8")
    report = {
        "matlab": {"elapsed_time": 20.0},
        "python": {"elapsed_time": 10.0},
        "vertices": {"matlab_count": 1863, "python_count": 535},
        "edges": {"matlab_count": 1379, "python_count": 67},
        "network": {"matlab_strand_count": 682, "python_strand_count": 10},
    }
    (analysis_dir / "comparison_report.json").write_text(json.dumps(report), encoding="utf-8")

    output_file = analysis_dir / "summary.txt"
    generate_summary(run_dir, output_file)
    summary = output_file.read_text(encoding="utf-8")

    assert "1,863" in summary
    assert "1,379" in summary
    assert "682" in summary


def test_generate_summary_includes_extended_edge_diagnostics(tmp_path: Path):
    run_dir = tmp_path / "20260210_100526_full_run"
    matlab_dir = run_dir / "01_Input" / "matlab_results"
    python_dir = run_dir / "02_Output" / "python_results"
    analysis_dir = run_dir / "03_Analysis"
    matlab_dir.mkdir(parents=True)
    python_dir.mkdir(parents=True)
    analysis_dir.mkdir(parents=True)
    (matlab_dir / "batch_260210-100526").mkdir(parents=True)
    (python_dir / "network.json").write_text("{}", encoding="utf-8")
    report = {
        "python": {"comparison_mode": {"energy_source": "matlab_batch_hdf5"}},
        "edges": {
            "diagnostics": {
                "candidate_endpoint_coverage": {
                    "candidate_endpoint_pair_count": 21,
                    "matched_matlab_endpoint_pair_count": 13,
                    "missing_matlab_endpoint_pair_count": 8,
                    "extra_candidate_endpoint_pair_count": 5,
                    "python_endpoint_pair_count": 12,
                    "missing_matlab_endpoint_pair_samples": [[2, 356]],
                    "supplement_candidate_endpoint_pair_samples": [[12, 88]],
                    "missing_matlab_seed_origin_samples": [
                        {
                            "seed_origin_index": 2,
                            "missing_matlab_incident_endpoint_pair_count": 3,
                            "candidate_endpoint_pair_count": 1,
                            "missing_matlab_incident_endpoint_pair_samples": [[2, 356]],
                        }
                    ],
                    "extra_candidate_seed_origin_samples": [
                        {
                            "seed_origin_index": 12,
                            "extra_candidate_endpoint_pair_count": 4,
                            "candidate_endpoint_pair_count": 4,
                            "extra_candidate_endpoint_pair_samples": [[12, 88]],
                        }
                    ],
                },
                "candidate_audit": {
                    "schema_version": 1,
                    "candidate_connection_count": 13,
                    "candidate_origin_count": 5,
                    "source_breakdown": {
                        "frontier": {"candidate_connection_count": 9, "candidate_origin_count": 3},
                        "watershed": {"candidate_connection_count": 4, "candidate_origin_count": 1},
                        "fallback": {"candidate_connection_count": 0, "candidate_origin_count": 1},
                    },
                    "top_origin_summaries": [
                        {
                            "origin_index": 12,
                            "watershed_candidate_count": 4,
                            "frontier_candidate_count": 0,
                            "fallback_candidate_count": 0,
                            "candidate_connection_count": 4,
                        },
                        {
                            "origin_index": 5,
                            "watershed_candidate_count": 0,
                            "frontier_candidate_count": 3,
                            "fallback_candidate_count": 0,
                            "candidate_connection_count": 3,
                        },
                    ],
                    "diagnostic_counters": {
                        "watershed_reachability_rejected": 1,
                        "watershed_mutual_frontier_rejected": 2,
                        "watershed_endpoint_degree_rejected": 3,
                        "watershed_energy_rejected": 4,
                        "watershed_cap_rejected": 5,
                        "watershed_short_trace_rejected": 6,
                        "watershed_accepted": 9,
                    },
                },
                "python": {
                    "candidate_traced_edge_count": 10,
                    "terminal_edge_count": 3,
                    "chosen_edge_count": 2,
                    "watershed_join_supplement_count": 4,
                    "dangling_edge_count": 7,
                    "duplicate_directed_pair_count": 1,
                    "antiparallel_pair_count": 0,
                    "negative_energy_rejected_count": 4,
                    "conflict_rejected_count": 5,
                    "degree_pruned_count": 6,
                    "orphan_pruned_count": 7,
                    "cycle_pruned_count": 8,
                    "terminal_direct_hit_count": 9,
                    "terminal_reverse_center_hit_count": 10,
                    "terminal_reverse_near_hit_count": 11,
                    "stop_reason_counts": {
                        "bounds": 12,
                        "nan": 13,
                        "energy_threshold": 14,
                        "energy_rise_step_halving": 15,
                        "max_steps": 16,
                        "direct_terminal_hit": 17,
                        "frontier_exhausted_nonnegative": 18,
                        "length_limit": 19,
                        "terminal_frontier_hit": 20,
                    },
                },
            }
        },
    }
    (analysis_dir / "comparison_report.json").write_text(json.dumps(report), encoding="utf-8")

    output_file = analysis_dir / "summary.txt"
    generate_summary(run_dir, output_file)
    summary = output_file.read_text(encoding="utf-8")

    assert "Terminal resolution direct/reverse-center/reverse-near: 9/10/11" in summary
    assert (
        "Stop reasons bounds/nan/threshold/rise/max-steps/direct-hit: 12/13/14/15/16/17" in summary
    )
    assert "Frontier stop reasons exhausted/length-limit/terminal-hit: 18/19/20" in summary
    assert "Watershed join supplements: 4" in summary
    assert "Candidate endpoint pairs candidate/matched-matlab/missing-matlab: 21/13/8" in summary
    assert "Candidate endpoint pairs extra-candidate/final-python: 5/12" in summary
    assert "First missing candidate endpoint pair: [2, 356]" in summary
    assert (
        f"Candidate audit artifact: {analysis_dir.parent / '02_Output' / 'python_results' / 'stages' / 'edges' / 'candidate_audit.json'}"
        in summary
    )
    assert (
        f"Candidate manifest path: {analysis_dir.parent / '02_Output' / 'python_results' / 'stages' / 'edges' / 'candidates.pkl'}"
        in summary
    )
    assert "Top audit origin summaries" in summary
    assert "Candidate audit: schema=v1 frontier=9/watershed=4/fallback=0" in summary
    assert (
        "Top missing seed origin 2: missing matlab incident pairs 3  seed candidate pairs 1"
        in summary
    )
    assert "First missing pair at top seed origin: [2, 356]" in summary
    assert "First watershed supplement candidate pair: [12, 88]" in summary
    assert "Top extra seed origin 12: extra candidate pairs 4  seed candidate pairs 4" in summary
    assert "First extra pair at top seed origin: [12, 88]" in summary
    assert (
        "Audit rejections reachability/mutual/endpoint-degree/energy/cap/short/accepted: 1/2/3/4/5/6/9"
        in summary
    )
    assert "Python energy source: matlab_batch_hdf5" in summary
    assert "Triage Recommendation" in summary
    assert "Start with candidate-endpoint coverage before edge or strand diffs." in summary


def test_generate_manifest_normalizes_staged_run_root(tmp_path: Path):
    run_dir = tmp_path / "20260210_100526_full_run"
    analysis_dir = run_dir / "03_Analysis"
    metadata_dir = run_dir / "99_Metadata"
    (run_dir / "01_Input" / "matlab_results").mkdir(parents=True)
    python_dir = run_dir / "02_Output" / "python_results"
    python_dir.mkdir(parents=True)
    analysis_dir.mkdir(parents=True)
    metadata_dir.mkdir(parents=True)
    (python_dir / "network.json").write_text("{}", encoding="utf-8")
    (python_dir / "stages").mkdir(parents=True)
    (python_dir / "stages" / "edges").mkdir(parents=True)
    (python_dir / "stages" / "edges" / "candidates.pkl").write_text("{}", encoding="utf-8")
    (python_dir / "stages" / "edges" / "candidate_audit.json").write_text("{}", encoding="utf-8")

    content = generate_manifest(analysis_dir, metadata_dir / "run_manifest.md")
    normalized = content.replace("\\", "/")

    assert content.splitlines()[0] == "# SLAVV Comparison Run: 20260210_100526_full_run"
    assert "`02_Output/python_results/network.json`" in normalized
    assert "`02_Output/python_results/stages/edges/candidates.pkl`" in normalized
    assert "`02_Output/python_results/stages/edges/candidate_audit.json`" in normalized


def test_generate_manifest_uses_report_elapsed_times(tmp_path: Path):
    run_dir = tmp_path / "20260210_100526_full_run"
    analysis_dir = run_dir / "03_Analysis"
    metadata_dir = run_dir / "99_Metadata"
    analysis_dir.mkdir(parents=True)
    metadata_dir.mkdir(parents=True)
    report = {
        "matlab": {"elapsed_time": 12.5},
        "python": {"elapsed_time": 3.25},
        "performance": {"speedup": 3.85, "faster": "Python"},
    }
    (analysis_dir / "comparison_report.json").write_text(json.dumps(report), encoding="utf-8")

    content = generate_manifest(run_dir, metadata_dir / "run_manifest.md")

    assert "- **MATLAB:** 12.5s" in content
    assert "- **Python:** 3.2s" in content
    assert "- **Speedup:** 3.85x (Python faster)" in content


def test_generate_manifest_includes_output_preflight_summary(
    tmp_path: Path,
    comparison_metadata_builder,
):
    run_dir = tmp_path / "20260210_100526_full_run"
    analysis_dir = run_dir / "03_Analysis"
    analysis_dir.mkdir(parents=True)
    metadata_dir = comparison_metadata_builder(
        run_dir,
        output_preflight={
            "output_root": str(run_dir),
            "resolved_output_root": str(run_dir),
            "preflight_status": "warning",
            "allows_launch": True,
            "free_space_gb": 24.0,
            "required_space_gb": 5.0,
            "warnings": [
                "Output root appears to be under OneDrive sync; a local non-synced drive is safer for MATLAB outputs."
            ],
            "errors": [],
            "recommended_action": "Proceed with caution.",
        },
    )

    content = generate_manifest(run_dir, metadata_dir / "run_manifest.md")

    assert "## Preflight" in content
    assert "- **Status:** warning" in content
    assert "- **Artifact:** `99_Metadata/output_preflight.json`" in content
    assert "OneDrive sync" in content


def test_generate_manifest_includes_matlab_resume_semantics(
    tmp_path: Path,
    comparison_metadata_builder,
):
    run_dir = tmp_path / "20260210_100526_full_run"
    analysis_dir = run_dir / "03_Analysis"
    matlab_dir = run_dir / "01_Input" / "matlab_results"
    batch_dir = matlab_dir / "batch_260401-140000"
    analysis_dir.mkdir(parents=True)
    batch_dir.mkdir(parents=True)
    metadata_dir = comparison_metadata_builder(
        run_dir,
        matlab_status={
            "matlab_resume_mode": "restart-current-stage",
            "matlab_batch_folder": str(batch_dir),
            "matlab_last_completed_stage": "",
            "matlab_next_stage": "energy",
            "matlab_rerun_prediction": "Rerun will reuse batch_260401-140000 but restart energy from the stage boundary.",
            "matlab_partial_stage_artifacts_present": True,
            "matlab_partial_stage_name": "energy",
            "stale_running_snapshot_suspected": True,
            "failure_summary": "ERROR: MATLAB error Exit Status: 0x00000001",
            "matlab_log_tail": [
                "Running Energy workflow...",
                "ERROR: MATLAB error Exit Status: 0x00000001",
            ],
            "matlab_resume_state_file": str(matlab_dir / "matlab_resume_state.json"),
            "matlab_log_file": str(matlab_dir / "matlab_run.log"),
        },
    )

    content = generate_manifest(run_dir, metadata_dir / "run_manifest.md")
    normalized = content.replace("\\", "/")

    assert "## Resume Semantics" in content
    assert "- **MATLAB resume mode:** restart-current-stage" in content
    assert "- **MATLAB batch folder:** `01_Input/matlab_results/batch_260401-140000`" in normalized
    assert "## Authoritative Files" in content
    assert "`99_Metadata/matlab_status.json`" in content
    assert "## Failure Summary" in content
    assert "ERROR: MATLAB error Exit Status: 0x00000001" in content
