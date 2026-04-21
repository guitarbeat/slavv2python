"""Focused tests for staged run layout support in parity tools."""

import json
from pathlib import Path

from slavv.parity.reporting import generate_summary
from slavv.parity.run_layout import (
    RunMetadata,
    build_experiment_index_entry,
    collect_directory_inventory,
    generate_manifest,
    infer_quality_gate,
    infer_run_status,
    list_runs,
    load_run_info,
    load_run_metadata,
    refresh_managed_archive_metadata,
    resolve_run_layout,
    select_pointer_targets,
    write_pointer_file,
    write_run_status,
)


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


def test_collect_directory_inventory_ignores_transient_os_errors(tmp_path: Path, monkeypatch):
    run_dir = tmp_path / "20260209_173550_full_run"
    run_dir.mkdir(parents=True)

    def broken_rglob(self: Path, _pattern: str):
        raise FileNotFoundError("transient scan failure")

    monkeypatch.setattr(Path, "rglob", broken_rglob)

    inventory = collect_directory_inventory(run_dir)

    assert inventory["total_size"] == 0
    assert all(values == [] for values in inventory["inventory"].values())


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


def test_load_run_metadata_for_staged_run(tmp_path: Path):
    run_dir = tmp_path / "20260210_100526_full_run"
    metadata_dir = run_dir / "99_Metadata"
    metadata_dir.mkdir(parents=True)
    (run_dir / "03_Analysis").mkdir(parents=True)

    (metadata_dir / "run_snapshot.json").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "status": "completed",
                "target_stage": "network",
                "current_stage": "network",
                "overall_progress": 1.0,
                "stages": {},
                "optional_tasks": {},
            }
        ),
        encoding="utf-8",
    )
    (metadata_dir / "loop_assessment.json").write_text(
        json.dumps({"verdict": "analysis_ready", "safe_to_analyze_only": True}),
        encoding="utf-8",
    )
    (metadata_dir / "output_preflight.json").write_text(
        json.dumps({"preflight_status": "passed", "allows_launch": True}),
        encoding="utf-8",
    )

    metadata = load_run_metadata(run_dir)

    assert metadata.run_root == run_dir
    assert metadata.layout_kind == "staged"
    assert metadata.run_snapshot is not None
    assert metadata.run_snapshot.status == "completed"
    assert metadata.loop_assessment is not None
    assert metadata.loop_assessment["verdict"] == "analysis_ready"
    assert metadata.preflight_report is not None
    assert metadata.preflight_report["preflight_status"] == "passed"


def test_load_run_metadata_for_legacy_run(tmp_path: Path):
    run_dir = tmp_path / "legacy_run"
    run_dir.mkdir(parents=True)
    (run_dir / "run_snapshot.json").write_text(
        json.dumps(
            {
                "run_id": "run-legacy",
                "status": "pending",
                "target_stage": "network",
                "stages": {},
                "optional_tasks": {},
            }
        ),
        encoding="utf-8",
    )

    metadata = load_run_metadata(run_dir)

    assert metadata.run_root == run_dir
    assert metadata.layout_kind == "legacy_flat"
    assert metadata.run_snapshot is not None
    assert metadata.run_snapshot.run_id == "run-legacy"
    assert metadata.lifecycle_status is None


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


def test_list_runs_includes_metadata_only_runs(tmp_path: Path):
    run_dir = tmp_path / "20260209_180000_interrupted_run"
    metadata_dir = run_dir / "99_Metadata"
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "run_snapshot.json").write_text(
        json.dumps(
            {
                "run_id": "abc123",
                "target_stage": "edges",
                "stages": {},
                "optional_tasks": {},
                "artifacts": {},
                "errors": [],
                "provenance": {},
            }
        ),
        encoding="utf-8",
    )

    names = [run["name"] for run in list_runs(tmp_path)]

    assert "20260209_180000_interrupted_run" in names


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
    edge_stage_dir = python_dir / "stages" / "edges"
    matlab_dir.mkdir(parents=True)
    edge_stage_dir.mkdir(parents=True)
    analysis_dir.mkdir(parents=True)
    (matlab_dir / "batch_260210-100526").mkdir(parents=True)
    (python_dir / "network.json").write_text("{}", encoding="utf-8")
    (edge_stage_dir / "candidate_audit.json").write_text("{}", encoding="utf-8")
    (edge_stage_dir / "candidates.pkl").write_bytes(b"candidates")
    (analysis_dir / "shared_neighborhood_audit.json").write_text("{}", encoding="utf-8")
    report = {
        "python": {"comparison_mode": {"energy_source": "matlab_batch_hdf5"}},
        "edges": {
            "matched_endpoint_pair_count": 9,
            "missing_endpoint_pair_count": 3,
            "extra_endpoint_pair_count": 4,
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
                        "watershed_metric_threshold_rejected": 5,
                        "watershed_cap_rejected": 6,
                        "watershed_short_trace_rejected": 7,
                        "watershed_accepted": 9,
                    },
                },
                "shared_neighborhood_audit": {
                    "schema_version": 1,
                    "top_neighborhood": {
                        "origin_index": 866,
                        "missing_matlab_incident_endpoint_pair_count": 3,
                        "candidate_endpoint_pair_count": 1,
                        "final_chosen_endpoint_pair_count": 1,
                        "first_divergence_stage": "pre_manifest_rejection",
                        "first_divergence_reason": "rejected_parent_has_child at terminal 1023",
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
                    "conflict_rejected_by_source": {"frontier": 3, "watershed": 2},
                    "conflict_blocking_source_counts": {"frontier": 4, "watershed": 1},
                    "conflict_source_pairs": {
                        "frontier->frontier": 3,
                        "watershed->frontier": 1,
                        "watershed->watershed": 1,
                    },
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
                "chosen_candidate_sources": {
                    "counts": {"frontier": 1, "watershed": 1, "fallback": 0},
                    "watershed_endpoint_pair_count": 1,
                    "watershed_matched_matlab_endpoint_pair_count": 0,
                    "watershed_extra_python_endpoint_pair_count": 1,
                    "source_breakdown": {
                        "frontier": {
                            "matched_matlab_edge_count": 1,
                            "extra_python_edge_count": 0,
                            "matched": {
                                "median_energy": -220.5,
                                "median_length": 12.0,
                            },
                        },
                        "watershed": {
                            "matched_matlab_edge_count": 0,
                            "extra_python_edge_count": 1,
                            "extra": {
                                "median_energy": -88.1,
                                "median_length": 21.0,
                            },
                        },
                    },
                },
                "extra_frontier_missing_vertex_overlap": {
                    "extra_frontier_edge_count": 7,
                    "shared_missing_vertex_edge_count": 5,
                    "top_strength_overlap_counts": {
                        "20": {
                            "threshold": 20,
                            "shared_missing_vertex_count": 5,
                            "evaluated_edge_count": 7,
                        },
                        "50": {
                            "threshold": 50,
                            "shared_missing_vertex_count": 5,
                            "evaluated_edge_count": 7,
                        },
                    },
                    "top_shared_vertices": [
                        {
                            "vertex_index": 359,
                            "missing_matlab_endpoint_pair_count": 4,
                            "extra_frontier_endpoint_pair_count": 2,
                            "missing_matlab_pairs_present_in_candidates": 0,
                        },
                        {
                            "vertex_index": 768,
                            "missing_matlab_endpoint_pair_count": 3,
                            "extra_frontier_endpoint_pair_count": 3,
                            "missing_matlab_pairs_present_in_candidates": 1,
                        },
                    ],
                },
            },
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
    assert "Final endpoint pairs matched/matlab-only/python-only: 9/3/4" in summary
    assert "Candidate endpoint pairs candidate/matched-matlab/missing-matlab: 21/13/8" in summary
    assert "Candidate endpoint pairs extra-candidate/final-python: 5/12" in summary
    assert "Conflict rejects by source frontier/watershed/fallback/unknown: 3/2/0/0" in summary
    assert "Conflict blockers by source frontier/watershed/fallback/unknown: 4/1/0/0" in summary
    assert "Conflict source pairs f->f/f->w/w->f/w->w: 3/0/1/1" in summary
    assert "Chosen candidate sources frontier/watershed/geodesic/fallback: 1/1/0/0" in summary
    assert "Chosen watershed endpoint pairs total/matched-matlab/extra-python: 1/0/1" in summary
    assert "Chosen frontier edges matched/extra: 1/0" in summary
    assert "Chosen frontier profile: median energy matched/extra -220.5/n/a" in summary
    assert "median length matched/extra 12/n/a" in summary
    assert "Chosen watershed edges matched/extra: 0/1" in summary
    assert "Chosen watershed profile: median energy matched/extra n/a/-88.1" in summary
    assert "median length matched/extra n/a/21" in summary
    assert "Extra frontier edges sharing missing-matlab vertex total: 5/7" in summary
    assert "Strongest extra frontier sharing missing-matlab vertex:" in summary
    assert "top20:5/7" in summary
    assert "top50:5/7" in summary
    assert "Top shared frontier/missing vertices: 359(m4/e2) 768(m3/e3)" in summary
    assert "Top shared vertex missing-pair candidate hits: 359(0/4) 768(1/3)" in summary
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
        "Audit rejections reachability/mutual/endpoint-degree/energy/metric-threshold/cap/short/accepted: 1/2/3/4/5/6/7/9"
        in summary
    )
    assert (
        f"Shared neighborhood audit artifact: {analysis_dir / 'shared_neighborhood_audit.json'}"
        in summary
    )
    assert "Top shared neighborhood 866: missing/candidate/final 3/1/1" in summary
    assert (
        "First divergence point: pre_manifest_rejection - rejected_parent_has_child at terminal 1023"
        in summary
    )
    assert "Python energy source: matlab_batch_hdf5" in summary
    assert "Triage Recommendation" in summary
    assert "Start with candidate-endpoint coverage before edge or strand diffs." in summary


def test_generate_summary_and_manifest_include_workflow_decision(tmp_path: Path):
    run_dir = tmp_path / "20260210_100526_full_run"
    matlab_dir = run_dir / "01_Input" / "matlab_results"
    python_dir = run_dir / "02_Output" / "python_results"
    analysis_dir = run_dir / "03_Analysis"
    metadata_dir = run_dir / "99_Metadata"
    matlab_dir.mkdir(parents=True)
    python_dir.mkdir(parents=True)
    analysis_dir.mkdir(parents=True)
    metadata_dir.mkdir(parents=True)
    (matlab_dir / "batch_260210-100526").mkdir(parents=True)
    (python_dir / "network.json").write_text("{}", encoding="utf-8")
    (analysis_dir / "comparison_report.json").write_text("{}", encoding="utf-8")
    (metadata_dir / "loop_assessment.json").write_text(
        json.dumps(
            {
                "requested_loop": "full_comparison",
                "verdict": "fresh_matlab_required",
                "safe_to_reuse": False,
                "safe_to_analyze_only": True,
                "requires_fresh_matlab": True,
                "recommended_action": "Launch a fresh MATLAB run in this staged run root.",
            }
        ),
        encoding="utf-8",
    )
    (metadata_dir / "matlab_health_check.json").write_text(
        json.dumps(
            {
                "success": True,
                "elapsed_seconds": 4.2,
                "message": "MATLAB launch probe succeeded.",
            }
        ),
        encoding="utf-8",
    )

    summary_file = analysis_dir / "summary.txt"
    generate_summary(run_dir, summary_file)
    summary = summary_file.read_text(encoding="utf-8")
    manifest = generate_manifest(run_dir)

    assert "Workflow Decision" in summary
    assert "Safe to analyze only: True" in summary
    assert "Requires fresh MATLAB: True" in summary
    assert "MATLAB Health Check" in summary
    assert "## Workflow Decision" in manifest
    assert "- **Requires fresh MATLAB:** True" in manifest
    assert "## MATLAB Health Check" in manifest


def test_generate_manifest_reports_skipped_matlab_launch_reuse_mode(tmp_path: Path):
    run_dir = tmp_path / "20260210_104500_full_run"
    metadata_dir = run_dir / "99_Metadata"
    metadata_dir.mkdir(parents=True)
    (run_dir / "01_Input" / "matlab_results" / "batch_260210-104500").mkdir(parents=True)
    (run_dir / "02_Output" / "python_results").mkdir(parents=True)
    (run_dir / "03_Analysis").mkdir(parents=True)

    (metadata_dir / "run_snapshot.json").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "status": "completed",
                "target_stage": "network",
                "current_stage": "network",
                "overall_progress": 1.0,
                "optional_tasks": {
                    "matlab_pipeline": {
                        "status": "completed",
                        "detail": "MATLAB launch skipped due to completed reusable batch (analysis-only).",
                        "artifacts": {
                            "launch": "skipped",
                            "skip_reason": "completed_reusable_batch",
                            "reuse_mode": "analysis-only",
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    manifest = generate_manifest(run_dir)

    assert (
        "- **MATLAB launch:** skipped due to completed reusable batch (analysis-only)" in manifest
    )
    assert "- **MATLAB skip reason:** completed_reusable_batch" in manifest


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
    assert "`99_Metadata/matlab_status.json`" in normalized
    assert "## Failure Summary" in content
    assert "ERROR: MATLAB error Exit Status: 0x00000001" in content


def test_list_runs_discovers_grouped_experiment_runs(tmp_path: Path):
    comparisons_root = tmp_path / "slavv_comparisons"
    run_dir = (
        comparisons_root / "experiments" / "release-verify" / "runs" / "20260413_release_verify"
    )
    (run_dir / "03_Analysis").mkdir(parents=True)
    (run_dir / "03_Analysis" / "comparison_report.json").write_text("{}", encoding="utf-8")

    runs = list_runs(comparisons_root)

    assert [run["name"] for run in runs] == ["20260413_release_verify"]
    assert runs[0]["run_shape"] == "grouped_run_root"


def test_infer_run_status_prefers_explicit_status_json(tmp_path: Path, comparison_metadata_builder):
    run_dir = tmp_path / "20260413_release_verify"
    metadata_dir = comparison_metadata_builder(
        run_dir,
        run_snapshot={
            "run_id": "run123",
            "status": "completed",
            "target_stage": "network",
            "stages": {},
            "optional_tasks": {},
            "artifacts": {},
            "errors": [],
            "provenance": {},
        },
    )
    write_run_status(
        run_dir,
        {
            "state": "failed",
            "retention": "archive",
            "quality_gate": "fail",
            "notes": "authoritative lifecycle metadata",
        },
    )

    status = infer_run_status(run_dir)

    assert metadata_dir.exists()
    assert status["state"] == "failed"
    assert status["retention"] == "archive"
    assert status["quality_gate"] == "fail"


def test_list_runs_uses_pointer_targets_to_force_keep_retention(tmp_path: Path):
    comparisons_root = tmp_path / "slavv_comparisons"
    run_dir = (
        comparisons_root / "experiments" / "saved-batch" / "runs" / "20260327_150656_clean_parity"
    )
    (run_dir / "03_Analysis").mkdir(parents=True)
    (run_dir / "03_Analysis" / "comparison_report.json").write_text("{}", encoding="utf-8")
    write_pointer_file(
        comparisons_root / "pointers" / "best_saved_batch.txt",
        "experiments/saved-batch/runs/20260327_150656_clean_parity",
    )

    runs = list_runs(comparisons_root)

    assert runs[0]["retention"] == "keep"


def test_build_experiment_index_entry_tolerates_missing_optional_artifacts(tmp_path: Path):
    comparisons_root = tmp_path / "slavv_comparisons"
    run_dir = (
        comparisons_root
        / "experiments"
        / "python-consistency"
        / "runs"
        / "20260328_142659_python_consistency"
    )
    (run_dir / "02_Output" / "python_results").mkdir(parents=True)

    entry = build_experiment_index_entry(run_dir, comparisons_root=comparisons_root)

    assert (
        entry["run_path"]
        == "experiments/python-consistency/runs/20260328_142659_python_consistency"
    )
    assert entry["quality_gate"] == "unknown"
    assert "parity" not in entry


def test_refresh_managed_archive_metadata_writes_status_index_and_pointers(tmp_path: Path):
    comparisons_root = tmp_path / "slavv_comparisons"
    run_dir = (
        comparisons_root / "experiments" / "release-verify" / "runs" / "20260413_release_verify"
    )
    analysis_dir = run_dir / "03_Analysis"
    analysis_dir.mkdir(parents=True)
    (run_dir / "99_Metadata").mkdir(parents=True)
    (analysis_dir / "comparison_report.json").write_text(
        json.dumps(
            {
                "vertices": {"matches_exactly": True},
                "edges": {"matches_exactly": True},
                "network": {"matches_exactly": True},
            }
        ),
        encoding="utf-8",
    )

    refreshed = refresh_managed_archive_metadata(run_dir)

    status_path = run_dir / "99_Metadata" / "status.json"
    index_path = comparisons_root / "experiments" / "release-verify" / "index.json"
    latest_pointer = comparisons_root / "pointers" / "latest_completed.txt"
    canonical_pointer = comparisons_root / "pointers" / "canonical_acceptance.txt"
    saved_batch_pointer = comparisons_root / "pointers" / "best_saved_batch.txt"

    assert refreshed["managed_archive"] is True
    assert json.loads(status_path.read_text(encoding="utf-8"))["state"] == "completed"
    assert json.loads(status_path.read_text(encoding="utf-8"))["retention"] == "keep"
    assert json.loads(index_path.read_text(encoding="utf-8"))["runs"][0]["run_path"] == (
        "experiments/release-verify/runs/20260413_release_verify"
    )
    assert latest_pointer.read_text(encoding="utf-8").strip() == (
        "experiments/release-verify/runs/20260413_release_verify"
    )
    assert canonical_pointer.read_text(encoding="utf-8").strip() == (
        "experiments/release-verify/runs/20260413_release_verify"
    )
    assert saved_batch_pointer.read_text(encoding="utf-8").strip() == (
        "experiments/release-verify/runs/20260413_release_verify"
    )


def test_select_pointer_targets_prefers_newest_timestamp_over_experiment_slug():
    release_run = Path(
        "D:/slavv_comparisons/experiments/release-verify/runs/20260413_release_verify"
    )
    live_run = Path(
        "D:/slavv_comparisons/experiments/live-parity/runs/20260421_live_parity_clean"
    )

    pointers = select_pointer_targets(
        [
            {
                "run_root": release_run,
                "slug": "release-verify",
                "target_relative_path": "experiments/release-verify/runs/20260413_release_verify",
                "status": {
                    "state": "completed",
                    "retention": "keep",
                    "quality_gate": "unknown",
                },
            },
            {
                "run_root": live_run,
                "slug": "live-parity",
                "target_relative_path": "experiments/live-parity/runs/20260421_live_parity_clean",
                "status": {
                    "state": "completed",
                    "retention": "eligible_for_cleanup",
                    "quality_gate": "partial",
                },
            },
        ]
    )

    assert pointers["latest_completed.txt"] == (
        "experiments/live-parity/runs/20260421_live_parity_clean"
    )
    assert pointers["canonical_acceptance.txt"] == (
        "experiments/live-parity/runs/20260421_live_parity_clean"
    )


def test_refresh_managed_archive_metadata_ignores_nested_checkpoint_run_snapshots(tmp_path: Path):
    comparisons_root = tmp_path / "slavv_comparisons"
    run_dir = (
        comparisons_root / "experiments" / "live-parity" / "runs" / "20260421_live_parity_clean"
    )
    analysis_dir = run_dir / "03_Analysis"
    analysis_dir.mkdir(parents=True)
    (run_dir / "99_Metadata").mkdir(parents=True)
    (run_dir / "02_Output" / "python_results" / "checkpoints").mkdir(parents=True)
    (analysis_dir / "comparison_report.json").write_text(
        json.dumps({"parity_gate": {"passed": False, "vertices_exact": True, "edges_exact": False}}),
        encoding="utf-8",
    )
    (
        run_dir / "02_Output" / "python_results" / "checkpoints" / "run_snapshot.json"
    ).write_text("{}", encoding="utf-8")

    refreshed = refresh_managed_archive_metadata(run_dir)

    assert "checkpoints" not in refreshed["experiments"]
    assert not (comparisons_root / "experiments" / "checkpoints" / "index.json").exists()


def test_infer_quality_gate_uses_parity_gate_schema(tmp_path: Path):
    run_dir = tmp_path / "20260421_live_parity_clean"
    analysis_dir = run_dir / "03_Analysis"
    analysis_dir.mkdir(parents=True)
    (analysis_dir / "comparison_report.json").write_text(
        json.dumps(
            {
                "parity_gate": {
                    "passed": False,
                    "vertices_exact": True,
                    "edges_exact": False,
                    "strands_exact": False,
                }
            }
        ),
        encoding="utf-8",
    )

    assert infer_quality_gate(run_dir) == "partial"


def test_generate_manifest_includes_status_json_summary(tmp_path: Path):
    run_dir = tmp_path / "20260413_release_verify"
    analysis_dir = run_dir / "03_Analysis"
    metadata_dir = run_dir / "99_Metadata"
    analysis_dir.mkdir(parents=True)
    metadata_dir.mkdir(parents=True)
    write_run_status(
        run_dir,
        {
            "state": "completed",
            "retention": "keep",
            "quality_gate": "partial",
            "notes": "Canonical release verification run",
        },
    )

    content = generate_manifest(run_dir, metadata_dir / "run_manifest.md")

    assert "## Lifecycle Status" in content
    assert "- **Artifact:** `99_Metadata/status.json`" in content
    assert "- **Retention:** keep" in content


def test_summary_and_manifest_skip_pruned_artifact_references_after_cleanup(tmp_path: Path):
    run_dir = (
        tmp_path / "slavv_comparisons" / "experiments" / "live-parity" / "runs" / "20260418_live"
    )
    analysis_dir = run_dir / "03_Analysis"
    metadata_dir = run_dir / "99_Metadata"
    python_dir = run_dir / "02_Output" / "python_results"
    edge_stage_dir = python_dir / "stages" / "edges"
    analysis_dir.mkdir(parents=True)
    metadata_dir.mkdir(parents=True)
    edge_stage_dir.mkdir(parents=True)
    (python_dir / "network.json").write_text("{}", encoding="utf-8")
    (python_dir / "python_comparison_parameters.json").write_text("{}", encoding="utf-8")
    (python_dir / "stages" / "vertices").mkdir(parents=True, exist_ok=True)
    (python_dir / "stages" / "vertices" / "stage_manifest.json").write_text("{}", encoding="utf-8")
    (edge_stage_dir / "stage_manifest.json").write_text("{}", encoding="utf-8")
    (edge_stage_dir / "candidate_audit.json").write_text("{}", encoding="utf-8")
    (edge_stage_dir / "candidate_lifecycle.json").write_text("{}", encoding="utf-8")
    (analysis_dir / "comparison_report.json").write_text(
        json.dumps(
            {
                "edges": {
                    "exact_match": False,
                    "diagnostics": {
                        "candidate_audit": {
                            "schema_version": 1,
                            "source_breakdown": {
                                "frontier": {"candidate_connection_count": 1},
                                "watershed": {"candidate_connection_count": 0},
                                "fallback": {"candidate_connection_count": 0},
                            },
                        },
                        "shared_neighborhood_audit": {
                            "top_neighborhood": {
                                "origin_index": 1,
                                "missing_matlab_incident_endpoint_pair_count": 1,
                                "candidate_endpoint_pair_count": 1,
                                "final_chosen_endpoint_pair_count": 1,
                                "first_divergence_stage": "pre_manifest_rejection",
                                "first_divergence_reason": "none",
                            }
                        },
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    (metadata_dir / "artifact_cleanup.json").write_text(
        json.dumps(
            {
                "profile": "managed-analysis-retention-v1",
                "applied": True,
                "files_removed": 4,
                "bytes_removed": 2048,
                "bytes_removed_human": "2.0 KiB",
                "empty_directories_removed": 1,
            }
        ),
        encoding="utf-8",
    )
    (metadata_dir / "matlab_status.json").write_text(
        json.dumps(
            {
                "matlab_resume_mode": "complete-noop",
                "matlab_batch_folder": str(
                    run_dir / "01_Input" / "matlab_results" / "batch_260418-120000"
                ),
                "matlab_last_completed_stage": "network",
                "matlab_next_stage": "",
                "matlab_rerun_prediction": "Archive is analysis-only.",
                "matlab_log_file": str(run_dir / "01_Input" / "matlab_results" / "matlab_run.log"),
            }
        ),
        encoding="utf-8",
    )

    summary_file = analysis_dir / "summary.txt"
    generate_summary(run_dir, summary_file)
    manifest = generate_manifest(run_dir, metadata_dir / "run_manifest.md")
    summary = summary_file.read_text(encoding="utf-8")

    assert "Candidate audit artifact:" in summary
    assert "Candidate manifest path:" not in summary
    assert "Shared neighborhood audit artifact:" not in summary
    assert "## Artifact Cleanup" in manifest
    assert "managed-analysis-retention-v1" in manifest
    assert "`02_Output/python_results/stages/edges/candidates.pkl`" not in manifest
    assert "matlab_run.log" not in manifest
    assert "batch_260418-120000" not in manifest


def test_generate_manifest_accepts_preloaded_metadata(tmp_path: Path):
    run_dir = tmp_path / "20260413_release_verify"
    metadata_dir = run_dir / "99_Metadata"
    metadata_dir.mkdir(parents=True)

    injected = RunMetadata(
        run_root=run_dir,
        layout_kind="staged",
        run_snapshot=None,
        lifecycle_status={"state": "completed", "retention": "keep", "quality_gate": "pass"},
        loop_assessment={"verdict": "analysis_ready", "safe_to_reuse": False},
        preflight_report={"preflight_status": "passed", "allows_launch": True},
        matlab_status=None,
        matlab_health_check=None,
        artifact_cleanup=None,
    )

    content = generate_manifest(
        run_dir,
        metadata_dir / "run_manifest.md",
        metadata=injected,
    )

    assert "## Lifecycle Status" in content
    assert "- **State:** completed" in content
    assert "## Workflow Decision" in content
    assert "- **Verdict:** analysis ready" in content
