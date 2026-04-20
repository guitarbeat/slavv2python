"""Comparison analysis and post-run artifact helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from slavv.runtime import RunContext

from .._persistence import load_json_dict_or_empty, write_json_file
from ..diagnostics import (
    format_shared_neighborhood_summary,
    generate_shared_neighborhood_diagnostics,
    load_shared_neighborhood_diagnostics,
    recommend_diagnostics_if_needed,
)
from .artifacts import _write_comparison_quick_view, _write_comparison_report

_COMPARISON_DEPTH_CHOICES = {"shallow", "deep"}


def _load_network_gate_parity_status(metadata_dir: Path) -> bool | None:
    """Return the latest persisted network-gate parity status when available."""
    execution_path = metadata_dir / "network_gate_execution.json"
    payload = load_json_dict_or_empty(execution_path)
    if not payload:
        return None
    return bool(payload.get("parity_achieved"))


def _surface_shared_neighborhood_diagnostics(
    *,
    run_root: Path,
    comparison: dict[str, Any],
    analysis_dir: Path,
    metadata_dir: Path,
) -> None:
    """Persist or summarize the canonical shared-neighborhood diagnostic report."""
    edges_parity_ok = bool(comparison.get("edges", {}).get("exact_match"))
    if edges_parity_ok:
        return

    network_gate_parity_ok = _load_network_gate_parity_status(metadata_dir)
    diagnostic_report = None
    try:
        diagnostic_report = generate_shared_neighborhood_diagnostics(run_root)
    except FileNotFoundError:
        diagnostic_report = load_shared_neighborhood_diagnostics(run_root)

    if diagnostic_report is not None:
        print("\nShared-Neighborhood Diagnostics")
        print(format_shared_neighborhood_summary(diagnostic_report))
        print(
            "Artifacts: "
            f"{analysis_dir / 'shared_neighborhood_diagnostics.json'} and "
            f"{analysis_dir / 'shared_neighborhood_diagnostics.md'}"
        )
        return

    if recommendation := recommend_diagnostics_if_needed(
        run_root=run_root,
        edges_parity_ok=edges_parity_ok,
        network_gate_parity_ok=network_gate_parity_ok,
    ):
        print("\nTriage Recommendation")
        print(recommendation)


def _should_load_deep_matlab_results(comparison_depth: str) -> bool:
    """Return whether comparison should parse the full MATLAB batch surface."""
    normalized = comparison_depth.strip().lower()
    if normalized not in _COMPARISON_DEPTH_CHOICES:
        raise ValueError(
            f"Unsupported comparison depth '{comparison_depth}'. "
            f"Expected one of: {sorted(_COMPARISON_DEPTH_CHOICES)}"
        )
    return normalized == "deep"


def _run_comparison_analysis(
    *,
    matlab_results: dict[str, Any],
    python_results: dict[str, Any],
    run_root: Path,
    analysis_dir: Path,
    metadata_dir: Path,
    comparison_depth: str,
    comparison_context: RunContext | None = None,
    load_matlab_batch_results_fn=None,
    compare_results_fn=None,
    build_shared_neighborhood_audit_fn=None,
) -> None:
    """Run the shared MATLAB-vs-Python analysis path and persist the report."""
    if comparison_context is not None:
        comparison_context.update_optional_task(
            "comparison_analysis",
            status="running",
            detail="Comparing MATLAB and Python results",
        )

    matlab_parsed = None
    if matlab_results.get("success") and matlab_results.get("batch_folder"):
        print("\nLoading MATLAB output data...")
        if _should_load_deep_matlab_results(comparison_depth):
            try:
                matlab_parsed = load_matlab_batch_results_fn(matlab_results["batch_folder"])
                print(f"Successfully loaded MATLAB data from {matlab_results['batch_folder']}")
            except Exception as exc:
                print(f"Warning: Could not load MATLAB output data: {exc}")
                print("Comparison will proceed with basic metrics only.")
        else:
            print("Shallow comparison mode enabled; skipping full MATLAB batch parse.")

    comparison = compare_results_fn(matlab_results, python_results, matlab_parsed)
    python_data = python_results.get("results") or {}
    shared_neighborhood_audit = None
    if matlab_parsed and "edges" in comparison:
        shared_neighborhood_audit = build_shared_neighborhood_audit_fn(
            matlab_parsed.get("edges", {}),
            python_data.get("edges", {}),
            python_results.get("candidate_edges") or python_data.get("candidate_edges"),
            comparison["edges"],
            python_results.get("candidate_audit") or python_data.get("candidate_audit"),
            python_results.get("candidate_lifecycle") or python_data.get("candidate_lifecycle"),
        )
    if shared_neighborhood_audit is not None:
        comparison.setdefault("edges", {}).setdefault("diagnostics", {})[
            "shared_neighborhood_audit"
        ] = shared_neighborhood_audit
        write_json_file(analysis_dir / "shared_neighborhood_audit.json", shared_neighborhood_audit)
    report_file = _write_comparison_report(comparison, analysis_dir / "comparison_report.json")
    quick_json, quick_tsv = _write_comparison_quick_view(comparison, analysis_dir)
    print(f"Quick compare artifacts: {quick_json} and {quick_tsv}")
    _surface_shared_neighborhood_diagnostics(
        run_root=run_root,
        comparison=comparison,
        analysis_dir=analysis_dir,
        metadata_dir=metadata_dir,
    )

    if comparison_context is not None:
        comparison_context.update_optional_task(
            "comparison_analysis",
            status="completed",
            detail="Comparison report generated",
            artifacts={
                "comparison_report": str(report_file),
                "comparison_depth": comparison_depth,
            },
        )


def _generate_post_comparison_artifacts(
    *,
    run_dir: Path,
    analysis_dir: Path,
    metadata_dir: Path,
    comparison_context: RunContext | None = None,
    include_summary: bool = True,
    include_manifest: bool = True,
    announce_manifest: bool = False,
    generate_summary_fn=None,
    generate_manifest_fn=None,
) -> None:
    """Best-effort generation of summary and manifest sidecar files."""
    from ..run_layout import (
        apply_managed_archive_cleanup,
        managed_comparisons_root_from_path,
        refresh_managed_archive_metadata,
    )

    managed_root = managed_comparisons_root_from_path(run_dir)
    if managed_root is not None:
        try:
            cleanup_payload = apply_managed_archive_cleanup(run_dir)
            if comparison_context is not None:
                cleanup_applied = bool(cleanup_payload.get("applied", False))
                detail = (
                    "Managed archive cleanup applied"
                    if cleanup_applied
                    else "Managed archive cleanup skipped"
                )
                if skip_reason := str(cleanup_payload.get("skip_reason", "") or "").strip():
                    detail = f"{detail}: {skip_reason}"
                comparison_context.update_optional_task(
                    "artifact_cleanup",
                    status="completed" if cleanup_applied else "skipped",
                    detail=detail,
                    artifacts={
                        "artifact_cleanup": str(metadata_dir / "artifact_cleanup.json"),
                        "profile": str(cleanup_payload.get("profile", "")),
                    },
                )
        except Exception as exc:
            print(f"Note: Could not compact managed archive artifacts: {exc}")

        try:
            refresh_managed_archive_metadata(managed_root)
        except Exception as exc:
            print(f"Note: Could not refresh managed archive metadata: {exc}")

    if include_summary:
        try:
            summary_file = analysis_dir / "summary.txt"
            generate_summary_fn(run_dir, summary_file)
        except Exception as exc:
            print(f"Note: Could not auto-generate summary: {exc}")

    if include_manifest:
        try:
            manifest_file = metadata_dir / "run_manifest.md"
            generate_manifest_fn(run_dir, manifest_file)
            if announce_manifest:
                print(f"Manifest generated: {manifest_file}")
            if comparison_context is not None:
                comparison_context.update_optional_task(
                    "manifest",
                    status="completed",
                    detail="Comparison manifest written",
                    artifacts={"manifest": str(manifest_file)},
                )
        except Exception as exc:
            print(f"Note: Could not auto-generate manifest: {exc}")
