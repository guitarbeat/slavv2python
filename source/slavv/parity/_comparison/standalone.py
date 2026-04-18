"""Standalone comparison workflow helpers."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def run_standalone_comparison(
    matlab_dir: Path,
    python_dir: Path,
    output_dir: Path,
    *,
    python_result_source: str = "auto",
    comparison_depth: str = "deep",
    resolve_run_layout_fn,
    assess_loop_request_fn,
    persist_loop_assessment_report_fn,
    load_python_results_from_source_fn,
    run_comparison_analysis_fn,
    generate_post_comparison_artifacts_fn,
    print_reuse_guidance_fn,
    load_matlab_batch_results_fn,
    compare_results_fn,
    build_shared_neighborhood_audit_fn,
    generate_summary_fn,
    generate_manifest_fn,
) -> int:
    """Run comparison on existing result directories."""
    print("\n" + "=" * 60)
    print("Running Standalone Comparison")
    print("=" * 60)
    print(f"MATLAB Dir: {matlab_dir}")
    print(f"Python Dir: {python_dir}")
    print(f"Output Dir: {output_dir}")

    layout = resolve_run_layout_fn(output_dir)
    run_root = layout["run_root"]
    analysis_dir = layout["analysis_dir"]
    metadata_dir = layout["metadata_dir"]
    os.makedirs(analysis_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)
    loop_assessment = assess_loop_request_fn(
        run_root,
        loop_kind="standalone_analysis",
        standalone_matlab_dir=matlab_dir,
        standalone_python_dir=python_dir,
    )
    persist_loop_assessment_report_fn(loop_assessment, metadata_dir)

    matlab_results = {
        "success": True,
        "output_dir": str(matlab_dir),
        "elapsed_time": 0.0,
    }

    matlab_layout = resolve_run_layout_fn(matlab_dir)
    matlab_root = matlab_layout["matlab_dir"]
    if batch_folders := [
        d for d in os.listdir(matlab_root) if (matlab_root / d).is_dir() and d.startswith("batch_")
    ]:
        batch_folder = str(matlab_root / sorted(batch_folders)[-1])
        matlab_results["batch_folder"] = batch_folder
        print(f"Found MATLAB batch folder: {batch_folder}")
    else:
        print("Warning: No MATLAB batch_* folder found.")

    python_layout = resolve_run_layout_fn(python_dir)
    python_root = python_layout["python_dir"]
    python_results = load_python_results_from_source_fn(python_root, python_result_source)
    if not python_results.get("success"):
        print(f"Error loading Python results: {python_results.get('error', 'unknown error')}")
        return 1

    run_comparison_analysis_fn(
        matlab_results=matlab_results,
        python_results=python_results,
        run_root=run_root,
        analysis_dir=analysis_dir,
        metadata_dir=metadata_dir,
        comparison_depth=comparison_depth,
        load_matlab_batch_results_fn=load_matlab_batch_results_fn,
        compare_results_fn=compare_results_fn,
        build_shared_neighborhood_audit_fn=build_shared_neighborhood_audit_fn,
    )
    generate_post_comparison_artifacts_fn(
        run_dir=run_root,
        analysis_dir=analysis_dir,
        metadata_dir=metadata_dir,
        generate_summary_fn=generate_summary_fn,
        generate_manifest_fn=generate_manifest_fn,
    )

    print_reuse_guidance_fn(
        run_root,
        input_file=None,
        params=None,
        loop_kind="standalone_analysis",
    )
    return 0
