from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

from slavv.runtime import RunContext

from ..workflow_assessment import LOOP_ANALYSIS_READY, LOOP_REUSE_READY


def orchestrate_comparison(
    *,
    input_file: str,
    output_dir: Path,
    matlab_path: str,
    project_root: Path,
    params: dict[str, Any],
    skip_matlab: bool = False,
    skip_python: bool = False,
    validate_only: bool = False,
    minimal_exports: bool = False,
    comparison_depth: str = "deep",
    python_result_source: str = "auto",
    python_parity_rerun_from: str = "edges",
    resolve_run_layout_fn,
    resolve_python_parity_import_plan_fn,
    determine_loop_kind_fn,
    assess_loop_request_fn,
    persist_loop_assessment_report_fn,
    record_loop_assessment_task_fn,
    evaluate_output_root_preflight_for_workflow_fn,
    persist_preflight_report_fn,
    summarize_output_preflight_fn,
    record_preflight_task_fn,
    generate_post_comparison_artifacts_fn,
    inspect_matlab_status_for_workflow_fn,
    persist_matlab_status_report_fn,
    record_matlab_status_task_fn,
    summarize_matlab_status_fn,
    write_normalized_params_file_fn,
    run_matlab_vectorization_fn,
    persist_matlab_failure_report_fn,
    import_matlab_batch_fn,
    load_matlab_batch_params_fn,
    bootstrap_existing_matlab_batch_for_python_parity_fn,
    run_python_vectorization_fn,
    load_python_results_from_source_fn,
    run_comparison_analysis_fn,
    print_reuse_guidance_fn,
    load_matlab_batch_results_fn,
    compare_results_fn,
    build_shared_neighborhood_audit_fn,
    generate_summary_fn,
    generate_manifest_fn,
) -> int:
    """Run full comparison workflow using injected facade dependencies."""
    layout = resolve_run_layout_fn(output_dir)
    run_root = layout["run_root"]
    matlab_output = layout["matlab_dir"]
    python_output = layout["python_dir"]
    analysis_dir = layout["analysis_dir"]
    metadata_dir = run_root / "99_Metadata"
    import_stages, requested_python_rerun_from = resolve_python_parity_import_plan_fn(
        python_parity_rerun_from
    )
    params_for_python = copy.deepcopy(params)
    params_for_python["comparison_exact_network"] = True
    params_for_python["python_parity_rerun_from"] = requested_python_rerun_from
    loop_kind = determine_loop_kind_fn(
        standalone_mode=False,
        validate_only=validate_only,
        skip_matlab=skip_matlab,
        skip_python=skip_python,
        python_parity_rerun_from=requested_python_rerun_from,
    )
    comparison_context = RunContext(
        run_dir=run_root,
        target_stage="network",
        provenance={"source": "comparison", "input_file": input_file},
    )
    loop_assessment = assess_loop_request_fn(
        run_root,
        loop_kind=loop_kind,
        input_path=Path(input_file) if input_file else None,
        params=params_for_python,
    )
    loop_assessment_path = persist_loop_assessment_report_fn(loop_assessment, metadata_dir)
    record_loop_assessment_task_fn(comparison_context, loop_assessment, loop_assessment_path)
    python_force_rerun_from: str | None = None
    preflight_report = None
    matlab_status_report = None
    auto_matlab_reuse_mode: str | None = None

    if validate_only:
        preflight_report = evaluate_output_root_preflight_for_workflow_fn(run_root, metadata_dir)
        preflight_report_path = persist_preflight_report_fn(preflight_report, metadata_dir)
        print("\n" + "=" * 60)
        print("Output Root Preflight")
        print("=" * 60)
        print(summarize_output_preflight_fn(preflight_report))
        for warning in preflight_report.warnings:
            print(f"WARNING: {warning}")
        for error in preflight_report.errors:
            print(f"ERROR: {error}")

        record_preflight_task_fn(comparison_context, preflight_report, preflight_report_path)
        if not preflight_report.allows_launch:
            comparison_context.mark_run_status(
                "failed",
                current_stage="preflight",
                detail=summarize_output_preflight_fn(preflight_report),
            )
            print(f"Recommended action: {preflight_report.recommended_action}")
            return 1

        print("\nValidation-only mode enabled.")
        print("Preflight checks passed; skipping MATLAB/Python execution.")
        comparison_context.mark_run_status(
            "completed",
            current_stage="preflight",
            detail="Validation-only mode completed successfully.",
        )
        generate_post_comparison_artifacts_fn(
            run_dir=run_root,
            analysis_dir=analysis_dir,
            metadata_dir=metadata_dir,
            comparison_context=comparison_context,
            include_summary=False,
            announce_manifest=True,
            generate_summary_fn=generate_summary_fn,
            generate_manifest_fn=generate_manifest_fn,
        )
        print_reuse_guidance_fn(
            run_root,
            input_file=input_file,
            params=params_for_python,
            loop_kind=loop_kind,
        )
        return 0

    if not skip_matlab:
        preflight_report = evaluate_output_root_preflight_for_workflow_fn(run_root, metadata_dir)
        preflight_report_path = persist_preflight_report_fn(preflight_report, metadata_dir)

        print("\n" + "=" * 60)
        print("Output Root Preflight")
        print("=" * 60)
        print(summarize_output_preflight_fn(preflight_report))
        for warning in preflight_report.warnings:
            print(f"WARNING: {warning}")
        for error in preflight_report.errors:
            print(f"ERROR: {error}")

        if preflight_report.allows_launch or preflight_report_path is not None:
            record_preflight_task_fn(comparison_context, preflight_report, preflight_report_path)

        if not preflight_report.allows_launch:
            comparison_context.mark_run_status(
                "failed",
                current_stage="preflight",
                detail=summarize_output_preflight_fn(preflight_report),
            )
            generate_post_comparison_artifacts_fn(
                run_dir=run_root,
                analysis_dir=analysis_dir,
                metadata_dir=metadata_dir,
                comparison_context=comparison_context,
                include_summary=False,
                announce_manifest=True,
                generate_summary_fn=generate_summary_fn,
                generate_manifest_fn=generate_manifest_fn,
            )
            print(f"Recommended action: {preflight_report.recommended_action}")
            return 1

    analysis_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    normalized_params_file = write_normalized_params_file_fn(metadata_dir, params_for_python)
    if preflight_report is not None:
        record_preflight_task_fn(
            comparison_context, preflight_report, metadata_dir / "output_preflight.json"
        )

    if not skip_matlab:
        matlab_status_report = inspect_matlab_status_for_workflow_fn(
            matlab_output,
            metadata_dir,
            input_file=input_file,
        )
        matlab_status_report_path = persist_matlab_status_report_fn(
            matlab_status_report, metadata_dir
        )
        record_matlab_status_task_fn(
            comparison_context,
            matlab_status_report,
            matlab_status_report_path,
        )
        print("\n" + "=" * 60)
        print("MATLAB Rerun Semantics")
        print("=" * 60)
        print(summarize_matlab_status_fn(matlab_status_report))

        completed_matlab_batch = (
            bool(matlab_status_report.matlab_batch_complete)
            or matlab_status_report.matlab_resume_mode == "complete-noop"
        ) and bool(matlab_status_report.matlab_batch_folder)
        if completed_matlab_batch:
            if loop_assessment.verdict == LOOP_ANALYSIS_READY and not skip_python:
                auto_matlab_reuse_mode = "analysis-only"
            elif loop_assessment.verdict == LOOP_REUSE_READY:
                auto_matlab_reuse_mode = "python-rerun"

        if auto_matlab_reuse_mode is not None:
            comparison_context.update_optional_task(
                "matlab_pipeline",
                status="completed",
                detail=(
                    "MATLAB launch skipped due to completed reusable batch "
                    f"({auto_matlab_reuse_mode})."
                ),
                artifacts={
                    "launch": "skipped",
                    "skip_reason": "completed_reusable_batch",
                    "reuse_mode": auto_matlab_reuse_mode,
                    "batch_folder": matlab_status_report.matlab_batch_folder or "",
                },
            )
            record_matlab_status_task_fn(
                comparison_context,
                matlab_status_report,
                matlab_status_report_path,
                matlab_launch_skipped_reason="completed_reusable_batch",
                matlab_reuse_mode=auto_matlab_reuse_mode,
            )

    effective_skip_matlab = skip_matlab or auto_matlab_reuse_mode is not None
    analysis_only_reuse = auto_matlab_reuse_mode == "analysis-only"
    effective_skip_python = skip_python or analysis_only_reuse

    matlab_results = None
    if not effective_skip_matlab:
        os.makedirs(matlab_output, exist_ok=True)
        comparison_context.update_optional_task(
            "matlab_pipeline",
            status="running",
            detail="Running MATLAB wrapper",
            artifacts={"params_file": str(normalized_params_file)},
        )
        matlab_results = run_matlab_vectorization_fn(
            input_file,
            str(matlab_output),
            matlab_path,
            project_root,
            params_file=str(normalized_params_file),
        )
        matlab_status_report = inspect_matlab_status_for_workflow_fn(
            matlab_output,
            metadata_dir,
            input_file=input_file,
        )
        matlab_status_report_path = persist_matlab_status_report_fn(
            matlab_status_report, metadata_dir
        )
        failure_summary_path = persist_matlab_failure_report_fn(matlab_status_report, metadata_dir)
        record_matlab_status_task_fn(
            comparison_context,
            matlab_status_report,
            matlab_status_report_path,
            failure_summary_path,
        )
        matlab_results["matlab_status"] = matlab_status_report.to_dict()
        if failure_summary_path is not None:
            matlab_results["failure_summary_file"] = str(failure_summary_path)
        comparison_context.update_optional_task(
            "matlab_pipeline",
            status="completed" if matlab_results.get("success") else "failed",
            detail=(
                "MATLAB wrapper finished"
                if matlab_results.get("success")
                else (
                    matlab_status_report.failure_summary
                    or matlab_results.get("error", "MATLAB wrapper failed")
                )
            ),
            artifacts=(
                {
                    "batch_folder": matlab_results["batch_folder"],
                    "params_file": str(normalized_params_file),
                    "log_file": matlab_results.get("log_file", ""),
                }
                if matlab_results.get("batch_folder")
                else {
                    "params_file": str(normalized_params_file),
                    "log_file": matlab_results.get("log_file", ""),
                }
            ),
        )
        if not matlab_results.get("success"):
            comparison_context.mark_run_status(
                "failed",
                current_stage="matlab",
                detail=(
                    matlab_status_report.failure_summary
                    or matlab_results.get("error", "MATLAB wrapper failed")
                ),
            )
        batch_folder = matlab_results.get("batch_folder")
        if matlab_results.get("success") and batch_folder:
            checkpoint_dir = python_output / "checkpoints"
            comparison_context.update_optional_task(
                "matlab_import",
                status="running",
                detail="Importing MATLAB energy and vertices into Python checkpoints",
                artifacts={"batch_folder": batch_folder},
            )
            try:
                imported = import_matlab_batch_fn(
                    batch_folder,
                    checkpoint_dir,
                    stages=import_stages,
                )
            except Exception as exc:
                comparison_context.update_optional_task(
                    "matlab_import",
                    status="failed",
                    detail=f"MATLAB checkpoint import failed: {exc}",
                    artifacts={"batch_folder": batch_folder},
                )
            else:
                imported_stages = set(imported)
                if set(import_stages).issubset(imported_stages):
                    python_force_rerun_from = requested_python_rerun_from
                try:
                    params_for_python |= load_matlab_batch_params_fn(batch_folder)
                except Exception as exc:
                    comparison_context.update_optional_task(
                        "matlab_import",
                        status="completed",
                        detail=f"Imported MATLAB checkpoints; settings overlay unavailable: {exc}",
                        artifacts=dict(imported),
                    )
                else:
                    comparison_context.update_optional_task(
                        "matlab_import",
                        status="completed",
                        detail=(
                            "Imported MATLAB checkpoints for Python parity rerun "
                            f"from {requested_python_rerun_from}"
                        ),
                        artifacts={
                            **dict(imported),
                            "python_parity_rerun_from": requested_python_rerun_from,
                            **(
                                {"python_force_rerun_from": python_force_rerun_from}
                                if python_force_rerun_from is not None
                                else {}
                            ),
                        },
                    )
                record_matlab_status_task_fn(
                    comparison_context,
                    matlab_status_report,
                    matlab_status_report_path,
                    failure_summary_path,
                    python_force_rerun_from=python_force_rerun_from,
                )
    elif auto_matlab_reuse_mode == "analysis-only" and matlab_status_report is not None:
        matlab_results = {
            "success": True,
            "elapsed_time": 0.0,
            "output_dir": str(matlab_output),
            "batch_folder": matlab_status_report.matlab_batch_folder,
            "matlab_status": matlab_status_report.to_dict(),
        }
    else:
        if auto_matlab_reuse_mode == "python-rerun":
            print("\nSkipping MATLAB execution (completed reusable batch detected)")
        else:
            print("\nSkipping MATLAB execution (--skip-matlab)")
        matlab_results, python_force_rerun_from = (
            bootstrap_existing_matlab_batch_for_python_parity_fn(
                matlab_output=matlab_output,
                python_output=python_output,
                input_file=input_file,
                params_for_python=params_for_python,
                comparison_context=comparison_context,
                metadata_dir=metadata_dir,
                python_parity_rerun_from=requested_python_rerun_from,
            )
        )

    python_results = None
    if not effective_skip_python:
        os.makedirs(python_output, exist_ok=True)
        is_network_gate_candidate = (
            python_force_rerun_from == "network" and requested_python_rerun_from == "network"
        )

        if is_network_gate_candidate:
            from ..network_gate import (
                execute_stage_isolated_network_gate,
                validate_network_gate_artifacts,
            )

            validation = validate_network_gate_artifacts(run_root)
            if validation.validation_passed:
                print("\n" + "=" * 60)
                print("Stage-Isolated Network Gate Validation")
                print("=" * 60)
                print("Network gate validation PASSED:")
                print("  - MATLAB edges artifact present")
                print("  - MATLAB vertices artifact present")
                print("  - MATLAB energy artifact present")
                print(f"  Edges fingerprint: {validation.matlab_edges_fingerprint[:16]}...")
                print(f"  Vertices fingerprint: {validation.matlab_vertices_fingerprint[:16]}...")

                comparison_context.update_optional_task(
                    "network_gate_validation",
                    status="completed",
                    detail="Network gate validation passed",
                    artifacts={
                        "matlab_edges_fingerprint": validation.matlab_edges_fingerprint,
                        "matlab_vertices_fingerprint": validation.matlab_vertices_fingerprint,
                    },
                )
                print("\n" + "=" * 60)
                print("Stage-Isolated Network Gate Execution")
                print("=" * 60)
                comparison_context.update_optional_task(
                    "network_gate_execution",
                    status="running",
                    detail="Executing stage-isolated network gate",
                )
                execution = execute_stage_isolated_network_gate(
                    run_root,
                    input_file=Path(input_file),
                    params=params_for_python,
                )
                if execution.execution_errors:
                    print("\nNetwork gate execution encountered errors:")
                    for error in execution.execution_errors:
                        print(f"  - {error}")
                    comparison_context.update_optional_task(
                        "network_gate_execution",
                        status="failed",
                        detail="Network gate execution failed",
                        artifacts={
                            "elapsed_seconds": execution.elapsed_seconds,
                            "execution_errors": execution.execution_errors,
                        },
                    )
                    comparison_context.mark_run_status(
                        "failed",
                        current_stage="network_gate_execution",
                        detail=f"Network gate execution failed: {execution.execution_errors[0]}",
                    )
                    return 1

                print(
                    f"\nNetwork gate execution completed in {execution.elapsed_seconds:.2f} seconds"
                )
                print("\nParity Status:")
                print(f"  Vertices: {'PASS' if execution.vertices_match else 'FAIL'}")
                print(f"  Edges:    {'PASS' if execution.edges_match else 'FAIL'}")
                print(f"  Strands:  {'PASS' if execution.strands_match else 'FAIL'}")
                print(
                    f"  Overall:  {'PARITY ACHIEVED' if execution.parity_achieved else 'PARITY NOT ACHIEVED'}"
                )
                if execution.proof_artifact_json_path:
                    print("\nProof Artifact:")
                    print(f"  JSON: {execution.proof_artifact_json_path}")
                    if execution.proof_index_path:
                        print(f"  Index: {execution.proof_index_path}")

                comparison_context.update_optional_task(
                    "network_gate_execution",
                    status="completed",
                    detail=(
                        "Network gate execution completed - parity achieved"
                        if execution.parity_achieved
                        else "Network gate execution completed - parity not achieved"
                    ),
                    artifacts={
                        "elapsed_seconds": execution.elapsed_seconds,
                        "parity_achieved": execution.parity_achieved,
                        "vertices_match": execution.vertices_match,
                        "edges_match": execution.edges_match,
                        "strands_match": execution.strands_match,
                        "python_network_fingerprint": execution.python_network_fingerprint,
                        "proof_artifact_json_path": execution.proof_artifact_json_path,
                        "proof_index_path": execution.proof_index_path,
                    },
                )
                comparison_context.mark_run_status(
                    "completed",
                    current_stage="network_gate",
                    detail=(
                        "Stage-isolated network gate achieved exact parity"
                        if execution.parity_achieved
                        else "Stage-isolated network gate completed but parity not achieved"
                    ),
                )
                python_results = {
                    "success": True,
                    "elapsed_time": execution.elapsed_seconds,
                    "output_dir": str(python_output),
                    "network_gate_execution": True,
                    "parity_achieved": execution.parity_achieved,
                }
            else:
                print(
                    "\nNote: Network gate artifacts not available, running standard Python pipeline"
                )
                comparison_context.update_optional_task(
                    "python_pipeline",
                    status="running",
                    detail=(
                        f"Running Python pipeline from {python_force_rerun_from}"
                        if python_force_rerun_from is not None
                        else "Running Python pipeline"
                    ),
                )
                python_kwargs: dict[str, Any] = {
                    "run_dir": str(run_root),
                    "force_rerun_from": python_force_rerun_from,
                }
                if minimal_exports:
                    python_kwargs["minimal_exports"] = True
                python_results = run_python_vectorization_fn(
                    input_file,
                    str(python_output),
                    params_for_python,
                    **python_kwargs,
                )
                comparison_context.update_optional_task(
                    "python_pipeline",
                    status="completed" if python_results.get("success") else "failed",
                    detail="Python pipeline finished",
                    artifacts={
                        "output_dir": str(python_output),
                        **(
                            {"force_rerun_from": python_force_rerun_from}
                            if python_force_rerun_from is not None
                            else {}
                        ),
                    },
                )
        else:
            comparison_context.update_optional_task(
                "python_pipeline",
                status="running",
                detail=(
                    f"Running Python pipeline from {python_force_rerun_from}"
                    if python_force_rerun_from is not None
                    else "Running Python pipeline"
                ),
            )
            python_kwargs: dict[str, Any] = {
                "run_dir": str(run_root),
                "force_rerun_from": python_force_rerun_from,
            }
            if minimal_exports:
                python_kwargs["minimal_exports"] = True
            python_results = run_python_vectorization_fn(
                input_file,
                str(python_output),
                params_for_python,
                **python_kwargs,
            )
            comparison_context.update_optional_task(
                "python_pipeline",
                status="completed" if python_results.get("success") else "failed",
                detail="Python pipeline finished",
                artifacts={
                    "output_dir": str(python_output),
                    **(
                        {"force_rerun_from": python_force_rerun_from}
                        if python_force_rerun_from is not None
                        else {}
                    ),
                },
            )
    elif effective_skip_python and python_output.exists():
        if analysis_only_reuse:
            print(
                "\nReusing existing Python outputs for analysis-only comparison; "
                f"loading results from {python_output}"
            )
        else:
            print(
                f"\nSkipping Python execution (--skip-python); attempting to load results from {python_output}"
            )
        try:
            python_results = load_python_results_from_source_fn(python_output, python_result_source)
        except Exception as e:
            print(f"Warning: Could not load existing Python results: {e}")
            python_results = None
    else:
        print("\nSkipping Python execution (--skip-python)")

    if matlab_results and python_results:
        run_comparison_analysis_fn(
            matlab_results=matlab_results,
            python_results=python_results,
            run_root=run_root,
            analysis_dir=analysis_dir,
            metadata_dir=metadata_dir,
            comparison_depth=comparison_depth,
            comparison_context=comparison_context,
            load_matlab_batch_results_fn=load_matlab_batch_results_fn,
            compare_results_fn=compare_results_fn,
            build_shared_neighborhood_audit_fn=build_shared_neighborhood_audit_fn,
        )

    generate_post_comparison_artifacts_fn(
        run_dir=run_root,
        analysis_dir=analysis_dir,
        metadata_dir=metadata_dir,
        comparison_context=comparison_context,
        announce_manifest=True,
        generate_summary_fn=generate_summary_fn,
        generate_manifest_fn=generate_manifest_fn,
    )

    if matlab_results and python_results:
        success = matlab_results.get("success") and python_results.get("success")
        if success:
            print_reuse_guidance_fn(
                run_root,
                input_file=input_file,
                params=params_for_python,
                loop_kind=loop_kind,
            )
        return 0 if success else 1
    return 0
