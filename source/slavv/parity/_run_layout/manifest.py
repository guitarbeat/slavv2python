from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from slavv.utils import format_size

from .inventory import RunMetadata, collect_directory_inventory, load_run_metadata
from .paths import resolve_run_layout


def _extract_parity_summary(report: dict[str, Any] | None) -> dict[str, Any]:
    if report is None:
        return {}
    parity: dict[str, Any] = {}
    for section, target in (("vertices", "vertices"), ("edges", "edges"), ("network", "strands")):
        payload = report.get(section)
        if not isinstance(payload, dict):
            continue
        if "matches_exactly" in payload:
            parity[target] = "pass" if bool(payload.get("matches_exactly")) else "fail"
        elif "matlab_count" in payload or "python_count" in payload:
            parity[target] = {
                "matlab_count": payload.get("matlab_count"),
                "python_count": payload.get("python_count"),
            }
    return parity


def _append_lifecycle_status_section(
    lines: list[str], lifecycle_status: dict[str, Any] | None
) -> None:
    if lifecycle_status is None:
        return
    lines.extend(
        [
            "## Lifecycle Status",
            "",
            f"- **State:** {lifecycle_status['state']}",
            f"- **Retention:** {lifecycle_status['retention']}",
            f"- **Quality gate:** {lifecycle_status['quality_gate']}",
        ]
    )
    if lifecycle_status.get("supersedes"):
        lines.append(f"- **Supersedes:** `{lifecycle_status['supersedes']}`")
    if lifecycle_status.get("superseded_by"):
        lines.append(f"- **Superseded by:** `{lifecycle_status['superseded_by']}`")
    if lifecycle_status.get("notes"):
        lines.append(f"- **Notes:** {lifecycle_status['notes']}")
    lines.extend(["- **Artifact:** `99_Metadata/status.json`", ""])


def _append_run_status_section(lines: list[str], run_snapshot: Any | None) -> None:
    if run_snapshot is None:
        return
    lines.extend(
        [
            "## Run Status",
            "",
            f"- **Status:** {run_snapshot.status}",
            f"- **Overall progress:** {run_snapshot.overall_progress * 100:.1f}%",
            f"- **Target stage:** {run_snapshot.target_stage}",
            f"- **Current stage:** {run_snapshot.current_stage or 'idle'}",
        ]
    )
    matlab_pipeline_task = run_snapshot.optional_tasks.get("matlab_pipeline")
    if matlab_pipeline_task is not None:
        launch_mode = str(matlab_pipeline_task.artifacts.get("launch", "") or "").strip()
        if launch_mode == "skipped":
            skip_reason = str(
                matlab_pipeline_task.artifacts.get("skip_reason", "completed_reusable_batch")
            )
            reuse_mode = str(matlab_pipeline_task.artifacts.get("reuse_mode", "") or "").strip()
            lines.extend(
                (
                    "- **MATLAB launch:** skipped due to completed reusable batch"
                    + (f" ({reuse_mode})" if reuse_mode else ""),
                    f"- **MATLAB skip reason:** {skip_reason}",
                )
            )
    lines.append("")


def _append_workflow_decision_section(
    lines: list[str],
    loop_assessment: dict[str, Any] | None,
) -> None:
    if not loop_assessment:
        return
    lines.extend(
        [
            "## Workflow Decision",
            "",
            f"- **Verdict:** {str(loop_assessment.get('verdict', 'unknown')).replace('_', ' ')}",
            f"- **Safe to reuse:** {bool(loop_assessment.get('safe_to_reuse', False))}",
            "- **Safe to analyze only:** "
            f"{bool(loop_assessment.get('safe_to_analyze_only', False))}",
            "- **Requires fresh MATLAB:** "
            f"{bool(loop_assessment.get('requires_fresh_matlab', False))}",
        ]
    )
    if recommended_action := str(loop_assessment.get("recommended_action", "") or "").strip():
        lines.append(f"- **Recommended action:** {recommended_action}")
    lines.extend(["- **Artifact:** `99_Metadata/loop_assessment.json`"])
    if reasons := loop_assessment.get("reasons") or []:
        lines.extend(["", "### Assessment Reasons"])
        lines.extend(f"- {reason}" for reason in reasons)
    lines.append("")


def _append_preflight_section(lines: list[str], preflight_report: dict[str, Any] | None) -> None:
    if not preflight_report:
        return
    lines.extend(
        [
            "## Preflight",
            "",
            f"- **Status:** {preflight_report.get('preflight_status', 'unknown')}",
            f"- **Allows launch:** {preflight_report.get('allows_launch', False)}",
        ]
    )
    if output_root := preflight_report.get("resolved_output_root") or preflight_report.get(
        "output_root", ""
    ):
        lines.append(f"- **Output root:** `{output_root}`")
    free_space_gb = preflight_report.get("free_space_gb")
    required_space_gb = preflight_report.get("required_space_gb")
    if isinstance(free_space_gb, (int, float)) and isinstance(required_space_gb, (int, float)):
        lines.append(
            "- **Free space:** "
            f"{free_space_gb:.1f} GB available / {required_space_gb:.1f} GB required"
        )
    if recommended_action := preflight_report.get("recommended_action"):
        lines.append(f"- **Recommended action:** {recommended_action}")
    lines.append("- **Artifact:** `99_Metadata/output_preflight.json`")
    warnings = preflight_report.get("warnings") or []
    errors = preflight_report.get("errors") or []
    if warnings:
        lines.extend(["", "### Preflight Warnings"])
        lines.extend(f"- {warning}" for warning in warnings)
    if errors:
        lines.extend(["", "### Preflight Errors"])
        lines.extend(f"- {error}" for error in errors)
    lines.append("")


def _append_matlab_status_section(
    lines: list[str],
    run_root: Path,
    matlab_status: dict[str, Any] | None,
    lifecycle_status: dict[str, Any] | None,
) -> None:
    if not matlab_status:
        return
    lines.extend(
        [
            "## Resume Semantics",
            "",
            f"- **MATLAB resume mode:** {matlab_status.get('matlab_resume_mode', 'unknown')}",
        ]
    )
    batch_folder = matlab_status.get("matlab_batch_folder", "")
    if batch_folder:
        lines.append(
            f"- **MATLAB batch folder:** `{_display_path(run_root, Path(str(batch_folder)))}`"
        )
    lines.append(
        "- **Last completed MATLAB stage:** "
        f"{matlab_status.get('matlab_last_completed_stage') or '(none)'}"
    )
    lines.append(f"- **Next MATLAB stage:** {matlab_status.get('matlab_next_stage') or '(none)'}")
    lines.append(
        f"- **Rerun prediction:** {matlab_status.get('matlab_rerun_prediction', 'unknown')}"
    )
    if matlab_status.get("matlab_partial_stage_artifacts_present"):
        lines.append(
            "- **Partial stage artifacts:** "
            f"{matlab_status.get('matlab_partial_stage_name') or 'present'}"
        )
    if matlab_status.get("stale_running_snapshot_suspected"):
        lines.append("- **Stale running snapshot suspected:** True")
    lines.extend(["", "## Authoritative Files", "", "- `99_Metadata/matlab_status.json`"])
    if lifecycle_status is not None:
        lines.append("- `99_Metadata/status.json`")
    if matlab_status.get("matlab_resume_state_file"):
        lines.append(
            f"- `{_display_path(run_root, Path(str(matlab_status['matlab_resume_state_file'])))}`"
        )
    if matlab_status.get("matlab_log_file"):
        lines.append(f"- `{_display_path(run_root, Path(str(matlab_status['matlab_log_file'])))}`")
    if batch_folder:
        lines.append(f"- `{_display_path(run_root, Path(str(batch_folder)))}`")
    lines.append("")
    if matlab_status.get("failure_summary"):
        lines.extend(
            ["## Failure Summary", "", f"- **Failure:** {matlab_status.get('failure_summary')}"]
        )
        if log_tail := matlab_status.get("matlab_log_tail") or []:
            lines.extend(["", "```text"])
            lines.extend(str(line) for line in log_tail[-10:])
            lines.extend(["```", ""])


def _append_matlab_health_check_section(
    lines: list[str],
    matlab_health_check: dict[str, Any] | None,
) -> None:
    if not matlab_health_check:
        return
    lines.extend(
        [
            "## MATLAB Health Check",
            "",
            f"- **Success:** {bool(matlab_health_check.get('success', False))}",
            f"- **Elapsed seconds:** {float(matlab_health_check.get('elapsed_seconds', 0.0)):.1f}",
        ]
    )
    if message := str(matlab_health_check.get("message", "") or "").strip():
        lines.append(f"- **Summary:** {message}")
    lines.extend(["- **Artifact:** `99_Metadata/matlab_health_check.json`", ""])


def generate_manifest(
    comparison_dir: Path,
    output_file: Path | None = None,
    *,
    metadata: RunMetadata | None = None,
) -> str:
    """Generate manifest/README for a comparison directory."""
    layout = resolve_run_layout(comparison_dir)
    run_root = layout["run_root"]

    if output_file is None:
        output_file = layout["manifest_file"]

    dir_name = run_root.name
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report_file = layout["report_file"]
    report = {}
    run_metadata = metadata or load_run_metadata(run_root)
    run_snapshot = run_metadata.run_snapshot
    lifecycle_status = run_metadata.lifecycle_status
    loop_assessment = run_metadata.loop_assessment
    preflight_report = run_metadata.preflight_report
    matlab_status = run_metadata.matlab_status
    matlab_health_check = run_metadata.matlab_health_check
    if report_file.exists():
        try:
            with open(report_file, encoding="utf-8") as handle:
                report = json.load(handle)
        except Exception:
            pass

    directory_inventory = collect_directory_inventory(run_root)
    inventory = directory_inventory["inventory"]
    total_size = directory_inventory["total_size"]

    lines = [
        f"# SLAVV Comparison Run: {dir_name}",
        "",
        f"**Generated:** {timestamp}",
        f"**Total Size:** {format_size(total_size)}",
        "",
    ]

    _append_lifecycle_status_section(lines, lifecycle_status)
    _append_run_status_section(lines, run_snapshot)
    _append_workflow_decision_section(lines, loop_assessment)
    _append_preflight_section(lines, preflight_report)
    _append_matlab_status_section(lines, run_root, matlab_status, lifecycle_status)
    _append_matlab_health_check_section(lines, matlab_health_check)

    if report:
        lines.extend(["## Comparison Summary", ""])
        if "performance" in report:
            perf = report["performance"]
            matlab_time = perf.get(
                "matlab_time_seconds", report.get("matlab", {}).get("elapsed_time", 0)
            )
            python_time = perf.get(
                "python_time_seconds", report.get("python", {}).get("elapsed_time", 0)
            )
            lines.extend(
                [
                    "### Performance",
                    f"- **MATLAB:** {matlab_time:.1f}s",
                    f"- **Python:** {python_time:.1f}s",
                    f"- **Speedup:** {perf.get('speedup', 0):.2f}x ({perf.get('faster', 'N/A')} faster)",
                    "",
                ]
            )
        if "vertices" in report:
            verts = report["vertices"]
            lines.extend(
                [
                    "### Vertices",
                    f"- **MATLAB:** {verts.get('matlab_count', 0):,}",
                    f"- **Python:** {verts.get('python_count', 0):,}",
                    "",
                ]
            )
        if "edges" in report:
            edges = report["edges"]
            lines.extend(
                [
                    "### Edges",
                    f"- **MATLAB:** {edges.get('matlab_count', 0):,}",
                    f"- **Python:** {edges.get('python_count', 0):,}",
                    "",
                ]
            )

    lines.extend(["## File Inventory", ""])
    vmv_files = inventory.get("vmv", [])
    casx_files = inventory.get("casx", [])
    if vmv_files or casx_files:
        lines.extend(["### 3D Visualization Files", ""])
    if vmv_files:
        lines.extend(["**VMV Files** (VessMorphoVis/Blender):"])
        lines.extend(
            f"- `{file_path.relative_to(run_root)}` ({format_size(file_path.stat().st_size)})"
            for file_path in sorted(vmv_files)
        )
        lines.append("")
    if casx_files:
        lines.extend(["**CASX Files** (CASX format):"])
        lines.extend(
            f"- `{file_path.relative_to(run_root)}` ({format_size(file_path.stat().st_size)})"
            for file_path in sorted(casx_files)
        )
        lines.append("")

    csv_files = inventory.get("csv", [])
    json_files = inventory.get("json", [])
    mat_files = inventory.get("mat", [])
    pkl_files = inventory.get("pkl", [])
    if csv_files or json_files or mat_files or pkl_files:
        lines.extend(["### Data Files", ""])
    if csv_files:
        lines.extend(
            [
                "**CSV Files:**",
                *(f"- `{file_path.relative_to(run_root)}`" for file_path in sorted(csv_files)),
                "",
            ]
        )
    if json_files:
        lines.extend(
            [
                "**JSON Files:**",
                *(f"- `{file_path.relative_to(run_root)}`" for file_path in sorted(json_files)),
                "",
            ]
        )
    if mat_files:
        lines.append("**MATLAB Files:**")
        lines.extend(f"- `{file_path.relative_to(run_root)}`" for file_path in sorted(mat_files))
        lines.append("")
    if pkl_files:
        lines.append("**Checkpoint Files:**")
        lines.extend(f"- `{file_path.relative_to(run_root)}`" for file_path in sorted(pkl_files))
        lines.append("")

    if png_files := inventory.get("png", []):
        lines.extend(["### Visualization Images", ""])
        lines.extend(f"- `{file_path.relative_to(run_root)}`" for file_path in sorted(png_files))
        lines.append("")

    manifest_content = "\n".join(lines)
    with open(output_file, "w", encoding="utf-8") as handle:
        handle.write(manifest_content)
    return manifest_content


def _display_path(run_root: Path, path: Path) -> str:
    """Render a path relative to the run root when possible."""
    try:
        return str(path.relative_to(run_root))
    except ValueError:
        return str(path)
