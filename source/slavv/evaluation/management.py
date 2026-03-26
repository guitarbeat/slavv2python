"""
Management utilities for SLAVV comparison data.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from slavv.runtime import load_run_snapshot
from slavv.utils import format_size

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


def get_directory_size(path: Path) -> int:
    """Calculate total size of directory in bytes."""
    total = 0
    try:
        for item in path.rglob("*"):
            if item.is_file():
                total += item.stat().st_size
    except PermissionError:
        pass
    return total


def load_run_info(run_dir: Path) -> dict[str, Any]:
    """Load information about a comparison run."""
    layout = resolve_run_layout(run_dir)
    run_root = layout["run_root"]
    info = {
        "name": run_root.name,
        "path": run_root,
        "size": get_directory_size(run_root),
        "has_matlab": layout["matlab_dir"].exists(),
        "has_python": layout["python_dir"].exists(),
        "has_report": layout["report_file"].exists(),
        "has_plots": layout["plots_dir"].exists(),
        "has_summary": layout["summary_file"].exists(),
    }

    # Load comparison report if exists
    if info["has_report"]:
        try:
            with open(layout["report_file"]) as f:
                report = json.load(f)
                info["matlab_time"] = report.get("matlab", {}).get("elapsed_time", 0)
                info["python_time"] = report.get("python", {}).get("elapsed_time", 0)
                info["matlab_vertices"] = report.get("matlab", {}).get("vertices_count", 0)
                info["python_vertices"] = report.get("python", {}).get("vertices_count", 0)
                info["speedup"] = report.get("performance", {}).get("speedup", 0)
        except Exception:
            pass

    return info


def create_experiment_path(base_dir: Path, label: str = "run") -> Path:
    """Create a flat, timestamped experiment path under base_dir."""
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    # Sanitize label
    safe_label = "".join([c if c.isalnum() or c in ("-", "_") else "-" for c in label])

    return base_dir / f"{timestamp}_{safe_label}"


def _first_existing(candidates: list[Path]) -> Path | None:
    """Return first existing path from candidates, else None."""
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resolve_run_root(path: Path) -> Path:
    """
    Resolve a run root from a path that may be a staged subdirectory.

    Supports:
    - run_dir/01_Input
    - run_dir/01_Input/matlab_results
    - run_dir/02_Output
    - run_dir/02_Output/python_results
    - run_dir/03_Analysis
    - run_dir/99_Metadata
    """
    if path.name in {"01_Input", "02_Output", "03_Analysis", "99_Metadata"}:
        return path.parent
    if path.name in {"matlab_results", "python_results"} and path.parent.name in {
        "01_Input",
        "02_Output",
    }:
        return path.parent.parent
    return path


def resolve_run_layout(run_dir: Path) -> dict[str, Path]:
    """Resolve canonical MATLAB/Python/report paths for legacy and staged layouts."""
    run_root = resolve_run_root(run_dir)

    matlab_candidates = [
        run_root / "01_Input" / "matlab_results",
        run_root / "matlab_results",
    ]
    python_candidates = [
        run_root / "02_Output" / "python_results",
        run_root / "python_results",
    ]
    analysis_candidates = [
        run_root / "03_Analysis",
        run_root,
    ]
    metadata_candidates = [
        run_root / "99_Metadata",
        run_root,
    ]

    matlab_dir = _first_existing(matlab_candidates) or matlab_candidates[0]
    python_dir = _first_existing(python_candidates) or python_candidates[0]
    analysis_dir = _first_existing(analysis_candidates) or analysis_candidates[0]
    metadata_dir = _first_existing(metadata_candidates) or metadata_candidates[0]

    report_candidates = [
        analysis_dir / "comparison_report.json",
        run_root / "comparison_report.json",
    ]
    summary_candidates = [
        analysis_dir / "summary.txt",
        run_root / "summary.txt",
    ]
    plots_candidates = [
        analysis_dir / "visualizations",
        run_root / "visualizations",
    ]
    manifest_candidates = [
        metadata_dir / "run_manifest.md",
        run_root / "MANIFEST.md",
    ]

    return {
        "run_root": run_root,
        "matlab_dir": matlab_dir,
        "python_dir": python_dir,
        "analysis_dir": analysis_dir,
        "metadata_dir": metadata_dir,
        "report_file": _first_existing(report_candidates) or report_candidates[0],
        "summary_file": _first_existing(summary_candidates) or summary_candidates[0],
        "plots_dir": _first_existing(plots_candidates) or plots_candidates[0],
        "manifest_file": _first_existing(manifest_candidates) or manifest_candidates[0],
    }


def list_runs(experiment_dir: Path) -> list[dict[str, Any]]:
    """List all comparison runs in the directory hierarchy."""
    if not experiment_dir.exists():
        return []

    runs = []
    seen_paths: set[Path] = set()

    def add_run(candidate_dir: Path):
        run_root = resolve_run_root(candidate_dir)
        if run_root in seen_paths:
            return
        seen_paths.add(run_root)
        runs.append(load_run_info(run_root))

    # Find directories that contain a run manifestation.
    for report in experiment_dir.rglob("comparison_report.json"):
        add_run(report.parent)

    # Standalone python runs if they don't have a report yet.
    for results in experiment_dir.rglob("python_results"):
        add_run(results.parent)

    # Standalone matlab runs are also valid run candidates.
    for results in experiment_dir.rglob("matlab_results"):
        add_run(results.parent)

    return sorted(runs, key=lambda x: x["name"], reverse=True)


def analyze_checkpoints(comparisons_dir: Path) -> list[dict[str, Any]]:
    """Analyze checkpoint files usage."""
    runs = list_runs(comparisons_dir)
    results = []

    for run in runs:
        path = run["path"]
        pkl_files = list(path.rglob("*.pkl"))
        pkl_size = sum(f.stat().st_size for f in pkl_files)

        results.append(
            {
                "name": run["name"],
                "path": path,
                "total_size": run["size"],
                "pkl_size": pkl_size,
                "pkl_count": len(pkl_files),
                "pkl_files": pkl_files,
            }
        )
    return results


def cleanup_checkpoints(run_data: dict[str, Any]) -> int:
    """Remove checkpoints from a specific run. Returns bytes freed."""
    freed = 0
    for pkl_file in run_data["pkl_files"]:
        try:
            size = pkl_file.stat().st_size
            pkl_file.unlink()
            freed += size
        except Exception:
            pass
    return freed


def get_file_inventory(directory: Path) -> dict[str, list[Path]]:
    """Get inventory of files organized by type."""
    inventory = {
        "vmv": [],
        "casx": [],
        "csv": [],
        "json": [],
        "mat": [],
        "png": [],
        "txt": [],
        "other": [],
    }

    for file_path in directory.rglob("*"):
        if file_path.is_file():
            ext = file_path.suffix.lower().lstrip(".")
            if ext in inventory:
                inventory[ext].append(file_path)
            else:
                inventory["other"].append(file_path)

    return inventory


def generate_manifest(comparison_dir: Path, output_file: Path | None = None) -> str:
    """Generate manifest/README for a comparison directory."""
    layout = resolve_run_layout(comparison_dir)
    run_root = layout["run_root"]

    if output_file is None:
        output_file = layout["manifest_file"]

    # Get directory name and timestamp
    dir_name = run_root.name
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Load comparison report
    report_file = layout["report_file"]
    report = {}
    run_snapshot = load_run_snapshot(run_root)
    if report_file.exists():
        try:
            with open(report_file) as f:
                report = json.load(f)
        except Exception:
            pass

    # Get file inventory
    inventory = get_file_inventory(run_root)

    # Calculate total size
    total_size = sum(f.stat().st_size for f in run_root.rglob("*") if f.is_file())

    # Build manifest content
    lines = []
    lines.append(f"# SLAVV Comparison Run: {dir_name}")
    lines.append("")
    lines.append(f"**Generated:** {timestamp}")
    lines.append(f"**Total Size:** {format_size(total_size)}")
    lines.append("")

    if run_snapshot is not None:
        lines.append("## Run Status")
        lines.append("")
        lines.append(f"- **Status:** {run_snapshot.status}")
        lines.append(f"- **Overall progress:** {run_snapshot.overall_progress * 100:.1f}%")
        lines.append(f"- **Target stage:** {run_snapshot.target_stage}")
        lines.append(f"- **Current stage:** {run_snapshot.current_stage or 'idle'}")
        lines.append("")

    # Comparison Summary
    if report:
        lines.append("## Comparison Summary")
        lines.append("")

        if "performance" in report:
            perf = report["performance"]
            matlab_time = perf.get(
                "matlab_time_seconds", report.get("matlab", {}).get("elapsed_time", 0)
            )
            python_time = perf.get(
                "python_time_seconds", report.get("python", {}).get("elapsed_time", 0)
            )
            lines.append("### Performance")
            lines.append(f"- **MATLAB:** {matlab_time:.1f}s")
            lines.append(f"- **Python:** {python_time:.1f}s")
            lines.append(
                f"- **Speedup:** {perf.get('speedup', 0):.2f}x ({perf.get('faster', 'N/A')} faster)"
            )
            lines.append("")

        if "vertices" in report:
            verts = report["vertices"]
            lines.append("### Vertices")
            lines.append(f"- **MATLAB:** {verts.get('matlab_count', 0):,}")
            lines.append(f"- **Python:** {verts.get('python_count', 0):,}")
            lines.append("")

        if "edges" in report:
            edges = report["edges"]
            lines.append("### Edges")
            lines.append(f"- **MATLAB:** {edges.get('matlab_count', 0):,}")
            lines.append(f"- **Python:** {edges.get('python_count', 0):,}")
            lines.append("")

    # File Inventory
    lines.append("## File Inventory")
    lines.append("")

    # 3D Visualization Files
    vmv_files = inventory.get("vmv", [])
    casx_files = inventory.get("casx", [])
    if vmv_files or casx_files:
        lines.append("### 3D Visualization Files")
        lines.append("")
        if vmv_files:
            lines.append("**VMV Files** (VessMorphoVis/Blender):")
            for f in sorted(vmv_files):
                rel_path = f.relative_to(run_root)
                size = format_size(f.stat().st_size)
                lines.append(f"- `{rel_path}` ({size})")
            lines.append("")
        if casx_files:
            lines.append("**CASX Files** (CASX format):")
            for f in sorted(casx_files):
                rel_path = f.relative_to(run_root)
                size = format_size(f.stat().st_size)
                lines.append(f"- `{rel_path}` ({size})")
            lines.append("")

    # Data Files
    csv_files = inventory.get("csv", [])
    json_files = inventory.get("json", [])
    mat_files = inventory.get("mat", [])
    if csv_files or json_files or mat_files:
        lines.append("### Data Files")
        lines.append("")
        if csv_files:
            lines.append("**CSV Files:**")
            for f in sorted(csv_files):
                rel_path = f.relative_to(run_root)
                lines.append(f"- `{rel_path}`")
            lines.append("")
        if json_files:
            lines.append("**JSON Files:**")
            for f in sorted(json_files):
                rel_path = f.relative_to(run_root)
                lines.append(f"- `{rel_path}`")
            lines.append("")
        if mat_files:
            lines.append("**MATLAB Files:**")
            for f in sorted(mat_files):
                rel_path = f.relative_to(run_root)
                lines.append(f"- `{rel_path}`")
            lines.append("")

    # Visualizations
    png_files = inventory.get("png", [])
    if png_files:
        lines.append("### Visualization Images")
        lines.append("")
        for f in sorted(png_files):
            rel_path = f.relative_to(run_root)
            lines.append(f"- `{rel_path}`")
        lines.append("")

    # Write to file
    manifest_content = "\n".join(lines)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(manifest_content)

    return manifest_content
