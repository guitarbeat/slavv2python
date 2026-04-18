from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

from slavv.runtime import load_run_snapshot

from ..matlab_status import load_matlab_status
from ..preflight import load_output_preflight
from ..workflow_assessment import load_loop_assessment, load_matlab_health_check
from .indexing import _load_pointer_targets
from .paths import (
    classify_run_path,
    comparisons_root_from_path,
    is_aggregate_run_container,
    is_legacy_flat_run_root,
    is_staged_run_root,
    relative_run_path,
    resolve_run_layout,
    resolve_run_root,
)
from .status import infer_run_status, read_run_status

_INVENTORY_EXTENSIONS = ("vmv", "casx", "csv", "json", "mat", "pkl", "png", "txt")


@dataclass
class RunMetadata:
    """Unified view of persisted run metadata for manifests and status readers."""

    run_root: Path
    layout_kind: str
    run_snapshot: Any | None
    lifecycle_status: dict[str, Any] | None
    loop_assessment: dict[str, Any] | None
    preflight_report: dict[str, Any] | None
    matlab_status: dict[str, Any] | None
    matlab_health_check: dict[str, Any] | None


def load_run_metadata(run_dir: Path) -> RunMetadata:
    """Load all persisted metadata for a run with a single layout resolution."""
    layout = resolve_run_layout(run_dir)
    run_root = layout["run_root"]
    metadata_dir = layout["metadata_dir"]
    if is_staged_run_root(run_root):
        layout_kind = "staged"
    elif is_legacy_flat_run_root(run_root):
        layout_kind = "legacy_flat"
    else:
        layout_kind = "unknown"

    return RunMetadata(
        run_root=run_root,
        layout_kind=layout_kind,
        run_snapshot=load_run_snapshot(run_root),
        lifecycle_status=read_run_status(run_root),
        loop_assessment=load_loop_assessment(metadata_dir),
        preflight_report=load_output_preflight(metadata_dir),
        matlab_status=load_matlab_status(metadata_dir),
        matlab_health_check=load_matlab_health_check(metadata_dir),
    )


def _build_empty_inventory() -> dict[str, list[Path]]:
    """Create the normalized file inventory buckets used by manifests and run info."""
    inventory: dict[str, list[Path]] = {ext: [] for ext in _INVENTORY_EXTENSIONS}
    inventory["other"] = []
    return inventory


def collect_directory_inventory(path: Path) -> dict[str, Any]:
    """Scan a directory once and return both total size and extension-grouped inventory."""
    inventory = _build_empty_inventory()
    total_size = 0

    try:
        for item in path.rglob("*"):
            if not item.is_file():
                continue
            size = item.stat().st_size
            total_size += size
            ext = item.suffix.lower().lstrip(".")
            inventory[ext if ext in inventory else "other"].append(item)
    except OSError:
        pass

    return {"total_size": total_size, "inventory": inventory}


def get_directory_size(path: Path) -> int:
    """Calculate total size of directory in bytes."""
    return int(collect_directory_inventory(path)["total_size"])


def discover_run_roots(search_root: Path) -> list[Path]:
    """Discover authoritative managed run roots under legacy and grouped layouts."""
    if not search_root.exists():
        return []

    discovered: set[Path] = set()
    for path in [search_root, *search_root.rglob("*")]:
        if not path.is_dir():
            continue
        if is_aggregate_run_container(path):
            for child in path.iterdir():
                if child.is_dir() and child.name.startswith("run_"):
                    discovered.add(resolve_run_root(child))
            continue
        if is_staged_run_root(path) or is_legacy_flat_run_root(path):
            discovered.add(resolve_run_root(path))
    return sorted(discovered)


def list_runs(experiment_dir: Path) -> list[dict[str, Any]]:
    """List all comparison runs in the directory hierarchy."""
    if not experiment_dir.exists():
        return []

    comparisons_root = comparisons_root_from_path(experiment_dir) or experiment_dir
    pointer_targets = set(_load_pointer_targets(comparisons_root).values())
    runs = []
    for run_root in discover_run_roots(experiment_dir):
        pointer_targeted = relative_run_path(run_root, comparisons_root) in pointer_targets
        runs.append(load_run_info(run_root, pointer_targeted=pointer_targeted))
    return sorted(runs, key=lambda item: item["name"], reverse=True)


def load_run_info(run_dir: Path, *, pointer_targeted: bool = False) -> dict[str, Any]:
    """Load information about a comparison run."""
    layout = resolve_run_layout(run_dir)
    run_root = layout["run_root"]
    directory_inventory = collect_directory_inventory(run_root)
    status = infer_run_status(run_root, pointer_targeted=pointer_targeted)
    info = {
        "name": run_root.name,
        "path": run_root,
        "size": directory_inventory["total_size"],
        "has_matlab": layout["matlab_dir"].exists(),
        "has_python": layout["python_dir"].exists(),
        "has_report": layout["report_file"].exists(),
        "has_plots": layout["plots_dir"].exists(),
        "has_summary": layout["summary_file"].exists(),
        "run_shape": classify_run_path(run_root),
        "state": status["state"],
        "retention": status["retention"],
        "quality_gate": status["quality_gate"],
    }

    if info["has_report"]:
        try:
            with open(layout["report_file"], encoding="utf-8") as handle:
                report = json.load(handle)
            info["matlab_time"] = report.get("matlab", {}).get("elapsed_time", 0)
            info["python_time"] = report.get("python", {}).get("elapsed_time", 0)
            info["matlab_vertices"] = report.get("matlab", {}).get("vertices_count", 0)
            info["python_vertices"] = report.get("python", {}).get("vertices_count", 0)
            info["speedup"] = report.get("performance", {}).get("speedup", 0)
        except Exception:
            pass

    return info


def analyze_checkpoints(comparisons_dir: Path) -> list[dict[str, Any]]:
    """Analyze checkpoint files usage."""
    runs = list_runs(comparisons_dir)
    results = []

    for run in runs:
        path = run["path"]
        pkl_files = list(path.rglob("*.pkl"))
        pkl_size = sum(file_path.stat().st_size for file_path in pkl_files)
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
    return collect_directory_inventory(directory)["inventory"]
