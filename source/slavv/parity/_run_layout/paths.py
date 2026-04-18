from __future__ import annotations

import re
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

_STAGED_DIRECTORIES = ("01_Input", "02_Output", "03_Analysis", "99_Metadata")
_EXPERIMENT_RUN_ROOT_RE = re.compile(r"^\d{8}(?:_\d{6})?(?:_.+)?$")


def create_experiment_path(base_dir: Path, label: str = "run") -> Path:
    """Create a flat, timestamped experiment path under base_dir."""
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    safe_label = "".join(c if c.isalnum() or c in ("-", "_") else "-" for c in label)
    return base_dir / f"{timestamp}_{safe_label}"


def _first_existing(candidates: list[Path]) -> Path | None:
    """Return first existing path from candidates, else None."""
    return next((candidate for candidate in candidates if candidate.exists()), None)


def is_staged_run_root(path: Path) -> bool:
    """Return True when path contains staged run-root directories."""
    if not path.exists() or not path.is_dir():
        return False
    return any((path / child).exists() for child in _STAGED_DIRECTORIES)


def is_legacy_flat_run_root(path: Path) -> bool:
    """Return True when path looks like a legacy flat run root."""
    if not path.exists() or not path.is_dir():
        return False
    return any(
        (path / child).exists()
        for child in (
            "matlab_results",
            "python_results",
            "comparison_report.json",
            "run_snapshot.json",
        )
    )


def is_experiment_grouped_run_root(path: Path) -> bool:
    """Return True when path lives under experiments/<slug>/runs/<run_root>."""
    return path.parent.name == "runs" and path.parent.parent.parent.name == "experiments"


def is_authoritative_managed_run_root(path: Path) -> bool:
    """Return True when path should be treated as a managed run root."""
    return is_staged_run_root(path) or is_legacy_flat_run_root(path)


def is_aggregate_run_container(path: Path) -> bool:
    """Return True when path contains run_* child runs but is not itself a run root."""
    if not path.exists() or not path.is_dir() or is_authoritative_managed_run_root(path):
        return False
    return any(
        (
            child.is_dir()
            and child.name.startswith("run_")
            and is_authoritative_managed_run_root(child)
        )
        for child in path.iterdir()
    )


def list_aggregate_child_runs(path: Path) -> list[Path]:
    """Return authoritative child runs for an aggregate run container."""
    if not is_aggregate_run_container(path):
        return []
    return sorted(
        [
            child
            for child in path.iterdir()
            if child.is_dir()
            and child.name.startswith("run_")
            and is_authoritative_managed_run_root(child)
        ]
    )


def classify_run_path(path: Path) -> str:
    """Classify a path for migration and inventory workflows."""
    if is_aggregate_run_container(path):
        return "aggregate_container"
    if is_experiment_grouped_run_root(path):
        return "grouped_run_root"
    if is_staged_run_root(path):
        return "staged_run_root"
    return "legacy_flat_run_root" if is_legacy_flat_run_root(path) else "unknown"


def resolve_run_root(path: Path) -> Path:
    """Resolve a run root from a path that may be a staged subdirectory."""
    if path.name in set(_STAGED_DIRECTORIES):
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

    matlab_candidates = [run_root / "01_Input" / "matlab_results", run_root / "matlab_results"]
    python_candidates = [run_root / "02_Output" / "python_results", run_root / "python_results"]
    analysis_candidates = [run_root / "03_Analysis", run_root]
    metadata_candidates = [run_root / "99_Metadata", run_root]

    matlab_dir = _first_existing(matlab_candidates) or matlab_candidates[0]
    python_dir = _first_existing(python_candidates) or python_candidates[0]
    analysis_dir = _first_existing(analysis_candidates) or analysis_candidates[0]
    metadata_dir = _first_existing(metadata_candidates) or metadata_candidates[0]

    report_candidates = [
        analysis_dir / "comparison_report.json",
        run_root / "comparison_report.json",
    ]
    summary_candidates = [analysis_dir / "summary.txt", run_root / "summary.txt"]
    plots_candidates = [analysis_dir / "visualizations", run_root / "visualizations"]
    manifest_candidates = [metadata_dir / "run_manifest.md", run_root / "MANIFEST.md"]

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


def slugify_experiment_name(value: str) -> str:
    """Normalize an experiment slug to lowercase kebab-case."""
    collapsed = re.sub(r"[^A-Za-z0-9]+", "-", value.strip().lower()).strip("-")
    return collapsed or "uncategorized"


def normalize_run_root_name(name: str) -> str:
    """Normalize a run-root name to supported date-first forms."""
    safe = re.sub(r"[^A-Za-z0-9_-]+", "-", name).strip("-")
    if _EXPERIMENT_RUN_ROOT_RE.match(safe):
        return safe

    if suffix_match := re.match(r"^(?P<label>.+?)_(?P<date>\d{8})(?:_(?P<time>\d{6}))?$", safe):
        date = suffix_match["date"]
        time = suffix_match["time"]
        label = suffix_match["label"]
        return f"{date}_{time}_{label}" if time else f"{date}_{label}"

    return safe


def comparisons_root_from_path(path: Path) -> Path | None:
    """Return the nearest comparisons-like root that owns experiments/ and pointers/."""
    resolved = path.resolve()
    for candidate in (resolved, *resolved.parents):
        if candidate.name.endswith("comparisons") or candidate.name == "slavv_comparisons":
            return candidate
        if (candidate / "experiments").exists() or (candidate / "pointers").exists():
            return candidate
    return None


def relative_run_path(path: Path, comparisons_root: Path) -> str:
    """Render a path relative to the comparisons root with POSIX separators."""
    return path.resolve().relative_to(comparisons_root.resolve()).as_posix()
