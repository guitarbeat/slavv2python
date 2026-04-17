"""
Management utilities for SLAVV comparison data.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from slavv.runtime import load_run_snapshot
from slavv.runtime.run_state import atomic_write_json
from slavv.utils import format_size

from .matlab_status import load_matlab_status
from .preflight import load_output_preflight
from .workflow_assessment import load_loop_assessment, load_matlab_health_check

logger = logging.getLogger(__name__)

_INVENTORY_EXTENSIONS = ("vmv", "casx", "csv", "json", "mat", "pkl", "png", "txt")
_STAGED_DIRECTORIES = ("01_Input", "02_Output", "03_Analysis", "99_Metadata")
_STATUS_STATES = {"completed", "failed", "incomplete", "superseded", "archived"}
_STATUS_RETENTION = {"keep", "eligible_for_cleanup", "archive"}
_STATUS_QUALITY = {"pass", "fail", "partial", "unknown"}
_POINTER_FILES = (
    "latest_completed.txt",
    "canonical_acceptance.txt",
    "best_saved_batch.txt",
)
_EXPERIMENT_RUN_ROOT_RE = re.compile(r"^\d{8}(?:_\d{6})?(?:_.+)?$")


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


def create_experiment_path(base_dir: Path, label: str = "run") -> Path:
    """Create a flat, timestamped experiment path under base_dir."""
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    safe_label = "".join(c if c.isalnum() or c in ("-", "_") else "-" for c in label)
    return base_dir / f"{timestamp}_{safe_label}"


def _first_existing(candidates: list[Path]) -> Path | None:
    """Return first existing path from candidates, else None."""
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


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
    for child in path.iterdir():
        if (
            child.is_dir()
            and child.name.startswith("run_")
            and is_authoritative_managed_run_root(child)
        ):
            return True
    return False


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
    if is_legacy_flat_run_root(path):
        return "legacy_flat_run_root"
    return "unknown"


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


def _read_json_file(path: Path) -> dict[str, Any] | None:
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _normalize_status_payload(status: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "state": str(status.get("state", "incomplete") or "incomplete"),
        "retention": str(status.get("retention", "eligible_for_cleanup") or "eligible_for_cleanup"),
        "quality_gate": str(status.get("quality_gate", "unknown") or "unknown"),
    }
    for key in ("supersedes", "superseded_by", "notes"):
        value = status.get(key)
        if value not in (None, ""):
            payload[key] = str(value)
    return payload


def validate_run_status_payload(status: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize a run status payload."""
    payload = _normalize_status_payload(status)
    if payload["state"] not in _STATUS_STATES:
        raise ValueError(f"Unsupported run status state: {payload['state']}")
    if payload["retention"] not in _STATUS_RETENTION:
        raise ValueError(f"Unsupported run status retention: {payload['retention']}")
    if payload["quality_gate"] not in _STATUS_QUALITY:
        raise ValueError(f"Unsupported run status quality gate: {payload['quality_gate']}")
    return payload


def read_run_status(run_dir: Path) -> dict[str, Any] | None:
    """Read managed lifecycle metadata from 99_Metadata/status.json when present."""
    metadata_dir = resolve_run_layout(run_dir)["metadata_dir"]
    payload = _read_json_file(metadata_dir / "status.json")
    if payload is None:
        return None
    return validate_run_status_payload(payload)


def write_run_status(run_dir: Path, status: dict[str, Any]) -> Path:
    """Persist deterministic lifecycle metadata for a run."""
    metadata_dir = resolve_run_layout(run_dir)["metadata_dir"]
    status_file = metadata_dir / "status.json"
    atomic_write_json(status_file, validate_run_status_payload(status))
    return status_file


def infer_quality_gate(run_dir: Path) -> str:
    """Infer quality gate from explicit status or comparison report heuristics."""
    explicit = read_run_status(run_dir)
    if explicit is not None:
        return str(explicit["quality_gate"])

    report = _read_json_file(resolve_run_layout(run_dir)["report_file"])
    if report is None:
        return "unknown"

    comparisons = []
    for section in ("vertices", "edges", "strands", "network"):
        payload = report.get(section)
        if isinstance(payload, dict):
            comparisons.append(payload)

    if any("matches_exactly" in item for item in comparisons):
        if all(
            bool(item.get("matches_exactly")) for item in comparisons if "matches_exactly" in item
        ):
            return "pass"
        return "partial"
    return "unknown"


def _infer_state_from_snapshot(run_dir: Path) -> str | None:
    snapshot = load_run_snapshot(run_dir)
    if snapshot is None:
        return None
    status = str(snapshot.status or "").lower()
    if status in {"completed", "completed_target"}:
        return "completed"
    if status in {"failed", "resume_blocked"}:
        return "failed"
    if status in {"pending", "running"}:
        return "incomplete"
    return None


def _infer_state_from_artifacts(run_dir: Path) -> str | None:
    layout = resolve_run_layout(run_dir)
    if layout["report_file"].exists() or layout["summary_file"].exists():
        return "completed"
    if layout["python_dir"].exists() or layout["matlab_dir"].exists():
        return "incomplete"
    return None


def aggregate_container_rollup(container_dir: Path) -> dict[str, Any]:
    """Summarize aggregate run_* children without treating the container as a managed run."""
    child_runs = list_aggregate_child_runs(container_dir)
    child_statuses = [infer_run_status(child) for child in child_runs]
    states = [status["state"] for status in child_statuses]
    if child_statuses and all(state == "completed" for state in states):
        state = "completed"
    elif any(state == "failed" for state in states):
        state = "failed"
    else:
        state = "incomplete"

    quality_gate = "unknown"
    if child_statuses:
        qualities = {status["quality_gate"] for status in child_statuses}
        if len(qualities) == 1:
            quality_gate = next(iter(qualities))
        elif "pass" in qualities or "partial" in qualities:
            quality_gate = "partial"

    return {
        "state": state,
        "retention": "eligible_for_cleanup",
        "quality_gate": quality_gate,
        "child_runs": [str(child) for child in child_runs],
        "child_states": states,
    }


def infer_run_status(run_dir: Path, *, pointer_targeted: bool = False) -> dict[str, Any]:
    """Infer lifecycle metadata using explicit status, snapshots, artifacts, then aggregate rollup."""
    explicit = read_run_status(run_dir)
    if explicit is not None:
        inferred = dict(explicit)
    elif is_aggregate_run_container(run_dir):
        inferred = aggregate_container_rollup(run_dir)
    else:
        inferred = {
            "state": _infer_state_from_snapshot(run_dir)
            or _infer_state_from_artifacts(run_dir)
            or "incomplete",
            "retention": "eligible_for_cleanup",
            "quality_gate": infer_quality_gate(run_dir),
        }
    if pointer_targeted:
        inferred["retention"] = "keep"
    return validate_run_status_payload(inferred)


def _extract_timestamp_from_name(name: str) -> str | None:
    for fmt in ("%Y%m%d_%H%M%S", "%Y%m%d"):
        prefix = 15 if fmt == "%Y%m%d_%H%M%S" else 8
        candidate = name[:prefix]
        try:
            parsed = datetime.strptime(candidate, fmt)
        except ValueError:
            continue
        return parsed.isoformat()
    return None


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


def slugify_experiment_name(value: str) -> str:
    """Normalize an experiment slug to lowercase kebab-case."""
    collapsed = re.sub(r"[^A-Za-z0-9]+", "-", value.strip().lower()).strip("-")
    return collapsed or "uncategorized"


def normalize_run_root_name(name: str) -> str:
    """Normalize a run-root name to supported date-first forms."""
    safe = re.sub(r"[^A-Za-z0-9_-]+", "-", name).strip("-")
    if _EXPERIMENT_RUN_ROOT_RE.match(safe):
        return safe

    suffix_match = re.match(r"^(?P<label>.+?)_(?P<date>\d{8})(?:_(?P<time>\d{6}))?$", safe)
    if suffix_match:
        date = suffix_match.group("date")
        time = suffix_match.group("time")
        label = suffix_match.group("label")
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


def _load_pointer_targets(comparisons_root: Path) -> dict[str, str]:
    pointers_dir = comparisons_root / "pointers"
    pointers: dict[str, str] = {}
    for file_name in _POINTER_FILES:
        pointer_path = pointers_dir / file_name
        if not pointer_path.exists():
            continue
        content = pointer_path.read_text(encoding="utf-8").strip()
        if content:
            pointers[file_name] = content
    return pointers


def read_pointer_file(pointer_file: Path) -> str | None:
    """Read a pointer file and return the single stored relative path."""
    if not pointer_file.exists():
        return None
    content = pointer_file.read_text(encoding="utf-8").strip()
    return content or None


def write_pointer_file(pointer_file: Path, relative_path: str) -> Path:
    """Persist a pointer file containing exactly one relative path."""
    pointer_file.parent.mkdir(parents=True, exist_ok=True)
    pointer_file.write_text(relative_path.strip() + "\n", encoding="utf-8")
    return pointer_file


def create_pointer_files(comparisons_root: Path) -> dict[str, Path]:
    """Ensure the required pointer files exist under the comparisons root."""
    pointers_dir = comparisons_root / "pointers"
    pointers_dir.mkdir(parents=True, exist_ok=True)
    created: dict[str, Path] = {}
    for file_name in _POINTER_FILES:
        path = pointers_dir / file_name
        if not path.exists():
            path.write_text("", encoding="utf-8")
        created[file_name] = path
    return created


def build_experiment_index_entry(
    run_dir: Path,
    *,
    comparisons_root: Path | None = None,
    pointer_targeted: bool = False,
) -> dict[str, Any]:
    """Build one machine-readable index row for an experiment-managed run."""
    run_root = resolve_run_layout(run_dir)["run_root"]
    comparisons_root = comparisons_root or comparisons_root_from_path(run_root) or run_root.parent
    report = _read_json_file(resolve_run_layout(run_root)["report_file"])
    status = infer_run_status(run_root, pointer_targeted=pointer_targeted)
    entry = {
        "run_path": relative_run_path(run_root, comparisons_root),
        "timestamp": _extract_timestamp_from_name(run_root.name),
        **status,
    }
    parity = _extract_parity_summary(report)
    if parity:
        entry["parity"] = parity
    return entry


def write_experiment_index(experiment_dir: Path, entries: list[dict[str, Any]]) -> Path:
    """Persist an experiment index.json payload."""
    index_path = experiment_dir / "index.json"
    atomic_write_json(index_path, {"runs": entries})
    return index_path


def discover_run_roots(search_root: Path) -> list[Path]:
    """Discover authoritative managed run roots under legacy and grouped layouts."""
    if not search_root.exists():
        return []

    discovered: set[Path] = set()
    for path in [search_root, *search_root.rglob("*")]:
        if not path.is_dir():
            continue
        if is_aggregate_run_container(path):
            for child in list_aggregate_child_runs(path):
                discovered.add(resolve_run_root(child))
            continue
        if is_authoritative_managed_run_root(path):
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


def _append_lifecycle_status_section(lines: list[str], lifecycle_status: dict[str, Any] | None) -> None:
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
            lines.append(
                "- **MATLAB launch:** skipped due to completed reusable batch"
                + (f" ({reuse_mode})" if reuse_mode else "")
            )
            lines.append(f"- **MATLAB skip reason:** {skip_reason}")
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
    recommended_action = str(loop_assessment.get("recommended_action", "") or "").strip()
    if recommended_action:
        lines.append(f"- **Recommended action:** {recommended_action}")
    lines.extend(["- **Artifact:** `99_Metadata/loop_assessment.json`"])
    reasons = loop_assessment.get("reasons") or []
    if reasons:
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
    output_root = preflight_report.get("resolved_output_root") or preflight_report.get(
        "output_root", ""
    )
    if output_root:
        lines.append(f"- **Output root:** `{output_root}`")
    free_space_gb = preflight_report.get("free_space_gb")
    required_space_gb = preflight_report.get("required_space_gb")
    if isinstance(free_space_gb, (int, float)) and isinstance(required_space_gb, (int, float)):
        lines.append(
            "- **Free space:** "
            f"{free_space_gb:.1f} GB available / {required_space_gb:.1f} GB required"
        )
    recommended_action = preflight_report.get("recommended_action")
    if recommended_action:
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
        lines.append(f"- **MATLAB batch folder:** `{_display_path(run_root, Path(str(batch_folder)))}`")
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
        lines.append(f"- `{_display_path(run_root, Path(str(matlab_status['matlab_resume_state_file'])))}`")
    if matlab_status.get("matlab_log_file"):
        lines.append(f"- `{_display_path(run_root, Path(str(matlab_status['matlab_log_file'])))}`")
    if batch_folder:
        lines.append(f"- `{_display_path(run_root, Path(str(batch_folder)))}`")
    lines.append("")
    if matlab_status.get("failure_summary"):
        lines.extend(["## Failure Summary", "", f"- **Failure:** {matlab_status.get('failure_summary')}"])
        log_tail = matlab_status.get("matlab_log_tail") or []
        if log_tail:
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
    message = str(matlab_health_check.get("message", "") or "").strip()
    if message:
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
            for file_path in sorted(vmv_files):
                lines.append(
                    f"- `{file_path.relative_to(run_root)}` ({format_size(file_path.stat().st_size)})"
                )
            lines.append("")
        if casx_files:
            lines.extend(["**CASX Files** (CASX format):"])
            for file_path in sorted(casx_files):
                lines.append(
                    f"- `{file_path.relative_to(run_root)}` ({format_size(file_path.stat().st_size)})"
                )
            lines.append("")

    csv_files = inventory.get("csv", [])
    json_files = inventory.get("json", [])
    mat_files = inventory.get("mat", [])
    pkl_files = inventory.get("pkl", [])
    if csv_files or json_files or mat_files or pkl_files:
        lines.extend(["### Data Files", ""])
        if csv_files:
            lines.append("**CSV Files:**")
            lines.extend(
                f"- `{file_path.relative_to(run_root)}`" for file_path in sorted(csv_files)
            )
            lines.append("")
        if json_files:
            lines.append("**JSON Files:**")
            lines.extend(
                f"- `{file_path.relative_to(run_root)}`" for file_path in sorted(json_files)
            )
            lines.append("")
        if mat_files:
            lines.append("**MATLAB Files:**")
            lines.extend(
                f"- `{file_path.relative_to(run_root)}`" for file_path in sorted(mat_files)
            )
            lines.append("")
        if pkl_files:
            lines.append("**Checkpoint Files:**")
            lines.extend(
                f"- `{file_path.relative_to(run_root)}`" for file_path in sorted(pkl_files)
            )
            lines.append("")

    png_files = inventory.get("png", [])
    if png_files:
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
