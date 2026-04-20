from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

from slavv.runtime.run_state import atomic_write_text
from slavv.utils import format_size

from .._persistence import load_json_dict_or_empty, write_json_file
from .indexing import build_experiment_index_entry, create_pointer_files, write_experiment_index
from .paths import (
    comparisons_root_from_path,
    is_aggregate_run_container,
    is_experiment_grouped_run_root,
    is_legacy_flat_run_root,
    is_staged_run_root,
    list_aggregate_child_runs,
    normalize_run_root_name,
    relative_run_path,
    resolve_run_layout,
    resolve_run_root,
    slugify_experiment_name,
)
from .status import infer_run_status, write_run_status

POINTER_ORDER = ("latest_completed.txt", "canonical_acceptance.txt", "best_saved_batch.txt")
_ARTIFACT_CLEANUP_PROFILE = "managed-analysis-retention-v1"
_NO_EXTENSION_BLOB_THRESHOLD_BYTES = 1_000_000
_KNOWN_ROOT_SLUGS = {
    "20260327_150656_clean_parity": "saved-batch",
    "20260327_161610_clean_python_full": "python-full",
    "20260330_parity_full_postfix": "postfix-parity",
    "20260330_cross_compare_postfix": "postfix-cross-compare",
    "20260328_023500_matlab_consistency": "matlab-consistency",
    "20260328_142659_python_consistency": "python-consistency",
    "20260330_python_consistency_postfix": "python-consistency",
    "20260401_live_parity_retry": "live-parity",
    "20260413_release_verify": "release-verify",
    "20260413_live": "20260413",
    "live_20260413b": "20260413b",
    "20260413_live_canonical": "canonical-20260413",
}


@dataclass
class _RemovalStats:
    files_removed: int = 0
    bytes_removed: int = 0

    def record(self, file_size: int) -> None:
        self.files_removed += 1
        self.bytes_removed += int(file_size)

    def to_dict(self) -> dict[str, int]:
        return {
            "files_removed": self.files_removed,
            "bytes_removed": self.bytes_removed,
        }


def is_managed_comparisons_root(path: Path) -> bool:
    """Return whether a comparisons-like root should use managed archive behavior."""
    if not path:
        return False
    return (
        path.name == "slavv_comparisons"
        or (path / "experiments").exists()
        or (path / "pointers").exists()
    )


def managed_comparisons_root_from_path(path: Path) -> Path | None:
    """Return the owning managed comparisons root when archive semantics apply."""
    run_root = resolve_run_root(path)
    comparisons_root = comparisons_root_from_path(run_root)
    if comparisons_root is not None and is_managed_comparisons_root(comparisons_root):
        return comparisons_root
    if is_experiment_grouped_run_root(run_root):
        candidate = run_root.parents[3]
        if is_managed_comparisons_root(candidate):
            return candidate
    return None


def infer_experiment_slug(run_root: Path) -> str:
    """Infer a stable experiment slug from a run-root name or grouped layout."""
    normalized_root = resolve_run_root(run_root)
    if is_experiment_grouped_run_root(normalized_root):
        return normalized_root.parent.parent.name

    if normalized_root.name in _KNOWN_ROOT_SLUGS:
        return _KNOWN_ROOT_SLUGS[normalized_root.name]

    if normalized_root.name.startswith("run_"):
        parent = normalized_root.parent
        if parent.name in _KNOWN_ROOT_SLUGS:
            return _KNOWN_ROOT_SLUGS[parent.name]

    tokens = normalized_root.name.split("_")
    if len(tokens) >= 3 and tokens[0].isdigit() and tokens[1].isdigit():
        label_tokens = tokens[2:]
    else:
        label_tokens = tokens[1:]
    label = "-".join(label_tokens) if label_tokens else normalized_root.name
    lowered = label.lower()
    if "saved" in lowered and "batch" in lowered:
        return "saved-batch"
    if "release" in lowered and "verify" in lowered:
        return "release-verify"
    if "consistency" in lowered and "python" in lowered:
        return "python-consistency"
    if "consistency" in lowered and "matlab" in lowered:
        return "matlab-consistency"
    if "parity" in lowered and "live" in lowered:
        return "live-parity"
    return slugify_experiment_name(label)


def infer_target_run_name(run_root: Path) -> str:
    """Infer a grouped target run-root name while avoiding aggregate-child collisions."""
    normalized_root = resolve_run_root(run_root)
    if normalized_root.name.startswith("run_") and normalized_root.parent.name in _KNOWN_ROOT_SLUGS:
        return normalize_run_root_name(f"{normalized_root.parent.name}_{normalized_root.name}")
    return normalize_run_root_name(normalized_root.name)


def select_pointer_targets(run_entries: list[dict[str, Any]]) -> dict[str, str]:
    """Choose pointer targets from normalized run entries."""
    completed = [entry for entry in run_entries if entry["status"]["state"] == "completed"]
    completed.sort(key=lambda entry: entry["target_relative_path"], reverse=True)
    pointers: dict[str, str] = {}
    if completed:
        pointers["latest_completed.txt"] = completed[0]["target_relative_path"]

    canonical = [
        entry
        for entry in completed
        if entry["status"]["quality_gate"] in {"pass", "partial"}
        and entry["status"]["retention"] == "keep"
    ]
    if not canonical:
        canonical = [
            entry for entry in completed if entry["status"]["quality_gate"] in {"pass", "partial"}
        ]
    canonical.sort(key=lambda entry: entry["target_relative_path"], reverse=True)
    if canonical:
        pointers["canonical_acceptance.txt"] = canonical[0]["target_relative_path"]
    elif completed:
        pointers["canonical_acceptance.txt"] = completed[0]["target_relative_path"]

    saved_batch = [
        entry
        for entry in completed
        if "saved-batch" in entry["target_relative_path"] or "saved" in entry["slug"]
    ]
    saved_batch.sort(key=lambda entry: entry["target_relative_path"], reverse=True)
    if saved_batch:
        pointers["best_saved_batch.txt"] = saved_batch[0]["target_relative_path"]
    elif completed:
        pointers["best_saved_batch.txt"] = completed[0]["target_relative_path"]

    return pointers


def refresh_managed_archive_metadata(path: Path) -> dict[str, Any]:
    """Refresh explicit status, experiment index, and pointer files for a managed archive."""
    comparisons_root = (
        path if is_managed_comparisons_root(path) else managed_comparisons_root_from_path(path)
    )
    if comparisons_root is None:
        return {"managed_archive": False, "pointers": {}}

    run_entries: list[dict[str, Any]] = []
    for run_root in _discover_run_roots(comparisons_root):
        try:
            relative_path = relative_run_path(run_root, comparisons_root)
        except ValueError:
            continue
        run_entries.append(
            {
                "run_root": run_root,
                "slug": infer_experiment_slug(run_root),
                "target_relative_path": relative_path,
                "status": infer_run_status(run_root),
            }
        )

    pointer_targets = select_pointer_targets(run_entries)
    create_pointer_files(comparisons_root)
    for file_name in POINTER_ORDER:
        _write_pointer_target(
            comparisons_root / "pointers" / file_name, pointer_targets.get(file_name)
        )

    experiment_entries: dict[str, list[dict[str, Any]]] = {}
    pointer_target_values = set(pointer_targets.values())
    for entry in run_entries:
        run_root = entry["run_root"]
        relative_path = entry["target_relative_path"]
        pointer_targeted = relative_path in pointer_target_values
        write_run_status(run_root, infer_run_status(run_root, pointer_targeted=pointer_targeted))
        experiment_entries.setdefault(entry["slug"], []).append(
            build_experiment_index_entry(
                run_root,
                comparisons_root=comparisons_root,
                pointer_targeted=pointer_targeted,
            )
        )

    for slug, entries in experiment_entries.items():
        entries.sort(key=lambda entry: str(entry.get("run_path", "")), reverse=True)
        write_experiment_index(comparisons_root / "experiments" / slug, entries)

    return {
        "managed_archive": True,
        "comparisons_root": str(comparisons_root),
        "run_count": len(run_entries),
        "pointers": pointer_targets,
        "experiments": sorted(experiment_entries),
    }


def load_artifact_cleanup(path_or_dir: Path) -> dict[str, Any] | None:
    """Load persisted artifact cleanup metadata when present."""
    candidate = path_or_dir / "artifact_cleanup.json" if path_or_dir.is_dir() else path_or_dir
    payload = load_json_dict_or_empty(candidate)
    return payload or None


def apply_managed_archive_cleanup(run_dir: Path) -> dict[str, Any]:
    """Compact a managed archive run to the retained analysis-first surface."""
    layout = resolve_run_layout(run_dir)
    run_root = layout["run_root"]
    metadata_dir = layout["metadata_dir"]
    metadata_dir.mkdir(parents=True, exist_ok=True)
    comparisons_root = managed_comparisons_root_from_path(run_root)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "managed_archive": comparisons_root is not None,
        "profile": _ARTIFACT_CLEANUP_PROFILE,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "applied": False,
        "skip_reason": "",
        "files_removed": 0,
        "bytes_removed": 0,
        "bytes_removed_human": format_size(0),
        "empty_directories_removed": 0,
        "categories": {},
    }
    if comparisons_root is None:
        payload["skip_reason"] = "not_managed_archive"
        return _write_artifact_cleanup(metadata_dir, payload)

    report_file = layout["analysis_dir"] / "comparison_report.json"
    status = infer_run_status(run_root)
    if status["state"] != "completed":
        payload["skip_reason"] = "run_not_completed"
        return _write_artifact_cleanup(metadata_dir, payload)
    if not report_file.exists():
        payload["skip_reason"] = "comparison_report_missing"
        return _write_artifact_cleanup(metadata_dir, payload)

    category_stats: dict[str, _RemovalStats] = {}
    for path, category in _cleanup_candidates(layout):
        _remove_file(path, category=category, category_stats=category_stats, payload=payload)

    payload["applied"] = True
    payload["categories"] = {
        category: stats.to_dict()
        for category, stats in sorted(category_stats.items())
        if stats.files_removed
    }
    payload["empty_directories_removed"] = _remove_empty_directories(run_root)
    payload["bytes_removed_human"] = format_size(int(payload["bytes_removed"]))
    return _write_artifact_cleanup(metadata_dir, payload)


def _cleanup_candidates(layout: dict[str, Path]) -> list[tuple[Path, str]]:
    python_dir = layout["python_dir"]
    matlab_dir = layout["matlab_dir"]
    candidates: dict[Path, str] = {}

    for path in python_dir.rglob("*.pkl"):
        candidates[path] = "python_payloads"
    for pattern in ("*.vmv", "*.casx", "*.csv"):
        for path in python_dir.rglob(pattern):
            candidates[path] = "python_exports"

    for path in matlab_dir.rglob("*.log"):
        candidates[path] = "matlab_logs"
    for pattern in ("*.mat", "*.tif", "*.tiff"):
        for path in matlab_dir.rglob(pattern):
            candidates[path] = "matlab_batch_bulk"
    for path in matlab_dir.rglob("*"):
        if not path.is_file() or path.suffix:
            continue
        if path.stat().st_size < _NO_EXTENSION_BLOB_THRESHOLD_BYTES:
            continue
        if not _has_ancestor_named(path, "data"):
            continue
        candidates[path] = "matlab_batch_bulk"

    return sorted(candidates.items(), key=lambda item: str(item[0]))


def _discover_run_roots(search_root: Path) -> list[Path]:
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
        if is_staged_run_root(path) or is_legacy_flat_run_root(path):
            discovered.add(resolve_run_root(path))
    return sorted(discovered)


def _has_ancestor_named(path: Path, name: str) -> bool:
    return any(parent.name == name for parent in path.parents)


def _remove_file(
    path: Path,
    *,
    category: str,
    category_stats: dict[str, _RemovalStats],
    payload: dict[str, Any],
) -> None:
    if not path.exists() or not path.is_file():
        return
    try:
        file_size = int(path.stat().st_size)
        path.unlink()
    except OSError:
        return

    stats = category_stats.setdefault(category, _RemovalStats())
    stats.record(file_size)
    payload["files_removed"] = int(payload["files_removed"]) + 1
    payload["bytes_removed"] = int(payload["bytes_removed"]) + file_size


def _remove_empty_directories(root: Path) -> int:
    removed = 0
    directories = sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in directories:
        try:
            next(directory.iterdir())
        except StopIteration:
            try:
                directory.rmdir()
            except OSError:
                continue
            removed += 1
        except OSError:
            continue
    return removed


def _write_artifact_cleanup(metadata_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    write_json_file(metadata_dir / "artifact_cleanup.json", payload)
    return payload


def _write_pointer_target(pointer_file: Path, relative_path: str | None) -> None:
    pointer_file.parent.mkdir(parents=True, exist_ok=True)
    content = f"{relative_path.strip()}\n" if relative_path else ""
    atomic_write_text(pointer_file, content)
