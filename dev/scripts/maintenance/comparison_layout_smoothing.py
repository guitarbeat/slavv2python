"""Inventory and consolidate SLAVV comparison layouts."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from slavv.parity.run_layout import (
    aggregate_container_rollup,
    build_experiment_index_entry,
    create_pointer_files,
    discover_run_roots,
    infer_run_status,
    is_aggregate_run_container,
    list_aggregate_child_runs,
    normalize_run_root_name,
    relative_run_path,
    slugify_experiment_name,
    write_experiment_index,
    write_pointer_file,
    write_run_status,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_COMPARISONS_ROOT = REPO_ROOT / "slavv_comparisons"
DEFAULT_REPORT_NAME = "comparison_layout_migration_report.json"
POINTER_ORDER = ("latest_completed.txt", "canonical_acceptance.txt", "best_saved_batch.txt")


def infer_experiment_slug(run_root: Path) -> str:
    """Infer a stable experiment slug from a run-root name."""
    tokens = run_root.name.split("_")
    label_tokens = tokens[2:] if len(tokens) >= 3 and tokens[0].isdigit() and tokens[1].isdigit() else tokens[1:]
    label = "-".join(label_tokens) if label_tokens else run_root.name
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


def build_target_run_path(comparisons_root: Path, run_root: Path, slug: str | None = None) -> Path:
    """Return the grouped target path for a run root."""
    experiment_slug = slug or infer_experiment_slug(run_root)
    return comparisons_root / "experiments" / experiment_slug / "runs" / normalize_run_root_name(run_root.name)


def find_doc_references_to_paths(repo_root: Path, paths: list[str]) -> list[dict[str, Any]]:
    """Return doc references containing any of the supplied path strings."""
    hits: list[dict[str, Any]] = []
    normalized_paths = [path.replace("\\", "/") for path in paths if path]
    if not normalized_paths:
        return hits
    for doc_path in (repo_root / "docs").rglob("*.md"):
        try:
            content = doc_path.read_text(encoding="utf-8")
        except OSError:
            continue
        for needle in normalized_paths:
            if needle in content.replace("\\", "/"):
                hits.append({"file": str(doc_path), "path": needle})
    return hits


def discover_comparison_layout(comparisons_root: Path) -> dict[str, Any]:
    """Discover managed runs and aggregate containers under a comparisons root."""
    managed_runs = discover_run_roots(comparisons_root)
    aggregate_containers = []
    for child in sorted(comparisons_root.iterdir()) if comparisons_root.exists() else []:
        if child.name in {"experiments", "pointers"}:
            continue
        if is_aggregate_run_container(child):
            aggregate_containers.append(
                {
                    "path": str(child),
                    "rollup": aggregate_container_rollup(child),
                    "children": [str(run) for run in list_aggregate_child_runs(child)],
                }
            )
    return {
        "managed_runs": managed_runs,
        "aggregate_containers": aggregate_containers,
    }


def _pointer_candidates(run_entries: list[dict[str, Any]]) -> dict[str, str]:
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
    if canonical:
        canonical.sort(key=lambda entry: entry["target_relative_path"], reverse=True)
        pointers["canonical_acceptance.txt"] = canonical[0]["target_relative_path"]
    elif completed:
        pointers["canonical_acceptance.txt"] = completed[0]["target_relative_path"]

    saved_batch = [
        entry for entry in completed if "saved-batch" in entry["target_relative_path"] or "saved" in entry["slug"]
    ]
    if saved_batch:
        saved_batch.sort(key=lambda entry: entry["target_relative_path"], reverse=True)
        pointers["best_saved_batch.txt"] = saved_batch[0]["target_relative_path"]
    elif completed:
        pointers["best_saved_batch.txt"] = completed[0]["target_relative_path"]
    return pointers


def build_migration_report(comparisons_root: Path, repo_root: Path | None = None) -> dict[str, Any]:
    """Build a machine-readable dry-run report for layout smoothing."""
    repo_root = repo_root or REPO_ROOT
    discovery = discover_comparison_layout(comparisons_root)
    run_entries: list[dict[str, Any]] = []
    for run_root in discovery["managed_runs"]:
        slug = infer_experiment_slug(run_root)
        target_path = build_target_run_path(comparisons_root, run_root, slug)
        current_relative = relative_run_path(run_root, comparisons_root)
        target_relative = relative_run_path(target_path, comparisons_root)
        status = infer_run_status(run_root)
        run_entries.append(
            {
                "source_path": str(run_root),
                "source_relative_path": current_relative,
                "target_path": str(target_path),
                "target_relative_path": target_relative,
                "slug": slug,
                "normalized_name": normalize_run_root_name(run_root.name),
                "status": status,
                "conflict": target_path.exists() and target_path.resolve() != run_root.resolve(),
                "action": (
                    "noop"
                    if target_path.resolve() == run_root.resolve()
                    else "move"
                ),
            }
        )

    pointer_targets = _pointer_candidates(run_entries)
    for entry in run_entries:
        if entry["target_relative_path"] in pointer_targets.values():
            entry["status"]["retention"] = "keep"

    stale_references = find_doc_references_to_paths(
        repo_root,
        [
            entry["source_relative_path"]
            for entry in run_entries
            if entry["source_relative_path"] != entry["target_relative_path"]
        ],
    )

    cleanup_candidates = [
        {
            "path": entry["source_relative_path"],
            "reason": f"retention={entry['status']['retention']} state={entry['status']['state']}",
        }
        for entry in run_entries
        if entry["status"]["state"] in {"failed", "superseded"}
        and entry["status"]["retention"] == "eligible_for_cleanup"
    ]

    report = {
        "comparisons_root": str(comparisons_root),
        "mode": "dry-run",
        "runs": run_entries,
        "aggregate_containers": discovery["aggregate_containers"],
        "pointer_proposals": pointer_targets,
        "stale_doc_references": stale_references,
        "cleanup_candidates": cleanup_candidates,
    }
    return report


def _remove_empty_parents(start: Path, stop_at: Path) -> list[str]:
    removed: list[str] = []
    current = start
    while current != stop_at and current.exists():
        try:
            current.rmdir()
        except OSError:
            break
        removed.append(str(current))
        current = current.parent
    return removed


def apply_migration_report(
    report: dict[str, Any],
    *,
    comparisons_root: Path,
    repo_root: Path | None = None,
    allow_delete: list[str] | None = None,
) -> dict[str, Any]:
    """Apply a dry-run migration report."""
    repo_root = repo_root or REPO_ROOT
    allow_delete = allow_delete or []
    applied_moves: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    removed_empty_dirs: list[str] = []
    skipped_deletions: list[dict[str, Any]] = []

    for run in report.get("runs", []):
        source_path = Path(run["source_path"])
        target_path = Path(run["target_path"])
        if run.get("conflict"):
            conflicts.append({"source": str(source_path), "target": str(target_path)})
            continue
        if run.get("action") == "move" and source_path.resolve() != target_path.resolve():
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source_path), str(target_path))
            applied_moves.append({"from": str(source_path), "to": str(target_path)})
            removed_empty_dirs.extend(_remove_empty_parents(source_path.parent, comparisons_root))
        else:
            target_path = source_path
        pointer_targeted = run["target_relative_path"] in report.get("pointer_proposals", {}).values()
        status = infer_run_status(target_path, pointer_targeted=pointer_targeted)
        write_run_status(target_path, status)

    create_pointer_files(comparisons_root)
    for name, relative_path in report.get("pointer_proposals", {}).items():
        write_pointer_file(comparisons_root / "pointers" / name, relative_path)

    experiment_entries: dict[str, list[dict[str, Any]]] = {}
    for run in discover_run_roots(comparisons_root):
        try:
            relative_path = relative_run_path(run, comparisons_root)
        except ValueError:
            continue
        pointer_targeted = relative_path in report.get("pointer_proposals", {}).values()
        slug = infer_experiment_slug(run)
        experiment_entries.setdefault(slug, []).append(
            build_experiment_index_entry(
                run,
                comparisons_root=comparisons_root,
                pointer_targeted=pointer_targeted,
            )
        )
    for slug, entries in experiment_entries.items():
        entries.sort(key=lambda entry: str(entry.get("run_path", "")), reverse=True)
        write_experiment_index(comparisons_root / "experiments" / slug, entries)

    for candidate in report.get("cleanup_candidates", []):
        relative_path = candidate["path"]
        if relative_path not in allow_delete:
            skipped_deletions.append({"path": relative_path, "reason": "not in allow-delete list"})
            continue
        target = comparisons_root / Path(relative_path)
        if target.exists() and any(target.iterdir()):
            shutil.rmtree(target)

    updated_report = build_migration_report(comparisons_root, repo_root=repo_root)
    updated_report["mode"] = "apply"
    updated_report["applied_moves"] = applied_moves
    updated_report["conflicts"] = conflicts
    updated_report["removed_empty_dirs"] = sorted(set(removed_empty_dirs))
    updated_report["skipped_deletions"] = skipped_deletions
    return updated_report


def write_report(report: dict[str, Any], report_path: Path) -> Path:
    """Write a machine-readable report."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inventory and smooth comparison-run layout.")
    parser.add_argument(
        "--comparisons-root",
        default=str(DEFAULT_COMPARISONS_ROOT),
        help="Comparison root to inventory and optionally migrate.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply moves, write status/index/pointer artifacts, and prune empty directories.",
    )
    parser.add_argument(
        "--report-path",
        default=None,
        help="Optional path for the machine-readable migration report.",
    )
    parser.add_argument(
        "--allow-delete",
        nargs="*",
        default=[],
        help="Explicit relative paths that may be deleted when marked as cleanup candidates.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    comparisons_root = Path(args.comparisons_root)
    repo_root = REPO_ROOT
    report = build_migration_report(comparisons_root, repo_root=repo_root)
    if args.apply:
        report = apply_migration_report(
            report,
            comparisons_root=comparisons_root,
            repo_root=repo_root,
            allow_delete=list(args.allow_delete),
        )
    report_path = Path(args.report_path) if args.report_path else comparisons_root / DEFAULT_REPORT_NAME
    write_report(report, report_path)
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
