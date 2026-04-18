from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

from slavv.runtime.run_state import atomic_write_json, atomic_write_text

from .paths import comparisons_root_from_path, relative_run_path, resolve_run_layout
from .status import _read_json_file, infer_run_status

_POINTER_FILES = (
    "latest_completed.txt",
    "canonical_acceptance.txt",
    "best_saved_batch.txt",
)


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


def _load_pointer_targets(comparisons_root: Path) -> dict[str, str]:
    pointers_dir = comparisons_root / "pointers"
    pointers: dict[str, str] = {}
    for file_name in _POINTER_FILES:
        pointer_path = pointers_dir / file_name
        if not pointer_path.exists():
            continue
        if content := pointer_path.read_text(encoding="utf-8").strip():
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
    atomic_write_text(pointer_file, relative_path.strip() + "\n")
    return pointer_file


def create_pointer_files(comparisons_root: Path) -> dict[str, Path]:
    """Ensure the required pointer files exist under the comparisons root."""
    pointers_dir = comparisons_root / "pointers"
    pointers_dir.mkdir(parents=True, exist_ok=True)
    created: dict[str, Path] = {}
    for file_name in _POINTER_FILES:
        path = pointers_dir / file_name
        if not path.exists():
            atomic_write_text(path, "")
        created[file_name] = path
    return created


def build_experiment_index_entry(
    run_dir: Path,
    *,
    comparisons_root: Path | None = None,
    pointer_targeted: bool = False,
) -> dict[str, Any]:
    """Build one machine-readable index row for an experiment-managed run."""
    from .manifest import _extract_parity_summary

    run_root = resolve_run_layout(run_dir)["run_root"]
    comparisons_root = comparisons_root or comparisons_root_from_path(run_root) or run_root.parent
    report = _read_json_file(resolve_run_layout(run_root)["report_file"])
    status = infer_run_status(run_root, pointer_targeted=pointer_targeted)
    entry = {
        "run_path": relative_run_path(run_root, comparisons_root),
        "timestamp": _extract_timestamp_from_name(run_root.name),
        **status,
    }
    if parity := _extract_parity_summary(report):
        entry["parity"] = parity
    return entry


def write_experiment_index(experiment_dir: Path, entries: list[dict[str, Any]]) -> Path:
    """Persist an experiment index.json payload."""
    index_path = experiment_dir / "index.json"
    atomic_write_json(index_path, {"runs": entries})
    return index_path
