"""Output-root preflight checks for comparison and MATLAB-adjacent workflows."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

DEFAULT_REQUIRED_FREE_SPACE_GB = 5.0
PREFLIGHT_STATUS_PASSED = "passed"
PREFLIGHT_STATUS_WARNING = "warning"
PREFLIGHT_STATUS_BLOCKED = "blocked"


@dataclass
class OutputRootPreflightReport:
    """Normalized decision for whether a selected output root is safe to use."""

    output_root: str
    resolved_output_root: str = ""
    preflight_status: str = PREFLIGHT_STATUS_BLOCKED
    allows_launch: bool = False
    writable: bool = False
    output_root_exists: bool = False
    output_root_created: bool = False
    free_space_gb: float | None = None
    required_space_gb: float = DEFAULT_REQUIRED_FREE_SPACE_GB
    onedrive_suspected: bool = False
    non_local_suspected: bool = False
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    recommended_action: str = ""
    cache_used: bool = False
    cache_created_at: str = ""
    cache_valid_for_seconds: int = 0

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def measure_free_space_gb(path: Path) -> float:
    """Return available free space in gigabytes for the path's backing volume."""
    target = path if path.exists() else path.parent
    usage = shutil.disk_usage(target)
    return usage.free / (1024**3)


def evaluate_output_root_preflight(
    output_root: str | Path,
    *,
    required_free_space_gb: float = DEFAULT_REQUIRED_FREE_SPACE_GB,
) -> OutputRootPreflightReport:
    """Validate an output root before a live MATLAB-enabled comparison run starts."""
    raw_output_root = Path(output_root).expanduser()
    report = OutputRootPreflightReport(
        output_root=str(raw_output_root),
        required_space_gb=required_free_space_gb,
    )

    try:
        resolved_output_root = raw_output_root.resolve(strict=False)
    except OSError as exc:
        report.errors.append(f"Could not resolve output root: {exc}")
        _finalize_preflight_report(report)
        return report

    report.resolved_output_root = str(resolved_output_root)
    report.onedrive_suspected = _is_onedrive_suspected(resolved_output_root)
    report.non_local_suspected = _is_non_local_path_suspected(raw_output_root, resolved_output_root)
    if report.onedrive_suspected:
        report.warnings.append(
            "Output root appears to be under OneDrive sync; a local non-synced drive is safer for MATLAB outputs."
        )
    if report.non_local_suspected:
        report.warnings.append(
            "Output root appears to be non-local or network-backed; MATLAB runs may be slower or less reliable there."
        )

    report.output_root_exists = resolved_output_root.exists()
    if report.output_root_exists and not resolved_output_root.is_dir():
        report.errors.append(f"Output root exists but is not a directory: {resolved_output_root}")
        _finalize_preflight_report(report)
        return report

    try:
        resolved_output_root.mkdir(parents=True, exist_ok=True)
        report.output_root_created = not report.output_root_exists
        report.output_root_exists = True
    except OSError as exc:
        report.errors.append(f"Could not create output root or its parents: {exc}")
        _finalize_preflight_report(report)
        return report

    try:
        _probe_directory_writable(resolved_output_root)
        report.writable = True
    except OSError as exc:
        report.errors.append(f"Output root is not writable: {exc}")

    try:
        report.free_space_gb = measure_free_space_gb(resolved_output_root)
    except OSError as exc:
        report.warnings.append(f"Could not determine free disk space: {exc}")
    else:
        if report.free_space_gb < required_free_space_gb:
            report.errors.append(
                "Low disk space: "
                f"{report.free_space_gb:.1f} GB available "
                f"(required minimum: {required_free_space_gb:.1f} GB)"
            )

    _finalize_preflight_report(report)
    return report


def persist_output_preflight(report: OutputRootPreflightReport, metadata_dir: Path) -> Path:
    """Persist the normalized output-root preflight report."""
    metadata_dir.mkdir(parents=True, exist_ok=True)
    report_path = metadata_dir / "output_preflight.json"
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report.to_dict(), handle, indent=2, sort_keys=True)
    return report_path


def load_output_preflight(path_or_dir: Path) -> dict[str, object] | None:
    """Load a persisted preflight report from a metadata dir or explicit file path."""
    candidate = path_or_dir / "output_preflight.json" if path_or_dir.is_dir() else path_or_dir
    if not candidate.exists():
        return None
    with open(candidate, encoding="utf-8") as handle:
        return json.load(handle)


def summarize_output_preflight(report: OutputRootPreflightReport) -> str:
    """Create a compact, human-readable preflight decision summary."""
    if report.preflight_status == PREFLIGHT_STATUS_BLOCKED:
        return "Output preflight blocked launch: " + "; ".join(report.errors)
    if report.preflight_status == PREFLIGHT_STATUS_WARNING:
        return "Output preflight passed with warnings: " + "; ".join(report.warnings)
    return "Output preflight passed."


def _probe_directory_writable(directory: Path) -> None:
    fd, probe_name = tempfile.mkstemp(
        dir=str(directory),
        prefix=".slavv-preflight-",
        suffix=".tmp",
    )
    os.close(fd)
    probe_path = Path(probe_name)
    if probe_path.exists():
        probe_path.unlink()


def _is_onedrive_suspected(path: Path) -> bool:
    lowered_parts = [part.lower() for part in path.parts]
    if any(part.startswith("onedrive") for part in lowered_parts):
        return True

    for env_name in ("OneDrive", "OneDriveConsumer", "OneDriveCommercial"):
        env_value = os.environ.get(env_name)
        if not env_value:
            continue
        try:
            env_path = Path(env_value).expanduser().resolve(strict=False)
        except OSError:
            continue
        try:
            path.relative_to(env_path)
            return True
        except ValueError:
            continue
    return False


def _is_non_local_path_suspected(raw_path: Path, resolved_path: Path) -> bool:
    raw_text = str(raw_path)
    resolved_text = str(resolved_path)
    return raw_text.startswith(("\\\\", "//")) or resolved_text.startswith(("\\\\", "//"))


def _finalize_preflight_report(report: OutputRootPreflightReport) -> None:
    if report.errors:
        report.preflight_status = PREFLIGHT_STATUS_BLOCKED
        report.allows_launch = False
        if any(message.startswith("Low disk space") for message in report.errors):
            report.recommended_action = (
                "Choose a local output root with more free space before launching MATLAB."
            )
        else:
            report.recommended_action = "Fix the blocking output-root issue before launch."
        return

    if report.warnings:
        report.preflight_status = PREFLIGHT_STATUS_WARNING
        report.allows_launch = True
        report.recommended_action = (
            "Proceed with caution; a local non-synced drive is preferred for MATLAB runs."
        )
        return

    report.preflight_status = PREFLIGHT_STATUS_PASSED
    report.allows_launch = True
    report.recommended_action = "Proceed."
