"""Normalized MATLAB resume and failure-status inspection for comparison runs."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from slavv.io.matlab_parser import load_mat_file_safe

WORKFLOW_STAGES = ("energy", "vertices", "edges", "network")
DEFAULT_LOG_TAIL_LINES = 20
DEFAULT_STALE_RUNNING_SECONDS = 15 * 60
_CHUNK_FILE_PATTERN = re.compile(r"^(?P<observed>\d+) of (?P<total>\d+)$")


@dataclass
class MatlabStatusReport:
    """Normalized view of MATLAB rerun semantics and recent failure evidence."""

    output_directory: str
    input_file: str = ""
    matlab_resume_state_file: str = ""
    matlab_resume_state_present: bool = False
    matlab_resume_state_status: str = ""
    matlab_resume_state_updated_at: str = ""
    matlab_log_file: str = ""
    matlab_log_present: bool = False
    matlab_batch_folder: str = ""
    matlab_batch_timestamp: str = ""
    matlab_batch_complete: bool = False
    matlab_resume_mode: str = "fresh"
    matlab_last_completed_stage: str = ""
    matlab_next_stage: str = "energy"
    matlab_partial_stage_artifacts_present: bool = False
    matlab_partial_stage_name: str = ""
    matlab_partial_stage_details: dict[str, Any] = field(default_factory=dict)
    matlab_rerun_prediction: str = ""
    stale_running_snapshot_suspected: bool = False
    failure_summary: str = ""
    matlab_log_tail: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    authoritative_files: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def inspect_matlab_status(
    output_directory: str | Path,
    *,
    input_file: str | Path | None = None,
    log_tail_lines: int = DEFAULT_LOG_TAIL_LINES,
    stale_running_seconds: int = DEFAULT_STALE_RUNNING_SECONDS,
) -> MatlabStatusReport:
    """Inspect MATLAB artifacts and derive rerun semantics."""
    output_path = Path(output_directory)
    normalized_input = _normalize_compare_path(input_file) if input_file else ""
    resume_state_path = output_path / "matlab_resume_state.json"
    log_path = output_path / "matlab_run.log"
    resume_state = _load_json_file(resume_state_path)

    report = MatlabStatusReport(
        output_directory=str(output_path),
        input_file=str(input_file) if input_file else "",
        matlab_resume_state_file=str(resume_state_path),
        matlab_resume_state_present=bool(resume_state),
        matlab_resume_state_status=str(resume_state.get("status", "")),
        matlab_resume_state_updated_at=str(resume_state.get("updated_at", "")),
        matlab_log_file=str(log_path),
        matlab_log_present=log_path.exists(),
    )
    report.authoritative_files = {
        "resume_state": str(resume_state_path),
        "matlab_log": str(log_path),
    }

    batch_folder = _find_matching_batch_folder(output_path, normalized_input, resume_state)
    roi_names: list[str] = []
    if batch_folder is not None:
        report.matlab_batch_folder = str(batch_folder)
        report.matlab_batch_timestamp = _extract_batch_timestamp(batch_folder)
        report.authoritative_files["batch_folder"] = str(batch_folder)
        _optional_inputs, roi_names = _load_batch_settings(batch_folder)
        report.matlab_last_completed_stage = _detect_completed_stage(batch_folder, roi_names)
        report.matlab_batch_complete = report.matlab_last_completed_stage == "network"
        report.matlab_next_stage = _next_stage_after(report.matlab_last_completed_stage)
    else:
        report.matlab_next_stage = "energy"

    running_stage = _running_stage_from_status(report.matlab_resume_state_status)
    partial_stage = running_stage or report.matlab_next_stage
    if batch_folder is not None and partial_stage:
        partial_details = _detect_partial_stage_artifacts(batch_folder, partial_stage, roi_names)
        if partial_details:
            report.matlab_partial_stage_artifacts_present = True
            report.matlab_partial_stage_name = partial_stage
            report.matlab_partial_stage_details = partial_details

    report.matlab_log_tail = _tail_log_file(log_path, line_count=log_tail_lines)
    report.failure_summary = _extract_failure_summary(report.matlab_log_tail)
    report.stale_running_snapshot_suspected = _is_stale_running_snapshot(
        report,
        log_path,
        stale_running_seconds=stale_running_seconds,
    )
    if report.stale_running_snapshot_suspected:
        report.warnings.append(
            "MATLAB resume state still says running, but the available log/evidence suggests the process ended."
        )

    report.matlab_resume_mode = _derive_resume_mode(report)
    report.matlab_rerun_prediction = _derive_rerun_prediction(report)
    return report


def persist_matlab_status(report: MatlabStatusReport, metadata_dir: Path) -> Path:
    """Persist normalized MATLAB status metadata."""
    metadata_dir.mkdir(parents=True, exist_ok=True)
    report_path = metadata_dir / "matlab_status.json"
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report.to_dict(), handle, indent=2, sort_keys=True)
    return report_path


def load_matlab_status(path_or_dir: Path) -> dict[str, Any] | None:
    """Load a persisted MATLAB status report."""
    candidate = path_or_dir / "matlab_status.json" if path_or_dir.is_dir() else path_or_dir
    if not candidate.exists():
        return None
    with open(candidate, encoding="utf-8") as handle:
        return json.load(handle)


def persist_matlab_failure_summary(report: MatlabStatusReport, metadata_dir: Path) -> Path | None:
    """Persist a compact MATLAB failure summary when failure evidence exists."""
    if not report.failure_summary and not report.stale_running_snapshot_suspected:
        return None

    metadata_dir.mkdir(parents=True, exist_ok=True)
    failure_path = metadata_dir / "matlab_failure_summary.json"
    payload = {
        "matlab_batch_folder": report.matlab_batch_folder,
        "matlab_resume_mode": report.matlab_resume_mode,
        "matlab_last_completed_stage": report.matlab_last_completed_stage,
        "matlab_next_stage": report.matlab_next_stage,
        "matlab_rerun_prediction": report.matlab_rerun_prediction,
        "failure_summary": report.failure_summary,
        "stale_running_snapshot_suspected": report.stale_running_snapshot_suspected,
        "matlab_resume_state_status": report.matlab_resume_state_status,
        "matlab_resume_state_file": report.matlab_resume_state_file,
        "matlab_log_file": report.matlab_log_file,
        "matlab_log_tail": report.matlab_log_tail,
    }
    with open(failure_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return failure_path


def summarize_matlab_status(report: MatlabStatusReport) -> str:
    """Create a compact user-facing summary line for status surfaces."""
    if report.failure_summary:
        return f"{report.matlab_rerun_prediction} Last failure: {report.failure_summary}"
    if report.stale_running_snapshot_suspected:
        return f"{report.matlab_rerun_prediction} Existing running marker looks stale."
    return report.matlab_rerun_prediction


def _load_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _tail_log_file(path: Path, *, line_count: int) -> list[str]:
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            lines = handle.read().splitlines()
    except OSError:
        return []
    return [line for line in lines[-line_count:] if line.strip()]


def _find_matching_batch_folder(
    output_directory: Path,
    normalized_input: str,
    resume_state: dict[str, Any],
) -> Path | None:
    preferred_batch = _preferred_state_batch_folder(output_directory, resume_state)
    if preferred_batch is not None and _batch_matches_input(preferred_batch, normalized_input):
        return preferred_batch

    batch_folders = sorted(
        path for path in output_directory.iterdir() if path.is_dir() and path.name.startswith("batch_")
    ) if output_directory.exists() else []
    for batch_folder in reversed(batch_folders):
        if _batch_matches_input(batch_folder, normalized_input):
            return batch_folder
    return None


def _preferred_state_batch_folder(
    output_directory: Path,
    resume_state: dict[str, Any],
) -> Path | None:
    batch_folder = str(resume_state.get("batch_folder", "") or "").strip()
    if batch_folder:
        candidate = Path(batch_folder)
        if candidate.exists():
            return candidate

    batch_timestamp = str(resume_state.get("batch_timestamp", "") or "").strip()
    if not batch_timestamp:
        return None
    candidate = output_directory / f"batch_{batch_timestamp}"
    return candidate if candidate.exists() else None


def _batch_matches_input(batch_folder: Path, normalized_input: str) -> bool:
    if not batch_folder.exists():
        return False
    if not normalized_input:
        return True

    optional_inputs, _roi_names = _load_batch_settings(batch_folder)
    for candidate in optional_inputs:
        if _normalize_compare_path(candidate) == normalized_input:
            return True
    return False


def _load_batch_settings(batch_folder: Path) -> tuple[list[str], list[str]]:
    settings_dir = batch_folder / "settings"
    batch_file = settings_dir / "batch.mat"
    if not batch_file.exists():
        batch_file = settings_dir / "batch"
    if not batch_file.exists():
        return [], []

    data = load_mat_file_safe(batch_file)
    if not data:
        return [], []

    optional_inputs = _ensure_list_of_strings(data.get("optional_input"))
    roi_names = _ensure_list_of_strings(data.get("ROI_names"))
    return optional_inputs, roi_names


def _ensure_list_of_strings(raw_value: Any) -> list[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, str):
        return [raw_value]
    if isinstance(raw_value, bytes):
        return [raw_value.decode("utf-8", errors="replace")]

    try:
        values = list(raw_value)
    except TypeError:
        return [str(raw_value)]

    normalized = []
    for value in values:
        if isinstance(value, bytes):
            normalized.append(value.decode("utf-8", errors="replace"))
        else:
            normalized.append(str(value))
    return normalized


def _normalize_compare_path(path_value: str | Path | None) -> str:
    if path_value in (None, ""):
        return ""
    path_text = str(path_value).replace("/", os.sep).replace("\\", os.sep)
    try:
        normalized = str(Path(path_text).resolve(strict=False))
    except OSError:
        normalized = path_text
    return normalized.lower() if os.name == "nt" else normalized


def _detect_completed_stage(batch_folder: Path, roi_names: list[str]) -> str:
    completed_stage = ""
    if _stage_complete(batch_folder, roi_names, "energy"):
        completed_stage = "energy"
    if _stage_complete(batch_folder, roi_names, "vertices"):
        completed_stage = "vertices"
    if _stage_complete(batch_folder, roi_names, "edges"):
        completed_stage = "edges"
    if _stage_complete(batch_folder, roi_names, "network"):
        completed_stage = "network"
    return completed_stage


def _stage_complete(batch_folder: Path, roi_names: list[str], stage_name: str) -> bool:
    if not roi_names:
        return False

    if stage_name == "energy":
        for roi_name in roi_names:
            energy_files = [
                path
                for path in (batch_folder / "data").glob(f"energy_*{roi_name}")
                if path.is_file()
            ]
            chunk_dirs = [
                path
                for path in (batch_folder / "data").glob(f"energy_*{roi_name}_chunks_octave_*")
                if path.is_dir()
            ]
            if not energy_files or chunk_dirs:
                return False
        return True

    vectors_dir = batch_folder / "vectors"
    required_prefixes = {
        "vertices": ("vertices_", "curated_vertices_"),
        "edges": ("edges_", "curated_edges_"),
        "network": ("network_",),
    }.get(stage_name, ())
    if not required_prefixes:
        return False

    for roi_name in roi_names:
        for prefix in required_prefixes:
            listing = list(vectors_dir.glob(f"{prefix}*{roi_name}.mat"))
            if not listing:
                return False
    return True


def _next_stage_after(last_completed_stage: str) -> str:
    if not last_completed_stage:
        return "energy"
    try:
        stage_index = WORKFLOW_STAGES.index(last_completed_stage)
    except ValueError:
        return "energy"
    next_index = stage_index + 1
    return WORKFLOW_STAGES[next_index] if next_index < len(WORKFLOW_STAGES) else ""


def _running_stage_from_status(status: str) -> str:
    if status.startswith("running:"):
        return status.split(":", 1)[1]
    return ""


def _detect_partial_stage_artifacts(
    batch_folder: Path,
    stage_name: str,
    roi_names: list[str],
) -> dict[str, Any]:
    if not batch_folder.exists():
        return {}
    if not roi_names:
        return {}

    if stage_name == "energy":
        data_dir = batch_folder / "data"
        chunk_dirs = []
        energy_files = []
        for roi_name in roi_names:
            chunk_dirs.extend(
                path
                for path in data_dir.glob(f"energy_*{roi_name}_chunks_octave_*")
                if path.is_dir()
            )
            energy_files.extend(
                path for path in data_dir.glob(f"energy_*{roi_name}") if path.is_file()
            )
        if not chunk_dirs and not energy_files:
            return {}

        observed_chunks = 0
        total_chunks = 0
        for chunk_dir in chunk_dirs:
            for chunk_file in chunk_dir.iterdir():
                if not chunk_file.is_file():
                    continue
                observed_chunks += 1
                match = _CHUNK_FILE_PATTERN.match(chunk_file.name)
                if match:
                    total_chunks = max(total_chunks, int(match.group("total")))

        details: dict[str, Any] = {
            "artifact_count": len(energy_files) + len(chunk_dirs),
            "chunk_directory_count": len(chunk_dirs),
            "energy_file_count": len(energy_files),
        }
        if observed_chunks:
            details["observed_chunk_count"] = observed_chunks
        if total_chunks:
            details["expected_chunk_count"] = total_chunks
            details["chunk_completion_ratio"] = observed_chunks / total_chunks
        return details

    vectors_dir = batch_folder / "vectors"
    prefixes = {
        "vertices": ("vertices_", "curated_vertices_"),
        "edges": ("edges_", "curated_edges_"),
        "network": ("network_",),
    }.get(stage_name, ())
    artifacts = []
    for roi_name in roi_names:
        for prefix in prefixes:
            artifacts.extend(vectors_dir.glob(f"{prefix}*{roi_name}.mat"))
    if not artifacts:
        return {}
    return {"artifact_count": len(artifacts)}


def _extract_failure_summary(log_tail: list[str]) -> str:
    for line in reversed(log_tail):
        stripped = line.strip()
        if stripped.startswith("ERROR:"):
            return stripped
    for line in reversed(log_tail):
        stripped = line.strip()
        if "exit code:" in stripped.lower() and not stripped.endswith("exit code: 0"):
            return stripped
    return ""


def _is_stale_running_snapshot(
    report: MatlabStatusReport,
    log_path: Path,
    *,
    stale_running_seconds: int,
) -> bool:
    if not report.matlab_resume_state_status.startswith("running:"):
        return False
    if report.failure_summary or report.matlab_batch_complete:
        return True

    timestamps = []
    if report.matlab_resume_state_file and Path(report.matlab_resume_state_file).exists():
        timestamps.append(Path(report.matlab_resume_state_file).stat().st_mtime)
    if log_path.exists():
        timestamps.append(log_path.stat().st_mtime)
    if not timestamps:
        return False
    newest_timestamp = max(timestamps)
    return (time.time() - newest_timestamp) > stale_running_seconds


def _derive_resume_mode(report: MatlabStatusReport) -> str:
    if not report.matlab_batch_folder:
        return "fresh"
    if report.matlab_batch_complete:
        return "complete-noop"
    if report.matlab_partial_stage_artifacts_present and report.matlab_next_stage:
        return "restart-current-stage"
    if report.matlab_next_stage:
        return "resume-stage"
    return "fresh"


def _derive_rerun_prediction(report: MatlabStatusReport) -> str:
    if report.matlab_resume_mode == "fresh":
        return "No reusable MATLAB batch found; rerun will create a new batch and start at energy."

    batch_label = Path(report.matlab_batch_folder).name if report.matlab_batch_folder else "batch"
    if report.matlab_resume_mode == "complete-noop":
        return f"{batch_label} is already complete; rerun should be a no-op unless inputs change."

    if report.matlab_resume_mode == "restart-current-stage":
        prediction = (
            f"Rerun will reuse {batch_label} but restart {report.matlab_next_stage} from the stage boundary."
        )
        if report.matlab_partial_stage_artifacts_present and report.matlab_partial_stage_name:
            prediction = (
                f"{prediction[:-1]} Partial {report.matlab_partial_stage_name} artifacts were found."
            )
        return prediction

    return f"Rerun will reuse {batch_label} and start at {report.matlab_next_stage}."


def _extract_batch_timestamp(batch_folder: Path) -> str:
    name = batch_folder.name
    return name[6:] if name.startswith("batch_") else name
