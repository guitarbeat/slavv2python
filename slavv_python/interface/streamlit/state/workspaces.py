"""Read-only discovery helpers for Streamlit pipeline workspaces."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from slavv_python.engine.state import load_run_snapshot
from slavv_python.engine.state.layout import resolve_run_layout

from .workflow import STAGE_ORDER

if TYPE_CHECKING:
    from collections.abc import Iterable


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_WORKSPACE_ROOTS = (
    ("App runs", Path(tempfile.gettempdir()) / "slavv_runs"),
    ("Repository runs", REPOSITORY_ROOT / "workspace" / "runs"),
)


@dataclass(frozen=True)
class WorkspaceStage:
    """Inspectable progress metadata for one pipeline stage."""

    name: str
    status: str
    progress: float
    elapsed_seconds: float
    detail: str


@dataclass(frozen=True)
class WorkspaceRecord:
    """Lightweight metadata for one structured pipeline run."""

    name: str
    run_dir: str
    source: str
    run_id: str
    status: str
    target_stage: str
    current_stage: str
    progress: float
    updated_at: str
    image_shape: tuple[int, ...] | None
    ready_stages: tuple[str, ...]
    stages: tuple[WorkspaceStage, ...]
    parameters_available: bool
    loadable: bool
    error_count: int
    is_active: bool = False


def _record_for_run(
    run_dir: Path, source: str, active_run_dir: Path | None
) -> WorkspaceRecord | None:
    """Build one record without loading stage checkpoint payloads."""
    snapshot = load_run_snapshot(run_dir)
    if snapshot is None:
        return None
    layout = resolve_run_layout(run_dir=run_dir)
    ready_stages = tuple(stage for stage in STAGE_ORDER if layout.checkpoint_path(stage).is_file())
    parameters_available = (run_dir / "99_Metadata" / "validated_params.json").is_file()
    raw_shape = snapshot.provenance.get("image_shape")
    image_shape = (
        tuple(int(value) for value in raw_shape)
        if isinstance(raw_shape, (list, tuple)) and raw_shape
        else None
    )
    resolved = run_dir.resolve()
    display_name = f"App run · {snapshot.run_id}" if source == "App runs" else run_dir.name
    stages = tuple(
        WorkspaceStage(
            name=stage,
            status=(snapshot.stages[stage].status if stage in snapshot.stages else "not available"),
            progress=(snapshot.stages[stage].progress if stage in snapshot.stages else 0.0),
            elapsed_seconds=(
                snapshot.stages[stage].elapsed_seconds if stage in snapshot.stages else 0.0
            ),
            detail=(snapshot.stages[stage].detail if stage in snapshot.stages else ""),
        )
        for stage in STAGE_ORDER
    )
    return WorkspaceRecord(
        name=display_name,
        run_dir=str(resolved),
        source=source,
        run_id=snapshot.run_id,
        status=snapshot.status,
        target_stage=snapshot.target_stage,
        current_stage=snapshot.current_stage,
        progress=max(0.0, min(float(snapshot.overall_progress), 1.0)),
        updated_at=snapshot.updated_at,
        image_shape=image_shape,
        ready_stages=ready_stages,
        stages=stages,
        parameters_available=parameters_available,
        loadable=parameters_available and bool(ready_stages),
        error_count=len(snapshot.errors),
        is_active=active_run_dir is not None and resolved == active_run_dir,
    )


def discover_workspaces(
    roots: Iterable[tuple[str, str | Path]] = DEFAULT_WORKSPACE_ROOTS,
    *,
    active_run_dir: str | Path | None = None,
    limit: int = 200,
) -> tuple[WorkspaceRecord, ...]:
    """Discover structured runs beneath known roots, newest first."""
    active_path: Path | None = None
    if active_run_dir:
        try:
            active_path = Path(active_run_dir).expanduser().resolve(strict=True)
        except (OSError, RuntimeError):
            active_path = None

    records: dict[str, WorkspaceRecord] = {}
    for source, raw_root in roots:
        root = Path(raw_root).expanduser()
        if not root.is_dir():
            continue
        for snapshot_path in root.rglob("run_snapshot.json"):
            if snapshot_path.parent.name != "99_Metadata":
                continue
            run_dir = snapshot_path.parent.parent
            try:
                record = _record_for_run(run_dir, source, active_path)
            except (OSError, RuntimeError, ValueError):
                continue
            if record is not None:
                records[record.run_dir] = record

    if active_path is not None and str(active_path) not in records:
        try:
            record = _record_for_run(active_path, "Current session", active_path)
        except (OSError, RuntimeError, ValueError):
            record = None
        if record is not None:
            records[record.run_dir] = record

    ordered = sorted(
        records.values(),
        key=lambda record: (record.is_active, record.updated_at, record.name),
        reverse=True,
    )
    return tuple(ordered[: max(0, limit)])


__all__ = [
    "DEFAULT_WORKSPACE_ROOTS",
    "WorkspaceRecord",
    "WorkspaceStage",
    "discover_workspaces",
]
