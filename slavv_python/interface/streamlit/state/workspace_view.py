"""Read-only view model shared by Streamlit workspace surfaces.

The view deliberately contains metadata only.  It is assembled from session state
or an indexed ``WorkspaceRecord`` and never writes to the run directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from slavv_python.engine.state import RunSnapshot, load_run_snapshot

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from .workflow import WorkflowSummary
    from .workspaces import WorkspaceRecord


@dataclass(frozen=True)
class WorkspaceRunView:
    """Immutable metadata projection for a current or discovered run."""

    run_dir: str | None
    dataset_name: str
    source_kind: str
    read_only: bool
    snapshot: RunSnapshot | None
    ready_stages: tuple[str, ...]
    target_stage: str
    current_stage: str
    progress: float
    image_shape: tuple[int, ...] | None
    is_active: bool = False
    is_parity_job: bool = False
    parameters_available: bool = False
    loadable: bool = False

    @property
    def has_run(self) -> bool:
        """Whether the projection points at a persisted run."""
        return self.run_dir is not None


def workspace_view_from_session(
    session_state: Mapping[str, Any],
    *,
    summary: WorkflowSummary | None = None,
    snapshot_loader: Callable[[str], RunSnapshot | None] = load_run_snapshot,
) -> WorkspaceRunView:
    """Build the current read-only projection without touching persisted state."""
    run_dir = str(session_state.get("current_run_dir") or "") or None
    snapshot = snapshot_loader(run_dir) if run_dir else None
    ready_stages = summary.ready_stages if summary is not None else ()
    dataset_name = (
        summary.dataset_name
        if summary is not None
        else str(session_state.get("dataset_name", "No dataset loaded"))
    )
    source_kind = summary.source_kind if summary is not None else "empty"
    read_only = (
        summary.read_only
        if summary is not None
        else bool(session_state.get("run_read_only", False))
    )
    return _view(
        run_dir=run_dir,
        dataset_name=dataset_name,
        source_kind=source_kind,
        read_only=read_only,
        snapshot=snapshot,
        ready_stages=ready_stages,
        image_shape=_session_image_shape(session_state),
    )


def workspace_view_from_record(
    record: WorkspaceRecord,
    *,
    snapshot_loader: Callable[[str], RunSnapshot | None] = load_run_snapshot,
) -> WorkspaceRunView:
    """Build the projection used when inspecting an indexed workspace record."""
    snapshot = snapshot_loader(record.run_dir)
    return _view(
        run_dir=record.run_dir,
        dataset_name=record.name,
        source_kind=record.source,
        read_only=True,
        snapshot=snapshot,
        ready_stages=record.ready_stages,
        image_shape=record.image_shape,
        is_active=record.is_active,
        is_parity_job=record.is_parity_job,
        parameters_available=record.parameters_available,
        loadable=record.loadable,
    )


def _session_image_shape(session_state: Mapping[str, Any]) -> tuple[int, ...] | None:
    raw = session_state.get("image_shape")
    if isinstance(raw, (tuple, list)) and raw:
        return tuple(int(value) for value in raw)
    return None


def _view(
    *,
    run_dir: str | None,
    dataset_name: str,
    source_kind: str,
    read_only: bool,
    snapshot: RunSnapshot | None,
    ready_stages: tuple[str, ...],
    image_shape: tuple[int, ...] | None,
    is_active: bool = False,
    is_parity_job: bool = False,
    parameters_available: bool = False,
    loadable: bool = False,
) -> WorkspaceRunView:
    return WorkspaceRunView(
        run_dir=run_dir,
        dataset_name=dataset_name,
        source_kind=source_kind,
        read_only=read_only,
        snapshot=snapshot,
        ready_stages=ready_stages,
        target_stage=snapshot.target_stage if snapshot is not None else "network",
        current_stage=snapshot.current_stage if snapshot is not None else "",
        progress=(
            max(0.0, min(float(snapshot.overall_progress), 1.0))
            if snapshot is not None
            else 0.0
        ),
        image_shape=image_shape,
        is_active=is_active,
        is_parity_job=is_parity_job,
        parameters_available=parameters_available,
        loadable=loadable,
    )


__all__ = ["WorkspaceRunView", "workspace_view_from_record", "workspace_view_from_session"]
