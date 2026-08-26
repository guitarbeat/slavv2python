"""Shared workflow readiness and persisted-run loading for the Streamlit app."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from slavv_python.engine.state import load_run_snapshot
from slavv_python.engine.state.io import load_json_dict
from slavv_python.engine.state.layout import resolve_run_layout
from slavv_python.schema.app_run import AppRunState
from slavv_python.schema.results import (
    EdgeSet,
    EnergyResult,
    NetworkResult,
    PipelineResult,
    VertexSet,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from slavv_python.engine.state import RunSnapshot


STAGE_ORDER = ("energy", "vertices", "edges", "network")
_CHECKPOINT_TYPES = {
    "energy": EnergyResult,
    "vertices": VertexSet,
    "edges": EdgeSet,
    "network": NetworkResult,
}


@dataclass(frozen=True)
class WorkflowSummary:
    """Compact session state used by navigation, dashboard, and empty states."""

    dataset_name: str
    source_kind: str
    read_only: bool
    run_dir: str | None
    ready_stages: tuple[str, ...]
    next_page: str
    next_label: str
    curation_mode: str | None


@dataclass(frozen=True)
class RunLoadResult:
    """Validated result of opening an existing structured run directory."""

    app_run: AppRunState | None
    snapshot: RunSnapshot | None
    loaded_stages: tuple[str, ...]
    warnings: tuple[str, ...] = ()
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.app_run is not None and self.error is None


def summarize_workflow(session_state: Mapping[str, Any]) -> WorkflowSummary:
    """Summarize stage readiness and the next useful page for the current session."""
    raw = session_state.get("processing_results")
    if raw is None:
        return WorkflowSummary(
            dataset_name="No dataset loaded",
            source_kind="empty",
            read_only=False,
            run_dir=None,
            ready_stages=(),
            next_page="processing",
            next_label="Process a TIFF",
            curation_mode=None,
        )

    app_run = AppRunState.from_value(raw)
    pipeline = app_run.pipeline
    ready = tuple(
        stage
        for stage, value in (
            ("energy", pipeline.energy_data),
            ("vertices", pipeline.vertices),
            ("edges", pipeline.edges),
            ("network", pipeline.network),
        )
        if value is not None
    )
    if "edges" not in ready:
        next_page, next_label = "processing", "Complete through Edges"
    elif session_state.get("last_curation_mode") is None:
        next_page, next_label = "curation", "Review the network"
    elif "network" not in ready:
        next_page, next_label = "processing", "Build the Network"
    else:
        next_page, next_label = "visualization", "Inspect curated results"

    return WorkflowSummary(
        dataset_name=str(
            session_state.get("dataset_name") or app_run.dataset_name or "Unnamed dataset"
        ),
        source_kind=app_run.source_kind,
        read_only=bool(session_state.get("run_read_only", app_run.read_only)),
        run_dir=str(session_state.get("current_run_dir") or app_run.run_dir or "") or None,
        ready_stages=ready,
        next_page=next_page,
        next_label=next_label,
        curation_mode=session_state.get("last_curation_mode"),
    )


def load_persisted_run(run_dir: str | Path) -> RunLoadResult:
    """Load compatible stage checkpoints from a structured run without modifying it."""
    root = Path(run_dir).expanduser()
    try:
        root = root.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        return RunLoadResult(None, None, (), error=f"Run directory is unavailable: {exc}")
    if not root.is_dir():
        return RunLoadResult(None, None, (), error="The selected run path is not a directory.")

    snapshot = load_run_snapshot(root)
    if snapshot is None:
        return RunLoadResult(
            None,
            None,
            (),
            error="No compatible 99_Metadata/run_snapshot.json was found.",
        )
    params = load_json_dict(root / "99_Metadata" / "validated_params.json")
    if params is None:
        return RunLoadResult(
            None,
            snapshot,
            (),
            error="No compatible 99_Metadata/validated_params.json was found.",
        )

    layout = resolve_run_layout(run_dir=root)
    loaded: dict[str, Any] = {}
    warnings: list[str] = []
    for stage in STAGE_ORDER:
        path = layout.checkpoint_path(stage)
        if not path.is_file():
            warnings.append(f"{stage.title()} checkpoint is not available.")
            continue
        try:
            loaded[stage] = _CHECKPOINT_TYPES[stage].load(path)
        except Exception as exc:
            return RunLoadResult(
                None,
                snapshot,
                tuple(loaded),
                tuple(warnings),
                error=f"Could not load {path.name}: {exc}",
            )

    if not loaded:
        return RunLoadResult(
            None,
            snapshot,
            (),
            tuple(warnings),
            error="The run contains no loadable pipeline checkpoints.",
        )

    energy = loaded.get("energy")
    image_shape = None
    if energy is not None:
        image_shape = tuple(int(value) for value in energy.energy.shape)
    app_run = AppRunState(
        pipeline=PipelineResult(
            parameters=params,
            energy_data=energy,
            vertices=loaded.get("vertices"),
            edges=loaded.get("edges"),
            network=loaded.get("network"),
        ),
        image_shape=image_shape,
        dataset_name=root.name,
        run_dir=str(root),
        source_kind="reopened",
        read_only=True,
    )
    return RunLoadResult(app_run, snapshot, tuple(loaded), tuple(warnings))


def install_loaded_run(session_state: dict[str, Any], result: RunLoadResult) -> None:
    """Install a successful persisted-run load into shared application state."""
    if not result.ok or result.app_run is None:
        raise ValueError(result.error or "The run did not load successfully.")
    app_run = result.app_run
    session_state["processing_results"] = app_run
    session_state["parameters"] = dict(app_run.pipeline.parameters)
    session_state["image_shape"] = app_run.image_shape or (100, 100, 50)
    session_state["dataset_name"] = app_run.dataset_name or "Reopened run"
    session_state["current_run_dir"] = app_run.run_dir
    session_state["run_snapshot"] = result.snapshot.to_dict() if result.snapshot else None
    session_state["run_read_only"] = True
    for key in (
        "curation_baseline_counts",
        "last_curation_mode",
        "analysis_stats",
        "share_report_prepared_signature",
        "curation_source_volume",
        "matlab_curator_session",
        "matlab_curator_payload_cache",
    ):
        session_state.pop(key, None)
    session_state["matlab_curator_generation"] = (
        int(session_state.get("matlab_curator_generation", 0)) + 1
    )


__all__ = [
    "STAGE_ORDER",
    "RunLoadResult",
    "WorkflowSummary",
    "install_loaded_run",
    "load_persisted_run",
    "summarize_workflow",
]
