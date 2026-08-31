from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING

from slavv_python.interface.streamlit.services import exports as export_service
from slavv_python.interface.streamlit.state.curation import (
    apply_curated_session_results,
    summarize_processing_counts,
)
from slavv_python.interface.streamlit.state.manual_curation import curate_manual_selection
from slavv_python.interface.streamlit.state.workflow import (
    install_loaded_run,
    load_persisted_run,
    summarize_workflow,
)
from slavv_python.interface.streamlit.state.workspaces import discover_workspaces
from slavv_python.schema import normalize_pipeline_result
from slavv_python.schema.app_run import AppRunState
from tests.support.payload_builders import build_processing_results
from tests.support.run_state_builders import build_snapshot_dict, materialize_run_snapshot

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _materialize_run(root: Path, stages: tuple[str, ...]) -> Path:
    materialize_run_snapshot(
        root,
        build_snapshot_dict(status="completed_target", target_stage=stages[-1]),
    )
    metadata = root / "99_Metadata"
    (metadata / "validated_params.json").write_text(
        json.dumps({"microns_per_voxel": [1.0, 1.0, 1.0]}),
        encoding="utf-8",
    )
    pipeline = normalize_pipeline_result(build_processing_results())
    checkpoint_dir = root / "02_Output" / "python_results" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    values = {
        "energy": pipeline.energy_data,
        "vertices": pipeline.vertices,
        "edges": pipeline.edges,
        "network": pipeline.network,
    }
    for stage in stages:
        values[stage].save(checkpoint_dir / f"checkpoint_{stage}.pkl")
    return root


def test_workflow_recommends_processing_without_results() -> None:
    summary = summarize_workflow({})
    assert summary.ready_stages == ()
    assert summary.next_page == "processing"


def test_workflow_recommends_curation_then_visualization() -> None:
    run = AppRunState.from_value(build_processing_results())
    summary = summarize_workflow({"processing_results": run, "dataset_name": "sample"})
    assert summary.ready_stages == ("energy", "vertices", "edges", "network")
    assert summary.next_page == "curation"
    curated = summarize_workflow(
        {
            "processing_results": run,
            "dataset_name": "sample",
            "last_curation_mode": "Browser manual review",
        }
    )
    assert curated.next_page == "visualization"


def test_manual_curation_rebuilds_shared_downstream_state() -> None:
    results = build_processing_results()
    session: dict[str, object] = {
        "processing_results": AppRunState.from_value(results),
        "analysis_stats": {"stale": True},
        "share_report_prepared_signature": "stale",
    }
    vertices, edges = curate_manual_selection(
        results["vertices"],
        results["edges"],
        rejected_vertex_ids=(2,),
    )
    apply_curated_session_results(
        session,
        vertices,
        edges,
        curation_mode="Browser manual review",
    )
    counts = summarize_processing_counts(session["processing_results"])
    assert counts["Vertices"] == 2
    assert counts["Edges"] == 1
    assert counts["Strands"] >= 1
    assert "analysis_stats" not in session
    assert "share_report_prepared_signature" not in session


def test_load_complete_persisted_run_is_read_only(tmp_path: Path) -> None:
    root = _materialize_run(tmp_path / "run-a", ("energy", "vertices", "edges", "network"))
    result = load_persisted_run(root)
    assert result.ok
    assert result.loaded_stages == ("energy", "vertices", "edges", "network")
    assert result.app_run is not None
    assert result.app_run.read_only
    session: dict[str, object] = {}
    install_loaded_run(session, result)
    assert session["run_read_only"] is True
    assert session["current_run_dir"] == str(root.resolve())


def test_load_partial_run_reports_missing_stages(tmp_path: Path) -> None:
    root = _materialize_run(tmp_path / "run-b", ("energy", "vertices"))
    result = load_persisted_run(root)
    assert result.ok
    assert result.loaded_stages == ("energy", "vertices")
    assert any("Edges checkpoint" in warning for warning in result.warnings)


def test_load_run_rejects_missing_and_corrupt_surfaces(tmp_path: Path) -> None:
    missing = load_persisted_run(tmp_path / "missing")
    assert not missing.ok
    root = _materialize_run(tmp_path / "corrupt", ("energy",))
    checkpoint = root / "02_Output" / "python_results" / "checkpoints" / "checkpoint_energy.pkl"
    checkpoint.write_bytes(b"not a checkpoint")
    corrupt = load_persisted_run(root)
    assert not corrupt.ok
    assert "checkpoint_energy.pkl" in (corrupt.error or "")


def test_read_only_run_skips_metadata_updates(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_if_called(run_dir: str) -> None:
        raise AssertionError(f"read-only run metadata opened for writing: {run_dir}")

    monkeypatch.setattr(
        export_service,
        "st",
        SimpleNamespace(session_state={"run_read_only": True}),
    )
    monkeypatch.setattr(
        export_service.RunContext,
        "from_existing",
        fail_if_called,
    )
    export_service.update_run_task(
        "persisted-run",
        "analysis",
        status="completed",
        detail="viewed",
    )


def test_workspace_discovery_indexes_runs_and_marks_active(tmp_path: Path) -> None:
    active = _materialize_run(
        tmp_path / "runs" / "active-run",
        ("energy", "vertices", "edges", "network"),
    )
    partial = _materialize_run(
        tmp_path / "runs" / "partial-run",
        ("energy", "vertices"),
    )

    records = discover_workspaces(
        (("Test runs", tmp_path / "runs"),),
        active_run_dir=active,
    )

    assert [record.name for record in records] == ["active-run", "partial-run"]
    assert records[0].is_active
    assert records[0].loadable
    assert records[0].ready_stages == ("energy", "vertices", "edges", "network")
    assert records[1].ready_stages == ("energy", "vertices")
    assert records[1].run_dir == str(partial.resolve())


def test_workspace_discovery_reports_metadata_only_run(tmp_path: Path) -> None:
    root = tmp_path / "runs" / "metadata-only"
    materialize_run_snapshot(root, build_snapshot_dict(status="running"))

    records = discover_workspaces((("Test runs", tmp_path / "runs"),))

    assert len(records) == 1
    assert not records[0].loadable
    assert records[0].ready_stages == ()
    assert not records[0].is_parity_job


def test_workspace_discovery_flags_parity_job_runs(tmp_path: Path) -> None:
    from slavv_python.interface.streamlit.state.workspaces import is_parity_run_dir

    plain = _materialize_run(tmp_path / "runs" / "plain-run", ("energy",))
    parity = _materialize_run(tmp_path / "runs" / "parity-run", ("energy",))
    (parity / "99_Metadata" / "parity_job.json").write_text("{}", encoding="utf-8")

    assert not is_parity_run_dir(plain)
    assert is_parity_run_dir(parity)

    records = discover_workspaces((("Test runs", tmp_path / "runs"),))
    by_name = {record.name: record for record in records}
    assert not by_name["plain-run"].is_parity_job
    assert by_name["parity-run"].is_parity_job


def test_resolve_workspace_refresh_seconds_clamps_choices() -> None:
    from slavv_python.interface.streamlit.views.workspaces import (
        resolve_workspace_refresh_seconds,
    )

    assert resolve_workspace_refresh_seconds(30) == 30
    assert resolve_workspace_refresh_seconds(45) == 45
    assert resolve_workspace_refresh_seconds(60) == 60
    assert resolve_workspace_refresh_seconds(15) == 45
    assert resolve_workspace_refresh_seconds("nope") == 45
