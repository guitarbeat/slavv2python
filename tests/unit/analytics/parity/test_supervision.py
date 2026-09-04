from __future__ import annotations

import pytest

from slavv_python.analytics.parity.runs.parity_job_lifecycle import (
    load_parity_job_manifest,
)
from slavv_python.analytics.parity.runs.supervision import (
    InvalidParityJobTransition,
    ParityRunSupervisor,
)


@pytest.mark.unit
def test_supervisor_start_and_finish_preserve_legacy_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "slavv_python.analytics.parity.runs.parity_job_lifecycle.now_iso",
        lambda: "2026-09-04T01:02:03Z",
    )
    run = ParityRunSupervisor(tmp_path / "run")

    started = run.start(pid=123, command=["python", "resume"], stage="edges")
    assert started["status"] == "running"
    assert started["pid"] == 123
    assert started["stage"] == "edges"

    finished = run.finish(status="succeeded", exit_code=0)
    assert finished["status"] == "succeeded"
    assert finished["ended_at"] == "2026-09-04T01:02:03Z"
    assert load_parity_job_manifest(tmp_path / "run") == finished


@pytest.mark.unit
def test_supervisor_rejects_resurrection_of_completed_run(tmp_path):
    run = ParityRunSupervisor(tmp_path / "run")
    run.start(pid=123, command="resume", stage="edges")
    run.finish(status="succeeded", exit_code=0)

    with pytest.raises(InvalidParityJobTransition, match=r"succeeded.*running"):
        run.start(pid=456, command="resume", stage="network")


@pytest.mark.unit
def test_supervisor_observation_exposes_checkpoint_alias(tmp_path):
    run = ParityRunSupervisor(tmp_path / "run")
    run.start(pid=123, command="resume", stage="edges")
    manifest = load_parity_job_manifest(tmp_path / "run")
    assert manifest is not None
    manifest["last_checkpoint"] = "04_Edges/candidates.pkl"
    from slavv_python.analytics.parity.utils import write_json_with_hash

    write_json_with_hash(run.run_dir / "99_Metadata" / "parity_job.json", manifest)
    observation = run.observe()
    assert observation.status == "running"
    assert observation.checkpoint == "04_Edges/candidates.pkl"
