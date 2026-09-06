from __future__ import annotations

import hashlib
import json
import plistlib
import subprocess
import sys
import time
from pathlib import Path

import pytest
from scripts.parity import macos_matlab_job as job


def _config(tmp_path: Path, command: list[str]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "job_id": "test-job",
        "label": "org.utfoil.slavv2python.matlab.test-job",
        "command": command,
        "cwd": str(tmp_path),
        "state_path": str(tmp_path / "state.json"),
        "log_path": str(tmp_path / "job.log"),
        "pause_path": str(tmp_path / "PAUSE"),
        "heartbeat_seconds": 0.05,
        "source_tiff": None,
        "source_sha256": None,
        "progress_root": str(tmp_path),
        "progress_glob": "chunks_*",
    }


def test_install_materializes_local_supervisor_and_launchagent(monkeypatch, tmp_path):
    support = tmp_path / "Application Support"
    agents = tmp_path / "LaunchAgents"
    state_root = tmp_path / "experiment"
    driver = tmp_path / "driver.m"
    matlab = tmp_path / "MATLAB.app/bin/matlab"
    driver.write_text("disp('probe')\n", encoding="utf-8")
    matlab.parent.mkdir(parents=True)
    matlab.write_text("binary", encoding="utf-8")
    matlab.chmod(0o755)

    monkeypatch.setattr(job, "SUPPORT_ROOT", support)
    monkeypatch.setattr(job, "LAUNCH_AGENTS_ROOT", agents)
    monkeypatch.setattr(job, "SYSTEM_PYTHON", Path(sys.executable))

    result = job.install_job(
        job_id="test-job",
        matlab=matlab,
        driver=driver,
        source_tiff=None,
        source_sha256=None,
        state_root=state_root,
        batch_root=tmp_path / "batch",
        bootstrap=False,
    )

    plist_path = Path(result["plist_path"])
    config_path = Path(result["config_path"])
    runner_path = Path(result["runner_path"])
    assert plist_path.parent == agents
    assert config_path.parent == support / "test-job"
    assert runner_path.parent == support / "test-job"
    assert plist_path.is_file()
    assert config_path.is_file()
    assert runner_path.is_file()
    plist = plistlib.loads(plist_path.read_bytes())
    assert plist["ProgramArguments"] == [
        str(Path(sys.executable)),
        str(runner_path),
        "_supervise",
        str(config_path),
    ]
    assert plist["RunAtLoad"] is False
    assert plist["ExitTimeOut"] == 120
    assert "KeepAlive" not in plist
    assert Path(plist["StandardOutPath"]).parent == support / "test-job"

    config = json.loads(config_path.read_text(encoding="utf-8"))
    runtime = support / "test-job" / "runtime"
    assert Path(config["state_path"]).parent == runtime
    assert Path(config["supervisor_log_path"]).parent == runtime
    assert Path(config["matlab_log_path"]).parent == runtime
    assert Path(config["matlab_pid_path"]).parent == runtime
    assert Path(config["matlab_result_path"]).parent == runtime
    assert Path(config["wrapper_path"]).parent == support / "test-job"
    assert Path(config["cwd"]) == support / "test-job"
    assert config["requested_state_root"] == str(state_root.resolve())
    assert config["command"][:7] == [
        "/usr/bin/caffeinate",
        "-dims",
        "/usr/bin/open",
        "-W",
        "-n",
        "-a",
        str((tmp_path / "MATLAB.app").resolve()),
    ]
    assert config["command"][7:9] == ["--args", "-logfile"]
    assert config["command"][-2] == "-batch"
    assert str(Path(config["wrapper_path"]).parent) in config["command"][-1]
    assert Path(config["wrapper_path"]).stem in config["command"][-1]
    wrapper = Path(config["wrapper_path"]).read_text(encoding="utf-8")
    assert str(driver.resolve()).replace("'", "''") in wrapper
    assert "feature('getpid')" in wrapper
    assert "matlab_result.json" in wrapper
    assert config["driver_sha256"] == hashlib.sha256(driver.read_bytes()).hexdigest()
    assert config["wrapper_sha256"] == hashlib.sha256(wrapper.encode()).hexdigest()


def test_install_verifies_source_hash_before_bootstrap(monkeypatch, tmp_path):
    support = tmp_path / "Application Support"
    agents = tmp_path / "LaunchAgents"
    driver = tmp_path / "driver.m"
    matlab = tmp_path / "MATLAB.app/bin/matlab"
    source = tmp_path / "source.tif"
    driver.write_text("disp('probe')\n", encoding="utf-8")
    matlab.parent.mkdir(parents=True)
    matlab.write_text("binary", encoding="utf-8")
    source.write_bytes(b"unchanged")

    monkeypatch.setattr(job, "SUPPORT_ROOT", support)
    monkeypatch.setattr(job, "LAUNCH_AGENTS_ROOT", agents)
    monkeypatch.setattr(job, "SYSTEM_PYTHON", Path(sys.executable))

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        job.install_job(
            job_id="test-job",
            matlab=matlab,
            driver=driver,
            source_tiff=source,
            source_sha256="0" * 64,
            state_root=tmp_path / "external-state",
            batch_root=tmp_path / "batch",
            bootstrap=False,
        )

    assert not (support / "test-job" / "config.json").exists()


def test_supervisor_finalizes_success_without_external_progress_scan(tmp_path):
    artifact_dir = tmp_path / "chunks_001"
    command = [
        sys.executable,
        "-c",
        (
            "from pathlib import Path; "
            f"p=Path({str(artifact_dir)!r}); p.mkdir(); "
            "(p/'chunk.mat').write_text('ok')"
        ),
    ]
    config = _config(tmp_path, command)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    assert job.supervise(config_path) == 0
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "completed"
    assert state["exit_code"] == 0
    assert state["supervisor_pid"] is None
    assert state["worker_pid"] is None
    assert state["progress"]["access"] == "operator-status-only"
    assert state["ended_at"]


def test_pause_terminates_group_and_resume_reuses_same_state_root(tmp_path):
    checkpoint = tmp_path / "checkpoint.txt"
    command = [
        sys.executable,
        "-c",
        (
            "from pathlib import Path; import time; "
            f"Path({str(checkpoint)!r}).write_text('durable'); time.sleep(30)"
        ),
    ]
    config = _config(tmp_path, command)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    supervisor = subprocess.Popen(
        [sys.executable, str(Path(job.__file__)), "_supervise", str(config_path)]
    )

    deadline = time.monotonic() + 10
    while not checkpoint.is_file() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert checkpoint.read_text(encoding="utf-8") == "durable"
    (tmp_path / "PAUSE").write_text("pause\n", encoding="utf-8")
    assert supervisor.wait(timeout=10) == 0
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "paused"
    assert state["signal"] == "SIGTERM"
    assert checkpoint.read_text(encoding="utf-8") == "durable"

    (tmp_path / "PAUSE").unlink()
    config["command"] = [sys.executable, "-c", "pass"]
    config_path.write_text(json.dumps(config), encoding="utf-8")
    assert job.supervise(config_path) == 0
    resumed = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert resumed["status"] == "completed"
    assert resumed["attempt"] == 2
    assert checkpoint.read_text(encoding="utf-8") == "durable"


def test_pause_terminates_tracked_matlab_pid_outside_worker_group(tmp_path):
    pid_path = tmp_path / "matlab.pid"
    wrapper_path = tmp_path / "matlab_driver_wrapper.m"
    for stale in (pid_path, tmp_path / "state.json", tmp_path / "PAUSE"):
        stale.unlink(missing_ok=True)
    wrapper_path.write_text("% identity marker\n", encoding="utf-8")
    launcher = (
        "import subprocess,sys; "
        "from pathlib import Path; "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)',sys.argv[2]],"
        "start_new_session=True); "
        "Path(sys.argv[1]).write_text(str(child.pid)); "
        "raise SystemExit(child.wait())"
    )
    command = [sys.executable, "-c", launcher, str(pid_path), str(wrapper_path)]
    config = _config(tmp_path, command)
    config.update(
        matlab_pid_path=str(pid_path),
        matlab_pid_command_fragment=str(wrapper_path),
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    supervisor = subprocess.Popen(
        [sys.executable, str(Path(job.__file__)), "_supervise", str(config_path)]
    )

    deadline = time.monotonic() + 10
    while not pid_path.is_file() and time.monotonic() < deadline:
        time.sleep(0.05)
    matlab_pid = int(pid_path.read_text(encoding="utf-8"))
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    deadline = time.monotonic() + 10
    while state.get("matlab_pid") != matlab_pid and time.monotonic() < deadline:
        time.sleep(0.05)
        state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert state["matlab_pid"] == matlab_pid

    (tmp_path / "PAUSE").write_text("pause\n", encoding="utf-8")
    assert supervisor.wait(timeout=10) == 0
    deadline = time.monotonic() + 5
    while job._pid_alive(matlab_pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not job._pid_alive(matlab_pid)
    paused = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert paused["status"] == "paused"
    assert paused["last_matlab_pid"] == matlab_pid
    assert paused["matlab_pid"] is None


def test_completion_marker_is_required_when_configured(tmp_path):
    result_path = tmp_path / "matlab_result.json"
    nonce_path = tmp_path / "attempt_nonce.txt"
    config = _config(tmp_path, [sys.executable, "-c", "pass"])
    config["matlab_result_path"] = str(result_path)
    config["attempt_nonce_path"] = str(nonce_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    assert job.supervise(config_path) == 1
    missing = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert missing["status"] == "failed"
    assert "completion marker" in missing["failure"]

    writer = (
        "from pathlib import Path; import json,sys; "
        "nonce=Path(sys.argv[2]).read_text().strip(); "
        "Path(sys.argv[1]).write_text(json.dumps({'status':'completed','attempt_nonce':nonce}))"
    )
    config["command"] = [
        sys.executable,
        "-c",
        writer,
        str(result_path),
        str(nonce_path),
    ]
    config_path.write_text(json.dumps(config), encoding="utf-8")
    assert job.supervise(config_path) == 0
    completed = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert completed["status"] == "completed"


@pytest.mark.parametrize(
    ("marker", "failure"),
    [
        ({"status": "completed", "attempt_nonce": "wrong"}, "wrong attempt nonce"),
        (
            {"status": "failed", "attempt_nonce": "{nonce}", "message": "driver broke"},
            "driver broke",
        ),
    ],
)
def test_completion_marker_rejects_wrong_nonce_and_matlab_failure(tmp_path, marker, failure):
    result_path = tmp_path / "matlab_result.json"
    nonce_path = tmp_path / "attempt_nonce.txt"
    writer = (
        "from pathlib import Path; import json,sys; "
        "nonce=Path(sys.argv[2]).read_text().strip(); "
        "payload=json.loads(sys.argv[3].replace('{nonce}',nonce)); "
        "Path(sys.argv[1]).write_text(json.dumps(payload))"
    )
    config = _config(tmp_path, [])
    config.update(
        command=[
            sys.executable,
            "-c",
            writer,
            str(result_path),
            str(nonce_path),
            json.dumps(marker),
        ],
        matlab_result_path=str(result_path),
        attempt_nonce_path=str(nonce_path),
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    assert job.supervise(config_path) == 1
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "failed"
    assert failure in state["failure"]


def test_status_reconciles_dead_running_pid_without_mutating_state(tmp_path, capsys):
    state_path = tmp_path / "state.json"
    original = {
        "status": "running",
        "supervisor_pid": 999_999_991,
        "worker_pid": 999_999_992,
    }
    state_path.write_text(json.dumps(original), encoding="utf-8")

    assert job.print_status(state_path) == 1
    observed = json.loads(capsys.readouterr().out)
    assert observed["effective_status"] == "stale-running"
    assert observed["supervisor_pid_alive"] is False
    assert observed["worker_pid_alive"] is False
    assert json.loads(state_path.read_text(encoding="utf-8")) == original


def test_status_observes_external_progress_without_mutating_state(tmp_path, capsys):
    state_path = tmp_path / "runtime" / "state.json"
    state_path.parent.mkdir()
    original = {
        "status": "paused",
        "supervisor_pid": None,
        "worker_pid": None,
        "matlab_pid": None,
    }
    state_path.write_text(json.dumps(original), encoding="utf-8")
    artifact_dir = tmp_path / "external" / "chunks_001"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "chunk.mat").write_text("ok", encoding="utf-8")
    config = _config(tmp_path, [sys.executable, "-c", "pass"])
    config.update(progress_root=str(tmp_path / "external"), progress_glob="chunks_*")

    assert job.print_status(state_path, config=config) == 0
    observed = json.loads(capsys.readouterr().out)
    assert observed["observed_progress"]["artifact_count"] == 1
    assert json.loads(state_path.read_text(encoding="utf-8")) == original
