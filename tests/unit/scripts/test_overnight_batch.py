from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
import threading
import time

import pytest
import tifffile
from scripts.parity import overnight_batch as batch


def _catalog(monkeypatch, tmp_path, *, mutate=None):
    db_root = tmp_path / "db"
    catalog_dir = db_root / "data/metadata"
    scan_dir = db_root / "data/raw/scans"
    catalog_dir.mkdir(parents=True)
    scan_dir.mkdir(parents=True)
    rows = []
    for name in batch.RUNS:
        path = scan_dir / f"{name}.tif"
        tifffile.imwrite(
            path,
            __import__("numpy").zeros((3, 2, 4), dtype="uint16"),
            photometric="minisblack",
        )
        payload = path.read_bytes()
        row = {
            "filename": name,
            "raw_relpath": f"data/raw/scans/{name}.tif",
            "raw_sha256": hashlib.sha256(payload).hexdigest(),
            "raw_bytes": len(payload),
            "resolution": "4 x 2 x 3",
            "qa_status": "ok",
            "record_disposition": "active_canonical",
            "is_active_record": 1,
            "has_raw": 1,
            "vector_relpath": f"data/derived/vectorization/{name}_vector.zip",
            "vector_sha256": "a" * 64,
            "vector_bytes": 123,
            "has_vector": 1,
        }
        if mutate:
            mutate(row)
        rows.append(tuple(row.values()))
    connection = sqlite3.connect(catalog_dir / "neurovasc_catalog.sqlite")
    connection.execute(
        "create table samples (filename text, raw_relpath text, raw_sha256 text, raw_bytes integer, resolution text, qa_status text, record_disposition text, is_active_record integer, has_raw integer, vector_relpath text, vector_sha256 text, vector_bytes integer, has_vector integer)"
    )
    connection.executemany("insert into samples values (?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    connection.commit()
    connection.close()
    monkeypatch.setattr(batch, "DB_ROOT", db_root)
    monkeypatch.setattr(batch, "CATALOG", catalog_dir / "neurovasc_catalog.sqlite")
    return db_root


def test_validate_catalog_accepts_hydrated_database(monkeypatch, tmp_path):
    _catalog(monkeypatch, tmp_path)
    result = batch.validate_catalog()
    assert set(result) == set(batch.RUNS)
    assert result["180709_E"]["dimensions_xyz"] == [4, 2, 3]


def test_validate_catalog_rejects_hash_mismatch(monkeypatch, tmp_path):
    db_root = tmp_path / "db"
    catalog_dir = db_root / "data/metadata"
    scan_dir = db_root / "data/raw/scans"
    catalog_dir.mkdir(parents=True)
    scan_dir.mkdir(parents=True)
    connection = sqlite3.connect(catalog_dir / "neurovasc_catalog.sqlite")
    connection.execute(
        "create table samples (filename text, raw_relpath text, raw_sha256 text, raw_bytes integer, resolution text, qa_status text, record_disposition text, is_active_record integer, has_raw integer, vector_relpath text, vector_sha256 text, vector_bytes integer, has_vector integer)"
    )
    for name in batch.RUNS:
        (scan_dir / f"{name}.tif").write_bytes(b"not-a-tiff")
        connection.execute(
            "insert into samples values (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                name,
                f"data/raw/scans/{name}.tif",
                "0" * 64,
                10,
                "1 x 1 x 1",
                "ok",
                "active_canonical",
                1,
                1,
                None,
                None,
                None,
                0,
            ),
        )
    connection.commit()
    connection.close()
    monkeypatch.setattr(batch, "DB_ROOT", db_root)
    monkeypatch.setattr(batch, "CATALOG", catalog_dir / "neurovasc_catalog.sqlite")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        batch.validate_catalog()


def test_pause_command_writes_sentinel(monkeypatch, tmp_path):
    monkeypatch.setattr(batch, "BATCH_ROOT", tmp_path)
    monkeypatch.setattr(batch, "PAUSE", tmp_path / "PAUSE")
    assert batch.main(["pause"]) == 0
    assert (tmp_path / "PAUSE").is_file()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("qa_status", "needs_review", "QA status"),
        ("record_disposition", "duplicate", "disposition"),
        ("raw_bytes", 1, "byte count"),
        ("resolution", "9 x 9 x 9", "dimensions"),
    ],
)
def test_validate_catalog_enforces_catalog_quality_fields(
    monkeypatch, tmp_path, field, value, message
):
    _catalog(monkeypatch, tmp_path, mutate=lambda row: row.__setitem__(field, value))
    with pytest.raises(RuntimeError, match=message):
        batch.validate_catalog()


def _patch_batch_paths(monkeypatch, tmp_path):
    root = tmp_path / "batch"
    monkeypatch.setattr(batch, "BATCH_ROOT", root)
    monkeypatch.setattr(batch, "MANIFEST", root / "batch_manifest.json")
    monkeypatch.setattr(batch, "PAUSE", root / "PAUSE")
    monkeypatch.setattr(batch, "ORACLE_ROOT", tmp_path / "oracle")
    monkeypatch.setattr(batch, "DATASET_ROOT", tmp_path / "dataset")
    return root


def test_exact_command_sequence_requires_proof(monkeypatch, tmp_path):
    root = _patch_batch_paths(monkeypatch, tmp_path)
    commands = batch.command_sequence_for(
        "180709_E",
        {"path": "/input/180709_E.tif"},
        dataset_root=tmp_path / "dataset",
    )

    assert len(commands) == 2
    assert "init-exact-run" in commands[0]
    assert "prove-exact-sequence" in commands[1]
    assert str(root / "runs/180709_E_exact") in commands[1]


def test_start_labels_exact_only_after_proof_succeeds(monkeypatch, tmp_path):
    _patch_batch_paths(monkeypatch, tmp_path)
    _catalog(monkeypatch, tmp_path)
    monkeypatch.setattr(batch, "require_exact_oracle_compatible", lambda _record: None)
    launched: list[list[str]] = []

    class FinishedProcess:
        pid = 991_337

        def __init__(self, command, **_kwargs):
            launched.append(command)

        @staticmethod
        def poll():
            return 0

    monkeypatch.setattr(batch.subprocess, "Popen", FinishedProcess)

    assert batch.start() == 0
    payload = batch.load_manifest()
    exact = payload["runs"]["180709_E"]
    assert exact["status"] == "completed"
    assert exact["mode"] == "exact"
    assert exact["proof_status"] == "passed"
    assert exact["epistemic_label"] == "MATLAB exact parity"
    assert any("prove-exact-sequence" in command for command in launched)
    assert payload["runs"]["180709_EL"]["epistemic_label"] == ("baseline only - not MATLAB parity")
    report = json.loads((batch.BATCH_ROOT / "final_report.json").read_text())
    assert report["batch_status"] == "completed"


def test_failed_proof_never_receives_exact_label(monkeypatch, tmp_path):
    _patch_batch_paths(monkeypatch, tmp_path)
    _catalog(monkeypatch, tmp_path)
    monkeypatch.setattr(batch, "require_exact_oracle_compatible", lambda _record: None)
    returncodes = iter((0, 1))

    class SequencedProcess:
        pid = 991_338

        def __init__(self, _command, **_kwargs):
            self.returncode = next(returncodes)

        def poll(self):
            return self.returncode

    monkeypatch.setattr(batch.subprocess, "Popen", SequencedProcess)

    assert batch.start() == 1
    exact = batch.load_manifest()["runs"]["180709_E"]
    assert exact["status"] == "failed"
    assert exact["mode"] == "exact_pending_proof"
    assert exact.get("proof_status") is None
    assert exact.get("epistemic_label") is None


def test_status_reports_stage_checkpoint_and_pid_liveness(monkeypatch, tmp_path, capsys):
    root = _patch_batch_paths(monkeypatch, tmp_path)
    run = root / "runs/180709_E_exact"
    (run / "99_Metadata").mkdir(parents=True)
    (run / "02_Checkpoints").mkdir()
    (run / "99_Metadata/run_snapshot.json").write_text(
        json.dumps({"status": "running", "current_stage": "edges"})
    )
    checkpoint = run / "02_Checkpoints/checkpoint_edges.pkl"
    checkpoint.write_bytes(b"checkpoint")
    batch.save_manifest(
        {
            "status": "running",
            "controller_pid": None,
            "runs": {
                "180709_E": {
                    "run_root": str(run),
                    "pid": None,
                    "log": str(run / "batch.log"),
                }
            },
        }
    )

    assert batch.status() == 0
    observed = json.loads(capsys.readouterr().out)
    state = observed["runs"]["180709_E"]["observation"]
    assert state["current_stage"] == "edges"
    assert state["run_status"] == "running"
    assert state["last_checkpoint"] == str(checkpoint)
    assert state["checkpoint_count"] == 1
    assert state["pid_alive"] is False


def test_active_writer_is_detected_before_manifest_replacement(monkeypatch):
    monkeypatch.setattr(batch, "_pid_alive", lambda pid: pid == 123)
    reason = batch._active_writer_reason(
        {"controller_pid": 123, "runs": {"180709_E": {"pid": None}}}
    )
    assert reason == "batch controller PID 123 is still alive"


def test_pause_preserves_checkpoint_and_resume_continues(monkeypatch, tmp_path):
    root = _patch_batch_paths(monkeypatch, tmp_path)
    _catalog(monkeypatch, tmp_path)
    monkeypatch.setattr(batch, "require_exact_oracle_compatible", lambda _record: None)

    def commands(name, _record, *, dataset_root=None):
        del dataset_root
        if name != "180709_E":
            return [[sys.executable, "-c", "pass"]]
        checkpoint = root / "runs" / name / "checkpoint.txt"
        script = (
            "import pathlib,sys,time; "
            f"p=pathlib.Path({str(checkpoint)!r}); p.parent.mkdir(parents=True,exist_ok=True); "
            "existed=p.exists(); p.write_text('durable'); "
            "sys.exit(0) if existed else time.sleep(30)"
        )
        first = [sys.executable, "-c", script]
        return [first, [sys.executable, "-c", "pass"]]

    monkeypatch.setattr(batch, "command_sequence_for", commands)
    errors: list[BaseException] = []

    def run_start():
        try:
            batch.start()
        except BaseException as exc:  # pragma: no cover - surfaced by assertion below
            errors.append(exc)

    thread = threading.Thread(target=run_start)
    thread.start()
    checkpoint = root / "runs/180709_E/checkpoint.txt"
    deadline = time.monotonic() + 10
    while not checkpoint.is_file() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert checkpoint.read_text() == "durable"
    batch.PAUSE.write_text("pause")
    thread.join(timeout=10)

    assert not thread.is_alive()
    assert not errors
    assert batch.load_manifest()["status"] == "paused"
    batch.PAUSE.unlink()
    assert batch.start(resume=True) == 0
    assert checkpoint.read_text() == "durable"
    assert batch.load_manifest()["status"] == "completed"
