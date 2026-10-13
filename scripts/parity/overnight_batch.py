#!/usr/bin/env python3
"""Pause-safe sequential parity/baseline batch for hydrated neurovasc-db TIFFs.

Examples (run from the repository root)::

  uv run python scripts/parity/overnight_batch.py start
  uv run python scripts/parity/overnight_batch.py status
  uv run python scripts/parity/overnight_batch.py pause
  uv run python scripts/parity/overnight_batch.py resume

The controller owns only ``workspace/scratch/overnight_parity_batch``. It never
modifies the database or protected parity roots.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import signal
import sqlite3
import subprocess
import sys
import time
from contextlib import contextmanager, suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import tifffile

if TYPE_CHECKING:
    from collections.abc import Iterator

REPO = Path(__file__).resolve().parents[2]
DB_ROOT = Path(os.environ.get("NEUROVASC_DB_ROOT", "/Volumes/LoveSSD/NeuroVASC/repos/neurovasc-db"))
CATALOG = DB_ROOT / "data/metadata/neurovasc_catalog.sqlite"
BATCH_ROOT = REPO / "workspace/scratch/overnight_parity_batch"
MANIFEST = BATCH_ROOT / "batch_manifest.json"
PAUSE = BATCH_ROOT / "PAUSE"
RUNS = ("180709_E", "180709_EL", "170802_01")
ORACLE_ROOT = Path(
    os.environ.get(
        "SLAVV_OVERNIGHT_ORACLE_ROOT",
        REPO / "workspace/oracles/180709_E_full_v2",
    )
)
DATASET_ROOT = Path(
    os.environ.get(
        "SLAVV_OVERNIGHT_DATASET_ROOT",
        REPO
        / "workspace/datasets/771eb62fd1322cf59e24f056aff2692b3375b94ce6dc9b25744428d4dbf1e353",
    )
)
MATLAB_CANDIDATES = (
    Path("/Volumes/LoveSSD/Applications/MATLAB_R2024b.app/bin/matlab"),
    Path("/Applications/MATLAB_R2024b.app/bin/matlab"),
)


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def catalog_rows() -> dict[str, dict[str, Any]]:
    if not CATALOG.is_file():
        raise RuntimeError(f"catalog not found: {CATALOG}")
    connection = sqlite3.connect(CATALOG)
    try:
        rows = connection.execute(
            """select filename, raw_relpath, raw_sha256, raw_bytes, resolution,
                      qa_status, record_disposition, is_active_record, has_raw,
                      vector_relpath, vector_sha256, vector_bytes, has_vector
               from samples where filename in (?, ?, ?)""",
            RUNS,
        ).fetchall()
    finally:
        connection.close()
    return {
        row[0]: {
            "filename": row[0],
            "path": str(DB_ROOT / row[1]),
            "expected_sha256": row[2],
            "expected_bytes": row[3],
            "resolution": row[4],
            "qa_status": row[5],
            "record_disposition": row[6],
            "is_active_record": bool(row[7]),
            "has_raw": bool(row[8]),
            "legacy_vectorization": {
                "role": "metadata_only_not_exact_oracle",
                "path": str(DB_ROOT / row[9]) if row[9] else None,
                "sha256": row[10],
                "bytes": row[11],
                "present": bool(row[12]),
            },
        }
        for row in rows
    }


def validate_catalog() -> dict[str, dict[str, Any]]:
    rows = catalog_rows()
    missing = [name for name in RUNS if name not in rows]
    if missing:
        raise RuntimeError(f"catalog records missing: {missing}")
    for name in RUNS:
        record = rows[name]
        path = Path(record["path"])
        if not (record["is_active_record"] and record["has_raw"]):
            raise RuntimeError(f"{name} is not an active hydrated raw record")
        if record["qa_status"] != "ok":
            raise RuntimeError(f"{name} QA status is {record['qa_status']!r}, expected 'ok'")
        if record["record_disposition"] != "active_canonical":
            raise RuntimeError(
                f"{name} disposition is {record['record_disposition']!r}, "
                "expected 'active_canonical'"
            )
        if not path.is_file():
            raise RuntimeError(f"TIFF missing: {path}")
        actual = sha256(path)
        if record["expected_sha256"] and actual != record["expected_sha256"]:
            raise RuntimeError(
                f"SHA-256 mismatch for {name}: {actual} != {record['expected_sha256']}"
            )
        actual_bytes = path.stat().st_size
        if record["expected_bytes"] is not None and actual_bytes != record["expected_bytes"]:
            raise RuntimeError(
                f"byte count mismatch for {name}: {actual_bytes} != {record['expected_bytes']}"
            )
        with tifffile.TiffFile(path) as tif:
            shape = tuple(int(value) for value in tif.series[0].shape)
            if len(shape) != 3:
                raise RuntimeError(f"{name} is not a 3-D TIFF: {shape}")
            dimensions_xyz = [shape[2], shape[1], shape[0]]
            expected_dimensions = [
                int(value) for value in re.findall(r"\d+", record["resolution"] or "")
            ]
            if len(expected_dimensions) != 3 or dimensions_xyz != expected_dimensions:
                raise RuntimeError(
                    f"dimensions mismatch for {name}: TIFF {dimensions_xyz} != "
                    f"catalog {expected_dimensions or record['resolution']!r}"
                )
        record["sha256"] = actual
        record["bytes"] = actual_bytes
        record["dimensions_xyz"] = dimensions_xyz
    return rows


def ensure_exact_dataset_surface(record: dict[str, Any]) -> Path:
    """Create a scratch manifest that points at the verified DB TIFF in place."""
    root = BATCH_ROOT / "dataset_180709_E"
    metadata = root / "99_Metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_hash": record["sha256"],
        "dataset_root": str(root),
        "input_bytes": record["bytes"],
        "input_filename": "180709_E.tif",
        "kind": "dataset",
        "manifest_version": 1,
        "stored_input_file": record["path"],
    }
    (metadata / "dataset_manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return root


def exact_oracle_ready() -> bool:
    required = (
        "energy.pkl",
        "vertices.pkl",
        "edges.pkl",
        "network.pkl",
    )
    return (ORACLE_ROOT / "99_Metadata/oracle_manifest.json").is_file() and all(
        (ORACLE_ROOT / "03_Analysis/normalized/oracle" / name).is_file() for name in required
    )


def require_exact_oracle_compatible(record: dict[str, Any]) -> None:
    """Require a complete oracle paired to this exact TIFF hash."""
    if not exact_oracle_ready():
        raise RuntimeError("matching normalized MATLAB oracle artifacts are unavailable")
    payload = json.loads(
        (ORACLE_ROOT / "99_Metadata/oracle_manifest.json").read_text(encoding="utf-8")
    )
    if payload.get("dataset_hash") != record["sha256"]:
        raise RuntimeError(
            "database TIFF hash does not match oracle dataset: "
            f"{record['sha256']} != {payload.get('dataset_hash')}"
        )


def run_root(name: str) -> Path:
    suffix = "180709_E_exact" if name == "180709_E" else name
    return BATCH_ROOT / "runs" / suffix


def command_sequence_for(
    name: str,
    record: dict[str, Any],
    *,
    dataset_root: Path | None = None,
) -> list[list[str]]:
    """Return the writer command and, for exact mode, its required proof."""
    destination = run_root(name)
    if name == "180709_E":
        writer = [
            "uv",
            "run",
            "slavv",
            "parity",
            "init-exact-run",
            "--dataset-root",
            str(dataset_root or DATASET_ROOT),
            "--oracle-root",
            str(ORACLE_ROOT),
            "--dest-run-root",
            str(destination),
            "--stop-after",
            "network",
        ]
        if (destination / "99_Metadata/experiment_provenance.json").is_file():
            writer.append("--resume")
        return [
            writer,
            [
                "uv",
                "run",
                "slavv",
                "parity",
                "prove-exact-sequence",
                "--source-run-root",
                str(destination),
                "--dest-run-root",
                str(destination),
                "--oracle-root",
                str(ORACLE_ROOT),
            ],
        ]
    return [
        [
            "uv",
            "run",
            "slavv",
            "run",
            "--input",
            record["path"],
            "--run-dir",
            str(destination),
            "--profile",
            "matlab_compat",
            "--n-jobs",
            "1",
            "--stop-after",
            "network",
        ]
    ]


def load_manifest() -> dict[str, Any]:
    if not MANIFEST.is_file():
        raise RuntimeError(f"batch manifest not found: {MANIFEST}; run start first")
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def save_manifest(payload: dict[str, Any]) -> None:
    BATCH_ROOT.mkdir(parents=True, exist_ok=True)
    temp = MANIFEST.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temp.replace(MANIFEST)


def _pid_alive(value: Any) -> bool:
    try:
        pid = int(value)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _active_writer_reason(manifest: dict[str, Any]) -> str | None:
    """Return a conflict description without replacing persisted state."""
    controller_pid = manifest.get("controller_pid")
    if controller_pid and int(controller_pid) != os.getpid() and _pid_alive(controller_pid):
        return f"batch controller PID {controller_pid} is still alive"
    for name, entry in manifest.get("runs", {}).items():
        if _pid_alive(entry.get("pid")):
            return f"{name} batch process PID {entry['pid']} is still alive"
        lease_path = Path(entry.get("run_root", run_root(name))) / "99_Metadata/writer_lease.json"
        if not lease_path.is_file():
            continue
        try:
            lease = json.loads(lease_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if lease.get("status") in {"running", "launched"} and _pid_alive(lease.get("pid")):
            return f"{name} writer lease PID {lease['pid']} is still alive"
    return None


@contextmanager
def _controller_lock() -> Iterator[None]:
    """Hold a non-blocking process lock for one start/resume controller."""
    BATCH_ROOT.mkdir(parents=True, exist_ok=True)
    with (BATCH_ROOT / "controller.lock").open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another overnight batch controller holds the lock") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()} acquired_utc={utc()}\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _terminate_process_group(process: subprocess.Popen[Any]) -> int:
    """Stop caffeinate and its worker while preserving durable checkpoints."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return process.wait()
    try:
        return process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        return process.wait()


def _run_observation(entry: dict[str, Any]) -> dict[str, Any]:
    root = Path(entry["run_root"])
    snapshot_path = root / "99_Metadata/run_snapshot.json"
    snapshot: dict[str, Any] | None = None
    if snapshot_path.is_file():
        try:
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            snapshot = {"error": f"unreadable snapshot: {snapshot_path}"}
    checkpoints = sorted(root.rglob("checkpoint_*.pkl")) if root.is_dir() else []
    candidates = sorted(root.rglob("candidates.pkl")) if root.is_dir() else []
    durable = [*checkpoints, *candidates]
    newest = max(durable, key=lambda path: path.stat().st_mtime) if durable else None
    return {
        "pid_alive": _pid_alive(entry.get("pid")),
        "current_stage": snapshot.get("current_stage") if snapshot else None,
        "run_status": snapshot.get("status") if snapshot else None,
        "last_checkpoint": str(newest) if newest else None,
        "checkpoint_count": len(durable),
        "log": entry.get("log"),
    }


def _write_final_report(manifest: dict[str, Any]) -> Path:
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": utc(),
        "batch_status": manifest["status"],
        "runs": [],
    }
    for name in RUNS:
        entry = manifest["runs"][name]
        file_record = entry["file"]
        report["runs"].append(
            {
                "name": name,
                "sha256": file_record["sha256"],
                "dimensions_xyz": file_record["dimensions_xyz"],
                "elapsed_seconds": entry.get("elapsed_seconds"),
                "terminal_status": entry.get("status"),
                "failed_command_index": entry.get("failed_command_index"),
                "skipped_reason": entry.get("oracle_block_reason"),
                "epistemic_label": entry.get("epistemic_label"),
                "proof_status": entry.get("proof_status"),
                "legacy_vectorization": file_record.get("legacy_vectorization"),
            }
        )
    path = BATCH_ROOT / "final_report.json"
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)
    return path


def start(*, resume: bool = False) -> int:
    with _controller_lock():
        if MANIFEST.is_file():
            existing = load_manifest()
            conflict = _active_writer_reason(existing)
            if conflict:
                raise RuntimeError(f"refusing duplicate writer: {conflict}")
            if not resume:
                raise RuntimeError(
                    "batch manifest already exists; use resume to preserve its run roots"
                )

        records = validate_catalog()
        exact_dataset_root = ensure_exact_dataset_surface(records["180709_E"])
        exact_error: str | None = None
        try:
            require_exact_oracle_compatible(records["180709_E"])
        except RuntimeError as exc:
            exact_error = str(exc)

        manifest = (
            load_manifest()
            if resume
            else {
                "schema_version": 2,
                "created_utc": utc(),
                "batch_root": str(BATCH_ROOT),
                "runs": {},
                "status": "pending",
            }
        )
        manifest["controller_pid"] = os.getpid()
        for name in RUNS:
            entry = manifest["runs"].setdefault(name, {})
            was_completed = entry.get("status") == "completed"
            preserve_completed = was_completed and (
                name != "180709_E" or entry.get("proof_status") == "passed"
            )
            exact = name == "180709_E" and exact_error is None
            entry.update(
                {
                    "file": records[name],
                    "mode": "exact_pending_proof"
                    if exact
                    else ("blocked_exact" if name == "180709_E" else "baseline_only"),
                    "run_root": str(run_root(name)),
                    "commands": command_sequence_for(
                        name, records[name], dataset_root=exact_dataset_root
                    )
                    if name != "180709_E" or exact
                    else [],
                    "log": str(BATCH_ROOT / "logs" / f"{name}.log"),
                }
            )
            if preserve_completed:
                entry["status"] = "completed"
            elif name == "180709_E" and not exact:
                entry.update(
                    {
                        "status": "blocked",
                        "oracle_block_reason": exact_error,
                        "epistemic_label": "exact parity not run",
                    }
                )
            else:
                entry.pop("oracle_block_reason", None)
                entry.setdefault("status", "pending")
        manifest["status"] = "running"
        manifest["started_utc"] = manifest.get("started_utc", utc())
        save_manifest(manifest)

        for name in RUNS:
            if PAUSE.exists():
                manifest["status"] = "paused"
                manifest["controller_pid"] = None
                save_manifest(manifest)
                return 0
            entry = manifest["runs"][name]
            if entry.get("status") in {"completed", "blocked"}:
                continue

            root = run_root(name)
            root.mkdir(parents=True, exist_ok=True)
            log_path = Path(entry["log"])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            started_monotonic = time.monotonic()
            prior_elapsed = float(entry.get("elapsed_seconds") or 0.0)
            entry.update(
                {
                    "status": "running",
                    "started_utc": entry.get("started_utc", utc()),
                    "pid": None,
                }
            )
            entry.pop("returncode", None)
            entry.pop("finished_utc", None)
            save_manifest(manifest)

            next_index = int(entry.get("next_command_index", 0))
            for command_index, cmd in enumerate(entry["commands"][next_index:], start=next_index):
                entry.update({"current_command_index": command_index, "current_command": cmd})
                wrapped = ["caffeinate", "-dims", *cmd] if sys.platform == "darwin" else cmd
                with log_path.open("a", encoding="utf-8") as log:
                    log.write(f"\n[{utc()}] START {' '.join(cmd)}\n")
                    process = subprocess.Popen(
                        wrapped,
                        cwd=REPO,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    entry["pid"] = process.pid
                    save_manifest(manifest)
                    while True:
                        returncode = process.poll()
                        if returncode is not None:
                            break
                        if PAUSE.exists():
                            returncode = _terminate_process_group(process)
                            entry.update(
                                {
                                    "status": "paused",
                                    "returncode": returncode,
                                    "pid": None,
                                    "elapsed_seconds": prior_elapsed
                                    + time.monotonic()
                                    - started_monotonic,
                                }
                            )
                            log.write(f"[{utc()}] PAUSED exit={returncode}\n")
                            manifest["status"] = "paused"
                            manifest["controller_pid"] = None
                            save_manifest(manifest)
                            return 0
                        time.sleep(2)
                    log.write(f"[{utc()}] EXIT {returncode}\n")
                entry["pid"] = None
                entry["returncode"] = returncode
                if returncode != 0:
                    entry.update(
                        {
                            "status": "failed",
                            "failed_command_index": command_index,
                            "finished_utc": utc(),
                            "elapsed_seconds": prior_elapsed + time.monotonic() - started_monotonic,
                        }
                    )
                    manifest["status"] = "failed"
                    manifest["controller_pid"] = None
                    save_manifest(manifest)
                    _write_final_report(manifest)
                    return returncode
                entry["next_command_index"] = command_index + 1
                if name == "180709_E" and command_index == 1:
                    entry["proof_status"] = "passed"
                    entry["mode"] = "exact"
                    entry["epistemic_label"] = "MATLAB exact parity"
                save_manifest(manifest)

            entry.update(
                {
                    "finished_utc": utc(),
                    "elapsed_seconds": prior_elapsed + time.monotonic() - started_monotonic,
                    "status": "completed",
                    "epistemic_label": entry.get("epistemic_label")
                    or "baseline only - not MATLAB parity",
                }
            )
            save_manifest(manifest)

        blocked = any(entry.get("status") == "blocked" for entry in manifest["runs"].values())
        manifest["status"] = "incomplete" if blocked else "completed"
        manifest["finished_utc"] = utc()
        manifest["controller_pid"] = None
        save_manifest(manifest)
        _write_final_report(manifest)
        return 2 if blocked else 0


def status() -> int:
    manifest = load_manifest()
    observed = dict(manifest)
    observed["controller_pid_alive"] = _pid_alive(manifest.get("controller_pid"))
    observed["runs"] = {
        name: {**entry, "observation": _run_observation(entry)}
        for name, entry in manifest.get("runs", {}).items()
    }
    print(json.dumps(observed, indent=2))
    return 0


def preflight() -> int:
    records = validate_catalog()
    if not ORACLE_ROOT.is_dir() or not DATASET_ROOT.is_dir():
        raise RuntimeError("180709_E workspace dataset/oracle roots are not available")
    exact_error: str | None = None
    try:
        require_exact_oracle_compatible(records["180709_E"])
    except RuntimeError as exc:
        exact_error = str(exc)
    import shutil

    matlab_path = shutil.which("matlab") or next(
        (str(path) for path in MATLAB_CANDIDATES if path.is_file()), None
    )
    payload = {
        "passed": exact_error is None,
        "checked_utc": utc(),
        "database_root": str(DB_ROOT),
        "catalog": str(CATALOG),
        "runs": records,
        "exact_dataset_root": str(DATASET_ROOT),
        "exact_oracle_root": str(ORACLE_ROOT),
        "matlab_available": matlab_path is not None,
        "matlab_path": matlab_path,
        "oracle_ready": exact_error is None,
        "oracle_error": exact_error,
        "n_jobs": 1,
        "protected_roots_untouched": True,
    }
    BATCH_ROOT.mkdir(parents=True, exist_ok=True)
    (BATCH_ROOT / "preflight.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    return 0 if payload["passed"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "start", "resume", "pause", "status"))
    args = parser.parse_args(argv)
    if args.action == "pause":
        BATCH_ROOT.mkdir(parents=True, exist_ok=True)
        PAUSE.write_text(f"requested_utc={utc()}\n", encoding="utf-8")
        print(f"pause requested: {PAUSE}")
        return 0
    if args.action == "status":
        return status()
    if args.action == "preflight":
        return preflight()
    if args.action == "resume":
        PAUSE.unlink(missing_ok=True)
        return start(resume=True)
    return start()


if __name__ == "__main__":
    raise SystemExit(main())
