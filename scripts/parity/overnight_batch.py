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
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tifffile

REPO = Path(__file__).resolve().parents[2]
DB_ROOT = Path(os.environ.get("NEUROVASC_DB_ROOT", "/Volumes/LoveSSD/neurovasc-db"))
CATALOG = DB_ROOT / "data/metadata/neurovasc_catalog.sqlite"
BATCH_ROOT = REPO / "workspace/scratch/overnight_parity_batch"
MANIFEST = BATCH_ROOT / "batch_manifest.json"
PAUSE = BATCH_ROOT / "PAUSE"
RUNS = ("180709_E", "180709_EL", "170802_01")
ORACLE_ROOT = REPO / "workspace/oracles/180709_E_full_v2"
DATASET_ROOT = REPO / "workspace/datasets/771eb62fd1322cf59e24f056aff2692b3375b94ce6dc9b25744428d4dbf1e353"


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
                      qa_status, record_disposition, is_active_record, has_raw
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
        if not path.is_file():
            raise RuntimeError(f"TIFF missing: {path}")
        actual = sha256(path)
        if record["expected_sha256"] and actual != record["expected_sha256"]:
            raise RuntimeError(f"SHA-256 mismatch for {name}: {actual} != {record['expected_sha256']}")
        with tifffile.TiffFile(path) as tif:
            if len(tif.series[0].shape) != 3:
                raise RuntimeError(f"{name} is not a 3-D TIFF: {tif.series[0].shape}")
        record["sha256"] = actual
        record["bytes"] = path.stat().st_size
    return rows


def run_root(name: str) -> Path:
    return BATCH_ROOT / "runs" / name


def command_for(name: str, record: dict[str, Any]) -> list[str]:
    destination = run_root(name)
    if name == "180709_E":
        return [
            "uv", "run", "slavv", "parity", "init-exact-run",
            "--dataset-root", str(DATASET_ROOT), "--oracle-root", str(ORACLE_ROOT),
            "--dest-run-root", str(destination), "--stop-after", "network", "--resume",
        ]
    return [
        "uv", "run", "slavv", "run", "--input", record["path"], "--run-dir", str(destination),
        "--profile", "matlab_compat", "--n-jobs", "1", "--stop-after", "network",
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


def start(*, resume: bool = False) -> int:
    BATCH_ROOT.mkdir(parents=True, exist_ok=True)
    records = validate_catalog()
    manifest = load_manifest() if resume and MANIFEST.is_file() else {
        "created_utc": utc(), "batch_root": str(BATCH_ROOT), "runs": {}, "status": "pending"
    }
    for name in RUNS:
        entry = manifest["runs"].setdefault(name, {})
        entry.update({"file": records[name], "mode": "exact" if name == "180709_E" else "baseline",
                      "run_root": str(run_root(name)), "command": command_for(name, records[name]),
                      "log": str(run_root(name) / "batch.log")})
    manifest["status"] = "running"
    manifest["started_utc"] = manifest.get("started_utc", utc())
    save_manifest(manifest)
    for name in RUNS:
        if PAUSE.exists():
            manifest["status"] = "paused"
            save_manifest(manifest)
            return 0
        entry = manifest["runs"][name]
        if entry.get("status") == "completed":
            continue
        prior_pid = entry.get("pid")
        if prior_pid:
            try:
                os.kill(int(prior_pid), 0)
            except (OSError, ValueError):
                pass
            else:
                raise RuntimeError(f"active batch writer already running (PID {prior_pid})")
        root = run_root(name)
        root.mkdir(parents=True, exist_ok=True)
        log_path = root / "batch.log"
        entry.update({"status": "running", "started_utc": entry.get("started_utc", utc()), "pid": None})
        save_manifest(manifest)
        cmd = entry["command"]
        wrapped = ["caffeinate", "-dims", *cmd] if sys.platform == "darwin" else cmd
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n[{utc()}] START {' '.join(cmd)}\n")
            process = subprocess.Popen(wrapped, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
            entry["pid"] = process.pid
            save_manifest(manifest)
            while True:
                returncode = process.poll()
                if returncode is not None:
                    break
                if PAUSE.exists():
                    process.terminate()
                    try:
                        returncode = process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        returncode = process.wait()
                    entry.update({"status": "paused", "returncode": returncode, "finished_utc": utc()})
                    log.write(f"[{utc()}] PAUSED exit={returncode}\n")
                    manifest["status"] = "paused"
                    save_manifest(manifest)
                    return 0
                time.sleep(2)
            log.write(f"[{utc()}] EXIT {returncode}\n")
        entry["returncode"] = returncode
        entry["finished_utc"] = utc()
        entry["status"] = "completed" if returncode == 0 else "failed"
        save_manifest(manifest)
        entry["pid"] = None
        if returncode != 0:
            manifest["status"] = "failed"
            save_manifest(manifest)
            return returncode
    manifest["status"] = "paused" if PAUSE.exists() else "completed"
    manifest["finished_utc"] = utc() if manifest["status"] == "completed" else None
    save_manifest(manifest)
    return 0


def status() -> int:
    manifest = load_manifest()
    print(json.dumps(manifest, indent=2))
    return 0


def preflight() -> int:
    records = validate_catalog()
    if not ORACLE_ROOT.is_dir() or not DATASET_ROOT.is_dir():
        raise RuntimeError("180709_E workspace dataset/oracle roots are not available")
    payload = {
        "checked_utc": utc(),
        "database_root": str(DB_ROOT),
        "catalog": str(CATALOG),
        "runs": records,
        "exact_dataset_root": str(DATASET_ROOT),
        "exact_oracle_root": str(ORACLE_ROOT),
        "matlab_available": bool(__import__("shutil").which("matlab")),
        "n_jobs": 1,
        "protected_roots_untouched": True,
    }
    BATCH_ROOT.mkdir(parents=True, exist_ok=True)
    (BATCH_ROOT / "preflight.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


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
