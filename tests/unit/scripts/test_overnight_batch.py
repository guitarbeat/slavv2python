from __future__ import annotations

import hashlib
import sqlite3

import pytest
import tifffile
from scripts.parity import overnight_batch as batch


def test_validate_catalog_accepts_hydrated_database(monkeypatch, tmp_path):
    db_root = tmp_path / "db"
    catalog_dir = db_root / "data/metadata"
    scan_dir = db_root / "data/raw/scans"
    catalog_dir.mkdir(parents=True)
    scan_dir.mkdir(parents=True)
    rows = []
    for name in batch.RUNS:
        path = scan_dir / f"{name}.tif"
        tifffile.imwrite(path, __import__("numpy").zeros((1, 1, 1), dtype="uint16"))
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        rows.append((name, f"data/raw/scans/{name}.tif", digest, len(payload), "1 x 1 x 1", "ok", "active_canonical", 1, 1))
    connection = sqlite3.connect(catalog_dir / "neurovasc_catalog.sqlite")
    connection.execute("create table samples (filename text, raw_relpath text, raw_sha256 text, raw_bytes integer, resolution text, qa_status text, record_disposition text, is_active_record integer, has_raw integer)")
    connection.executemany("insert into samples values (?,?,?,?,?,?,?,?,?)", rows)
    connection.commit()
    connection.close()
    monkeypatch.setattr(batch, "DB_ROOT", db_root)
    monkeypatch.setattr(batch, "CATALOG", catalog_dir / "neurovasc_catalog.sqlite")
    result = batch.validate_catalog()
    assert set(result) == set(batch.RUNS)


def test_validate_catalog_rejects_hash_mismatch(monkeypatch, tmp_path):
    db_root = tmp_path / "db"
    catalog_dir = db_root / "data/metadata"
    scan_dir = db_root / "data/raw/scans"
    catalog_dir.mkdir(parents=True)
    scan_dir.mkdir(parents=True)
    connection = sqlite3.connect(catalog_dir / "neurovasc_catalog.sqlite")
    connection.execute("create table samples (filename text, raw_relpath text, raw_sha256 text, raw_bytes integer, resolution text, qa_status text, record_disposition text, is_active_record integer, has_raw integer)")
    for name in batch.RUNS:
        (scan_dir / f"{name}.tif").write_bytes(b"not-a-tiff")
        connection.execute("insert into samples values (?,?,?,?,?,?,?,?,?)", (name, f"data/raw/scans/{name}.tif", "0" * 64, 10, "1 x 1 x 1", "ok", "active_canonical", 1, 1))
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
