from __future__ import annotations

from slavv_python.analytics.parity.runs.writer_lease import (
    load_writer_lease,
    write_writer_lease,
    writer_lease_path,
)


def test_writer_lease_replaces_legacy_ownership_atomically(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "slavv_python.analytics.parity.runs.writer_lease.resolve_python_commit",
        lambda _root: "abc123",
    )
    run_dir = tmp_path / "run"
    first = write_writer_lease(run_dir, pid=100, command="first", stage="energy", status="running")
    second = write_writer_lease(
        run_dir, pid=200, command="second", stage="vertices", status="running"
    )

    assert first["pid"] == 100
    assert second["pid"] == 200
    assert load_writer_lease(run_dir) == second
    assert writer_lease_path(run_dir).with_name("writer_lease.json.sha256").is_file()
