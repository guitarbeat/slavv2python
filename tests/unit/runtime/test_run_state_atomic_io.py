"""Tests for low-level run-state file writes."""

from __future__ import annotations

import json

from slavv_python.runtime.run_state import atomic_write_json


def test_atomic_write_json_replaces_previous_content(tmp_path):
    path = tmp_path / "snapshot.json"

    atomic_write_json(path, {"stage": "energy", "progress": 0.25})
    atomic_write_json(path, {"stage": "network", "progress": 1.0})

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "progress": 1.0,
        "stage": "network",
    }
