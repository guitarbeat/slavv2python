"""CI-safe checks for the tracked Phase 1 baseline freeze JSON."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from slavv_python.analytics.parity.constants import (
    HISTORICAL_CLAIM_RUN_NAME,
    LIVE_DEST_NAMES,
    PROTECTED_DEST_NAMES,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_FREEZE_PATH = _REPO_ROOT / "docs" / "reference" / "core" / "phase1-baseline-freeze.json"
_REQUIRED_KEYS = (
    "frozen",
    "freeze_date",
    "phase1_status",
    "claim_run_root",
    "oracle_id",
    "proof_file_sha256",
    "checkpoint_sha256",
    "oracle_manifest_sha256",
    "do_not_overwrite",
    "not_stretch",
)
_PROOF_FILES = (
    "exact_proof_edges.json",
    "exact_proof_network.json",
    "exact_proof_energy.json",
    "exact_proof_vertices.json",
)
_CHECKPOINT_KEYS = ("energy", "vertices", "edges", "network")
# Git tracks compact proof summaries (~1-2 KiB). The freeze hashes the original
# dest proofs, which are much larger and usually live only on Experiment Root disks.
_COMPACT_PROOF_MAX_BYTES = 16_384


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_freeze_json_schema_and_hashes() -> None:
    payload = json.loads(_FREEZE_PATH.read_text(encoding="utf-8"))
    for key in _REQUIRED_KEYS:
        assert key in payload, f"missing freeze key {key}"
    assert payload["frozen"] is True
    assert payload["phase1_status"] == "CLOSED"
    assert payload["not_stretch"] is True
    assert tuple(payload["do_not_overwrite"]) == LIVE_DEST_NAMES
    assert HISTORICAL_CLAIM_RUN_NAME not in LIVE_DEST_NAMES
    assert HISTORICAL_CLAIM_RUN_NAME in PROTECTED_DEST_NAMES
    assert set(LIVE_DEST_NAMES) <= set(PROTECTED_DEST_NAMES)
    assert LIVE_DEST_NAMES != PROTECTED_DEST_NAMES
    proofs = payload["proof_file_sha256"]
    for name in _PROOF_FILES:
        digest = proofs[name]
        assert isinstance(digest, str)
        assert len(digest) == 64
        assert all(ch in "0123456789abcdef" for ch in digest)
    checkpoints = payload["checkpoint_sha256"]
    for name in _CHECKPOINT_KEYS:
        digest = checkpoints[name]
        assert isinstance(digest, str)
        assert len(digest) == 64
    oracle = payload["oracle_manifest_sha256"]
    assert isinstance(oracle, str)
    assert len(oracle) == 64


def test_freeze_proof_hashes_match_disk_if_present() -> None:
    payload = json.loads(_FREEZE_PATH.read_text(encoding="utf-8"))
    runs = _REPO_ROOT / "workspace" / "runs" / "oracle_180709_E"
    roots = payload["proof_file_roots"]
    missing = []
    for name, digest in payload["proof_file_sha256"].items():
        path = runs / roots[name] / name
        if not path.is_file() or path.stat().st_size <= _COMPACT_PROOF_MAX_BYTES:
            missing.append(str(path))
            continue
        assert _sha256(path) == digest, f"hash mismatch for {path}"
    if missing and len(missing) == len(payload["proof_file_sha256"]):
        pytest.skip("claim dest proofs not on this machine")
