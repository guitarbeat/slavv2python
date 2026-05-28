"""Shared fixtures for parity pre-gate integration tests (ADR 0009)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
CROP_ORACLE_ID = "180709_E_crop_M"
DEFAULT_CROP_ORACLE_ROOT = REPO_ROOT / "workspace" / "oracles" / CROP_ORACLE_ID


def crop_oracle_root() -> Path | None:
    """Return the crop harness oracle root when promoted locally or via env override."""
    env_root = os.environ.get("SLAVV_CROP_ORACLE_ROOT", "").strip()
    if env_root:
        candidate = Path(env_root).expanduser().resolve()
        return candidate if candidate.is_dir() else None
    return DEFAULT_CROP_ORACLE_ROOT if DEFAULT_CROP_ORACLE_ROOT.is_dir() else None


@pytest.fixture
def crop_harness_oracle_root() -> Path:
    """Skip tier-2 tests when the crop harness oracle has not been promoted."""
    root = crop_oracle_root()
    if root is None:
        pytest.skip(
            "Crop harness oracle not available — promote 180709_E_crop_M per "
            "docs/reference/workflow/PARITY_PRE_GATE.md or set SLAVV_CROP_ORACLE_ROOT"
        )
    return root
