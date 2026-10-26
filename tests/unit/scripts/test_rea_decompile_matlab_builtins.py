"""Unit tests for REA MATLAB native built-ins reverse-engineering probe."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from scripts.curation.rea_decompile_matlab_builtins import (
    DEFAULT_MACA64_BIN,
    KNOWN_TARGETS,
    analyze_target,
    find_objdump,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_known_targets_registry() -> None:
    """Verify registry has required targets and fields."""
    assert "watershed_meyer" in KNOWN_TARGETS
    assert "watershed_vs" in KNOWN_TARGETS
    assert "bwlabeln" in KNOWN_TARGETS

    for _name, target in KNOWN_TARGETS.items():
        assert "dylib" in target
        assert "symbol" in target
        assert "demangled" in target
        assert target["dylib"].endswith(".dylib")


def test_find_objdump() -> None:
    """Verify an objdump tool is discoverable."""
    objdump = find_objdump()
    assert objdump.is_file()
    assert "objdump" in objdump.name


@pytest.mark.skipif(
    not DEFAULT_MACA64_BIN.is_dir(),
    reason="MATLAB maca64 binaries not found at expected path",
)
def test_analyze_target_watershed_meyer(tmp_path: Path) -> None:
    """Verify live analysis of watershed_meyer on installed MATLAB dylib."""
    summary = analyze_target("watershed_meyer", DEFAULT_MACA64_BIN)
    assert summary.target_name == "watershed_meyer"
    assert summary.instruction_count > 1000
    assert summary.branch_count > 100
    assert summary.call_count > 50
    assert any("check_inputs" in c for c in summary.callees)
