"""Unit tests for the MATLAB Image Processing built-ins audit probe."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from scripts.curation.matlab_builtins_audit import (
    DEFAULT_MACA64_BIN,
    TARGET_REGISTRY,
    generate_specification_documents,
    run_subsystem_audit,
    verify_distance_invariants,
    verify_imfilter_invariants,
    verify_morphology_invariants,
    verify_watershed_invariants,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_target_registry_completeness() -> None:
    """Verify all 4 core built-in subsystems are present in the registry."""
    expected_subsystems = {"watershed", "graydist", "morphop", "imfilter"}
    assert set(TARGET_REGISTRY.keys()) == expected_subsystems

    for _key, info in TARGET_REGISTRY.items():
        assert "dylib" in info
        assert "primary_symbol" in info
        assert "primary_name" in info
        assert "doc_file" in info
        assert "python_module" in info


def test_individual_invariant_verifiers() -> None:
    """Verify synthetic invariant checks pass."""
    ok_ws, msg_ws = verify_watershed_invariants()
    assert ok_ws is True
    assert "Lowest Fortran Linear Index" in msg_ws

    ok_gd, msg_gd = verify_distance_invariants()
    assert ok_gd is True
    assert "Euclidean EDT integer squared accumulator" in msg_gd

    ok_mo, msg_mo = verify_morphology_invariants()
    assert ok_mo is True
    assert "Linear structuring element" in msg_mo

    ok_im, msg_im = verify_imfilter_invariants()
    assert ok_im is True
    assert "Boundary extension" in msg_im


@pytest.mark.skipif(
    not DEFAULT_MACA64_BIN.is_dir(),
    reason="MATLAB maca64 binaries not found at expected path",
)
@pytest.mark.parametrize("target_key", ["watershed", "graydist", "morphop", "imfilter"])
def test_live_subsystem_audits(target_key: str) -> None:
    """Verify live disassembly and inspection of installed MATLAB dylibs."""
    audit = run_subsystem_audit(target_key, DEFAULT_MACA64_BIN, verify=True)
    assert audit.target_key == target_key
    assert audit.instructions > 50
    assert audit.invariants_verified is True


@pytest.mark.skipif(
    not DEFAULT_MACA64_BIN.is_dir(),
    reason="MATLAB maca64 binaries not found at expected path",
)
def test_specification_generation(tmp_path: Path) -> None:
    """Verify ground-truth markdown generation across all targets."""
    audits = {k: run_subsystem_audit(k, DEFAULT_MACA64_BIN, verify=True) for k in TARGET_REGISTRY}
    generate_specification_documents(audits, tmp_path)

    for info in TARGET_REGISTRY.values():
        doc_file = tmp_path / info["doc_file"]
        assert doc_file.is_file()
        content = doc_file.read_text(encoding="utf-8")
        assert len(content) > 500
        assert info["primary_name"] in content
