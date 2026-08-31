"""Unit tests for shared matlab2python audit ParityModuleMap."""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_MAP_MODULE = _REPO / "workspace/experiments/matlab2python_audit/tools/parity_module_map.py"
if not _MAP_MODULE.is_file():
    pytest.skip(
        "matlab2python audit tools absent (workspace/ is gitignored)",
        allow_module_level=True,
    )

from workspace.experiments.matlab2python_audit.tools.parity_module_map import (  # noqa: E402
    classify_and_resolve,
    resolve_python_counterpart,
    rewrite_manifest_counterparts,
)


def test_core_energy_maps_to_chunked_energy() -> None:
    target, classification, stage = classify_and_resolve("get_energy_V202.m", "energy")
    assert classification == "CORE_ALGORITHMIC"
    assert stage == "energy"
    assert target.endswith("matlab_get_energy_v202_chunked.py")
    assert resolve_python_counterpart("get_energy_V202.m", "energy") == target


def test_visualization_classified_without_phantom_stage_stem() -> None:
    target, classification, _ = classify_and_resolve("animate_strands_3D.m", "network")
    assert classification == "VISUALIZATION_TOOL"
    assert target.startswith("slavv_python/visualization")
    assert resolve_python_counterpart("animate_strands_3D.m", "network") == target


def test_unmapped_non_core_returns_empty_counterpart() -> None:
    assert resolve_python_counterpart("totally_unknown_helper.m", "edges") == ""


def test_rewrite_manifest_counterparts_updates_phantoms() -> None:
    manifest = {
        "modules": [
            {
                "matlab_file": "get_energy_V202.m",
                "stage": "energy",
                "python_counterpart": "slavv_python/pipeline/energy/get_energy_V202.py",
            },
            {
                "matlab_file": "animate_strands_3D.m",
                "stage": "network",
                "python_counterpart": "slavv_python/pipeline/network/animate_strands_3D.py",
            },
        ]
    }
    changed = rewrite_manifest_counterparts(manifest)
    assert changed == 2
    assert manifest["modules"][0]["python_counterpart"].endswith(
        "matlab_get_energy_v202_chunked.py"
    )
    assert manifest["modules"][0]["mapping_class"] == "CORE_ALGORITHMIC"
    assert "visualization" in manifest["modules"][1]["python_counterpart"]


@pytest.mark.unit
def test_e9_parity_module_map_doc_is_audit_aid_not_certification() -> None:
    """E9 / R2: shared map module states Certification stays on Oracle + proofs."""
    text = _MAP_MODULE.read_text(encoding="utf-8")
    assert "audit aid only" in text
    assert "Certification" in text
    assert "Exact Proof" in text


@pytest.mark.unit
def test_e9_sampled_inventory_resolves_via_shared_map_without_phantoms() -> None:
    """E9: sampled CORE mappings resolve under slavv_python without stage-stem phantoms."""
    samples = [
        ("get_energy_V202.m", "energy"),
        ("get_vertices_V200.m", "vertices"),
        ("get_edges_by_watershed.m", "edges"),
        ("get_network_V190.m", "network"),
    ]
    for matlab_file, stage in samples:
        target, classification, mapped_stage = classify_and_resolve(matlab_file, stage)
        assert classification == "CORE_ALGORITHMIC"
        assert mapped_stage == stage
        assert target.startswith("slavv_python/")
        # No phantom get_* stem copy under pipeline stage folder
        assert f"pipeline/{stage}/{Path(matlab_file).stem}.py" not in target
        assert resolve_python_counterpart(matlab_file, stage) == target
