"""E5 portfolio tests: MATLAB-edge → Python Network isolation."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

_REPO = Path(__file__).resolve().parents[3]
_CROP_ORACLE = _REPO / "workspace/oracles/180709_E_crop_M_v2"
_CROP_RUN = _REPO / "workspace/runs/oracle_180709_E/crop_M_exact_v3"
_EDGES = _CROP_ORACLE / "03_Analysis/normalized/oracle/edges.pkl"
_VERTICES = _CROP_ORACLE / "03_Analysis/normalized/oracle/vertices.pkl"
_NETWORK = _CROP_ORACLE / "03_Analysis/normalized/oracle/network.pkl"
_SCRIPT = _REPO / "scripts" / "network_matlab_edge_isolation.py"


def _load_e5_script() -> Any:
    name = "network_matlab_edge_isolation"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load E5 script: {_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


e5 = _load_e5_script()


@pytest.mark.unit
def test_e5_ae4_non_claim_forbids_phase1_closure_language() -> None:
    text = e5.R2_NON_CLAIM
    assert "Phase 1" in text
    assert "closure" in text.lower()
    assert "ADR 0012" in text


@pytest.mark.unit
def test_e5_missing_artifacts_blocked_not_falsified(tmp_path: Path) -> None:
    result = e5.run_e5_isolation(oracle_root=tmp_path / "missing_oracle")
    assert result["status"] == "blocked"
    assert "Phase 1" in result["r2_non_claim"]
    assert "closure" in result["r2_non_claim"].lower()


@pytest.mark.unit
@pytest.mark.skipif(
    not (_EDGES.is_file() and _VERTICES.is_file() and _NETWORK.is_file()),
    reason="E5 blocked: crop isolation artifacts absent (KTD4 — not falsified)",
)
def test_e5_crop_matlab_edge_isolation_records_verdict() -> None:
    """Run crop isolation; record pass/fail (do not treat missing as fail).

    Success signal is multiset match. A fail with matched MATLAB Edge Set
    falsifies the 'Network is exact under MATLAB edges' hypothesis on this
    crop surface — still not a Network rewrite mandate (portfolio scope).
    """
    result = e5.run_e5_isolation(oracle_root=_CROP_ORACLE, run_dir=_CROP_RUN)
    assert result["status"] != "blocked"
    assert result["status"] in {"pass", "fail"}, result
    assert "fresh claim root" in result["ae4"]
    assert "not Phase 1 closure" in result["ae4"].lower() or "Phase 1" in result["ae4"]
    assert "r2_non_claim" in result
    if result["status"] == "pass":
        assert result["passed"] is True
        assert result["n_only_matlab"] == 0
        assert result["n_only_python"] == 0
    else:
        # Hypothesis falsified on this surface — residual counts must be nonzero.
        assert result["passed"] is False
        assert result["n_only_matlab"] + result["n_only_python"] > 0
