"""Live-MATLAB gated localization smoke for double_junction_32.

Excluded from the default unit CI gate via ``parity`` + ``slow`` markers.
Skips cleanly when MATLAB is not installed.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
LADDER_SCRIPT = REPO_ROOT / "scripts" / "ladder" / "run.py"


def _load_ladder_module():
    spec = importlib.util.spec_from_file_location(
        "run_synthetic_complexity_ladder",
        LADDER_SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ladder_mod():
    return _load_ladder_module()


@pytest.mark.parity
@pytest.mark.slow
@pytest.mark.integration
def test_live_localize_double_junction_32_schema(ladder_mod):
    matlab = ladder_mod.resolve_matlab_exe()
    if matlab is None:
        pytest.skip("MATLAB executable not found")

    report = ladder_mod.run_localize(
        "double_junction_32",
        skip_matlab=False,
        reuse_python=False,
    )
    loc = report["localization"]
    assert "NOT Certification" in report["note"]
    assert report["claim_of_record"] is True
    assert report["rung_id"] == "double_junction_32"
    assert Path(report["report_path"]).is_file()

    payload = json.loads(Path(report["report_path"]).read_text(encoding="utf-8"))
    assert "localization" in payload
    assert "NOT Certification" in payload["note"]

    # Loader must not spuriously report MATLAB strand count 1 when N>1.
    m_count = (loc.get("counts") or {}).get("matlab_strands")
    if loc.get("comparable") and m_count is not None:
        assert m_count != 1 or (loc.get("counts") or {}).get("python_strands") == 1

    assert loc.get("outcome") in {
        "match",
        "measurement_fixed_match",
        "first_diff",
        "inconclusive",
    }
