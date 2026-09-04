"""Regression guards for ADR 0014 curator Trust claim language."""

from __future__ import annotations

from pathlib import Path

import pytest

from slavv_python.interface.streamlit.curation_trust_labels import (
    BROWSER_TRUST_WORKFLOW,
    DESKTOP_REVIEW_WORKFLOW,
    FORBIDDEN_TRUST_CLAIM_SUBSTRINGS,
    trust_claim_chrome_visible,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_INTERACTIVE_CURATOR = _REPO_ROOT / "slavv_python" / "visualization" / "interactive_curator.py"
_CURATION_VIEW = _REPO_ROOT / "slavv_python" / "interface" / "streamlit" / "views" / "curation.py"


@pytest.mark.unit
def test_browser_workflow_is_sole_trust_matlab_familiar_chooser_label() -> None:
    assert "Trust" in BROWSER_TRUST_WORKFLOW
    assert "MATLAB-familiar" in BROWSER_TRUST_WORKFLOW
    assert "MATLAB-style" not in DESKTOP_REVIEW_WORKFLOW
    assert "1:1" not in DESKTOP_REVIEW_WORKFLOW
    assert "Trust" not in DESKTOP_REVIEW_WORKFLOW


@pytest.mark.unit
def test_curation_view_uses_trust_label_constants() -> None:
    source = _CURATION_VIEW.read_text(encoding="utf-8")
    assert "BROWSER_TRUST_WORKFLOW" in source
    assert "DESKTOP_REVIEW_WORKFLOW" in source
    assert "Desktop manual review (MATLAB-style)" not in source
    assert "MATLAB-style browser curator" not in source


@pytest.mark.unit
def test_interactive_curator_docstring_rejects_one_to_one_claim() -> None:
    source = _INTERACTIVE_CURATOR.read_text(encoding="utf-8")
    docstring = source.split('"""', 2)[1]
    for forbidden in FORBIDDEN_TRUST_CLAIM_SUBSTRINGS:
        assert forbidden not in docstring, f"found forbidden claim language: {forbidden!r}"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        (None, True),
        ("", True),
        ("   ", True),
        ("Intensity volume unavailable", False),
    ],
)
def test_trust_claim_chrome_visible(reason: str | None, expected: bool) -> None:
    assert trust_claim_chrome_visible(reason) is expected
