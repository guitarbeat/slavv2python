"""Portfolio E6 doc honesty: PROJECT / ORIGINAL_REQUEST production-only probes.

These assertions do not import gitignored workspace audit tools, so they run in CI.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]


@pytest.mark.unit
def test_e6_project_and_original_request_agree_on_production_only_probes() -> None:
    """E6 / KTD2: living docs must not claim dual-run of cleaned_transpiled."""
    project = (_REPO / "PROJECT.md").read_text(encoding="utf-8")
    original = (_REPO / "ORIGINAL_REQUEST.md").read_text(encoding="utf-8")
    for text, label in ((project, "PROJECT.md"), (original, "ORIGINAL_REQUEST.md")):
        assert "production_probe" in text, f"{label} missing production_probe"
        assert "dual-run" in text.lower(), label
        assert "not" in text.lower(), label
        assert "cleaned_transpiled" in text
        assert "through transpiled logic vs" not in text
        assert "through both the translated MATLAB logic and the current" not in text
