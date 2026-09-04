"""Trust claim labels for curation GUI surfaces (ADR 0014)."""

from __future__ import annotations

# Operator-facing workflow radio labels (Streamlit Curation page).
BROWSER_TRUST_WORKFLOW = "Trust path: MATLAB-familiar browser curator"
DESKTOP_REVIEW_WORKFLOW = "Desktop manual review (experimental)"

FORBIDDEN_TRUST_CLAIM_SUBSTRINGS = (
    "1:1",
    "MATLAB-identical",
    "feature parity with the MATLAB GCI",
)


def trust_claim_chrome_visible(degraded_reason: str | None) -> bool:
    """Return True when Trust MATLAB-familiar chrome may be shown.

    Degraded / fallback browser presentations must not inherit the Trust badge
    (ADR 0014 / R3).
    """
    if degraded_reason is None:
        return True
    return not bool(str(degraded_reason).strip())
