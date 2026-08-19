"""State management for user-facing applications."""

from __future__ import annotations

from typing import Any

from slavv_python.schema.app_run import AppRunState, get_app_run


def normalize_state_results(processing_results: AppRunState | Any) -> dict[str, object]:
    """Return a normalized dict payload for export boundaries only."""
    return AppRunState.from_value(processing_results).to_dict()


__all__ = ["AppRunState", "get_app_run", "normalize_state_results"]
