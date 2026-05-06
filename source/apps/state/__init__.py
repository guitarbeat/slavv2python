"""State management for user-facing applications."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...models import normalize_pipeline_result

if TYPE_CHECKING:
    from collections.abc import Mapping


def normalize_state_results(processing_results: Mapping[str, Any]) -> dict[str, Any]:
    """Return a normalized dict payload for app state consumers."""
    return normalize_pipeline_result(processing_results).to_dict()


__all__ = ["normalize_state_results"]
