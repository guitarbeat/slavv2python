"""Candidate manifest append helpers."""

from __future__ import annotations

from typing import Any

from slavv_python.pipeline.edges.discovery import CandidateManifest


def _apply_manifest_to_dict(target: dict[str, Any], manifest: CandidateManifest) -> None:
    """Write manifest fields back into a legacy dict payload shell."""
    updated = manifest.to_payload()
    for key in list(target.keys()):
        if key not in updated:
            del target[key]
    target.update(updated)


def _append_candidate_unit(target: dict[str, Any], unit_payload: dict[str, Any]) -> None:
    """Append a per-origin candidate payload into the aggregate candidate manifest."""
    manifest = CandidateManifest.from_payload(target)
    manifest.append_unit(unit_payload)
    _apply_manifest_to_dict(target, manifest)
