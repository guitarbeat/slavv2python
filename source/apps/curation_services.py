"""Focused app-facing helpers for curation workflows."""

from __future__ import annotations

from typing import Any

from source.apps.curation_state import apply_curated_session_results


def apply_curated_results(
    session_state,
    curated_vertices: dict[str, object],
    curated_edges: dict[str, object],
    *,
    curation_mode: str,
) -> tuple[dict[str, int], dict[str, int]]:
    """Sync curated vertices and edges into session state with a rebuilt network."""
    return apply_curated_session_results(
        session_state,
        curated_vertices,
        curated_edges,
        curation_mode=curation_mode,
    )


def run_interactive_curator(
    energy_data: Any,
    vertices_data: Any,
    edges_data: Any,
    backend: str = "qt",
):
    """Import desktop curator backends lazily so the web app can load without GUI deps."""
    backend_name = str(backend).strip().lower()
    if backend_name in {"qt", "qt_pyvista", "pyvista"}:
        from source.visualization.interactive_curator import run_curator

        return run_curator(energy_data, vertices_data, edges_data)
    if backend_name == "napari":
        from source.visualization.napari_curator import run_curator_napari

        return run_curator_napari(energy_data, vertices_data, edges_data)
    raise ValueError("curator backend must be 'qt' or 'napari'")


__all__ = ["apply_curated_results", "run_interactive_curator"]
