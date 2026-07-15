"""Vascular edge extraction subpackage.

## Public lifecycle (preferred)

* ``EdgeManager`` — stage facade (discovery → selection workflow → Edge Set)
* ``select_and_finalize_edge_set`` — deep post-discovery path (choose → bridge → finalize)
* ``select_edge_discovery`` / ``TracingDiscovery`` / ``WatershedDiscovery`` —
  Edge Discovery strategy seam (domain names from AGENTS glossary)

## Edge Discovery (Certification-relevant)

| Domain term | Class | Engine |
|-------------|-------|--------|
| Tracing Discovery (Paper Path) | ``TracingDiscovery`` | directional candidates |
| Watershed Discovery (Exact Route) | ``WatershedDiscovery`` | ``generate_watershed_candidates`` → MATLAB global watershed |

Legacy class aliases: ``MaintainedTracingDiscovery``, ``FrontierTracingDiscovery``.

## Not Certification Edge Discovery

``extract_edges_watershed`` / ``extract_edges_watershed_resumable`` and
``naive_watershed`` use **skimage label adjacency**. They are experimental /
legacy helpers and must not be used as the Exact Route or ADR 0012 proof path.
"""

from __future__ import annotations

from .candidate_manifest import CandidateManifest
from .discovery import (
    EdgeDiscovery,
    EdgeDiscoveryContext,
    FrontierTracingDiscovery,
    MaintainedTracingDiscovery,
    TracingDiscovery,
    WatershedDiscovery,
    select_edge_discovery,
)
from .edges import (
    _load_edge_units,
    extract_edges,
    extract_edges_resumable,
    extract_edges_watershed,
    extract_edges_watershed_resumable,
)
from .manager import EdgeManager
from .selection_workflow import select_and_finalize_edge_set

__all__ = [
    "CandidateManifest",
    "EdgeDiscovery",
    "EdgeDiscoveryContext",
    "EdgeManager",
    "FrontierTracingDiscovery",
    "MaintainedTracingDiscovery",
    "TracingDiscovery",
    "WatershedDiscovery",
    "_load_edge_units",
    "extract_edges",
    "extract_edges_resumable",
    "extract_edges_watershed",
    "extract_edges_watershed_resumable",
    "select_and_finalize_edge_set",
    "select_edge_discovery",
]
