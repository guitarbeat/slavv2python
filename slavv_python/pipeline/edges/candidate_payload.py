"""Compatibility re-exports for CandidateManifest payload helpers.

Prefer importing from ``slavv_python.pipeline.edges.candidate_manifest``.
"""

from __future__ import annotations

from slavv_python.pipeline.edges.candidate_manifest import (
    endpoint_pairs_from_connections as _candidate_endpoint_pair_set,
)
from slavv_python.pipeline.edges.candidate_manifest import (
    incident_pair_counts as _candidate_incident_pair_counts,
)
from slavv_python.pipeline.edges.candidate_manifest import (
    normalize_candidate_connection_sources,
)
from slavv_python.pipeline.edges.candidate_manifest import (
    reorder_candidate_payload as _reorder_candidate_payload,
)

__all__ = [
    "_candidate_endpoint_pair_set",
    "_candidate_incident_pair_counts",
    "_reorder_candidate_payload",
    "normalize_candidate_connection_sources",
]
