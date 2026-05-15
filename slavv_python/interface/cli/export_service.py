"""Network export helpers for the SLAVV CLI."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def save_network_export(
    results: dict[str, Any],
    path: str,
    *,
    format: str,
) -> None:
    """Export the network results dictionary to the specified format and path."""
    logger.info(f"Exporting network to {path} ({format})...")

    from ...visualization.network_plots.exports import (
        export_casx,
        export_json,
        export_mat,
        export_vmv,
    )

    vertices = results.get("vertices", {})
    edges = results.get("edges", {})
    network = results.get("network", {})
    parameters = results.get("parameters", {})

    if format == "json":
        export_json(results, path)
    elif format == "mat":
        export_mat(vertices, edges, network, parameters, path)
    elif format == "casx":
        export_casx(vertices, edges, network, parameters, path)
    elif format == "vmv":
        export_vmv(vertices, edges, network, parameters, path)
    else:
        raise ValueError(f"Unsupported export format: {format}")


__all__ = ["save_network_export"]
