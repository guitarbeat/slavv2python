"""Network export helpers for the SLAVV CLI."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...io import Network

logger = logging.getLogger(__name__)


def save_network_export(
    network: Network,
    path: str,
    *,
    format: str,
) -> None:
    """Export the network to the specified format and path."""
    logger.info(f"Exporting network to {path} ({format})...")

    if format == "json":
        network.save_json(path)
    elif format == "mat":
        network.save_mat(path)
    elif format == "casx":
        network.save_casx(path)
    elif format == "vmv":
        network.save_vmv(path)
    else:
        raise ValueError(f"Unsupported export format: {format}")


__all__ = ["save_network_export"]
