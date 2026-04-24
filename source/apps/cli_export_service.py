"""Export helpers for CLI command handlers."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from .cli_parser import _EXPORT_FILE_NAMES

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)


def save_network_export(
    format_type: str,
    *,
    output_dir: str,
    network_obj: Any,
    results: Mapping[str, Any],
) -> str | None:
    """Persist one export format and return the written path when successful."""
    export_path = os.path.join(output_dir, _EXPORT_FILE_NAMES[format_type])

    if format_type == "mat":
        try:
            from slavv.visualization import NetworkVisualizer

            vis = NetworkVisualizer()
            vis.export_network_data(
                {
                    "vertices": results.get("vertices", {}),
                    "edges": results.get("edges", {}),
                    "network": results.get("network", {}),
                    "parameters": results.get("parameters", {}),
                },
                export_path,
                format="mat",
            )
            logger.info("Saved MAT to %s", export_path)
            return export_path
        except ImportError as exc:
            logger.warning("Error saving MAT file: %s", exc)
            return None

    from slavv.io import (
        save_network_to_casx,
        save_network_to_csv,
        save_network_to_json,
        save_network_to_vmv,
    )

    exporters = {
        "csv": save_network_to_csv,
        "json": save_network_to_json,
        "casx": save_network_to_casx,
        "vmv": save_network_to_vmv,
    }
    exporters[format_type](network_obj, export_path)
    logger.info("Saved %s to %s", format_type.upper(), export_path)
    return export_path


__all__ = ["save_network_export"]
