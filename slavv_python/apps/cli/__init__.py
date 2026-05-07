"""Grouped CLI package for the SLAVV command-line interface."""

from __future__ import annotations

from .commands import (
    _handle_analyze_command,
    _handle_info_command,
    _handle_plot_command,
    _handle_run_command,
    _handle_status_command,
)
from .dispatch import CLI_COMMAND_HANDLERS, dispatch_cli_command
from .entrypoint import main
from .export_service import save_network_export as _save_network_export
from .exported_network import (
    _build_strands_from_edge_connections,
    _infer_image_shape_from_vertices,
    _load_exported_network_json,
    _load_exported_results,
    _normalize_exported_edge_connections,
)
from .parser import _EXPORT_FILE_NAMES, _build_cli_parser
from .shared import _require_existing_file

__all__ = [
    "CLI_COMMAND_HANDLERS",
    "_EXPORT_FILE_NAMES",
    "_build_cli_parser",
    "_build_strands_from_edge_connections",
    "_handle_analyze_command",
    "_handle_info_command",
    "_handle_plot_command",
    "_handle_run_command",
    "_handle_status_command",
    "_infer_image_shape_from_vertices",
    "_load_exported_network_json",
    "_load_exported_results",
    "_normalize_exported_edge_connections",
    "_require_existing_file",
    "_save_network_export",
    "dispatch_cli_command",
    "main",
]
