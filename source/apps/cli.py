"""Command-line interface for source."""

from __future__ import annotations

from .cli_commands import (
    _handle_analyze_command,
    _handle_info_command,
    _handle_plot_command,
    _handle_run_command,
    _handle_status_command,
)
from .cli_dispatch import CLI_COMMAND_HANDLERS, dispatch_cli_command
from .cli_export_service import save_network_export as _save_network_export
from .cli_exported_network import (
    _build_strands_from_edge_connections,
    _infer_image_shape_from_vertices,
    _load_exported_network_json,
    _load_exported_results,
    _normalize_exported_edge_connections,
)
from .cli_parser import _EXPORT_FILE_NAMES, _build_cli_parser
from .cli_shared import (
    _DETAILED_LOG_FORMAT,
    _SIMPLE_LOG_FORMAT,
    _build_export_artifacts,
    _build_pipeline_parameters,
    _configure_logging,
    _expand_export_formats,
    _require_existing_file,
)

__all__ = [
    "CLI_COMMAND_HANDLERS",
    "_DETAILED_LOG_FORMAT",
    "_EXPORT_FILE_NAMES",
    "_SIMPLE_LOG_FORMAT",
    "_build_cli_parser",
    "_build_export_artifacts",
    "_build_pipeline_parameters",
    "_build_strands_from_edge_connections",
    "_configure_logging",
    "_expand_export_formats",
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


def main(argv=None):
    """CLI entry point."""
    parser = _build_cli_parser()
    args = parser.parse_args(argv)

    if args.version:
        from source import __version__

        print(f"slavv {__version__}")
        return

    dispatch_cli_command(parser, args)


if __name__ == "__main__":
    main()
