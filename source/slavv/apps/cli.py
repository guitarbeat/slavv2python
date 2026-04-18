"""Command-line interface for SLAVV."""

from __future__ import annotations

import sys

from .cli_commands import (
    _handle_analyze_command,
    _handle_import_matlab_command,
    _handle_info_command,
    _handle_parity_proof_command,
    _handle_plot_command,
    _handle_run_command,
    _handle_status_command,
    _save_network_export,
)
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
    "_handle_import_matlab_command",
    "_handle_info_command",
    "_handle_parity_proof_command",
    "_handle_plot_command",
    "_handle_run_command",
    "_handle_status_command",
    "_infer_image_shape_from_vertices",
    "_load_exported_network_json",
    "_load_exported_results",
    "_normalize_exported_edge_connections",
    "_require_existing_file",
    "_save_network_export",
    "main",
]


def main(argv=None):
    """CLI entry point."""
    parser = _build_cli_parser()
    args = parser.parse_args(argv)

    if args.version:
        from slavv import __version__

        print(f"slavv {__version__}")
        return

    if args.command == "info":
        _handle_info_command()
    elif args.command == "run":
        _handle_run_command(args)
    elif args.command == "import-matlab":
        _handle_import_matlab_command(args)
    elif args.command == "status":
        _handle_status_command(args)
    elif args.command == "parity-proof":
        _handle_parity_proof_command(args)
    elif args.command == "analyze":
        _handle_analyze_command(args)
    elif args.command == "plot":
        _handle_plot_command(args)
    else:
        parser.print_help()
        sys.exit(0 if args.command is None else 1)


if __name__ == "__main__":
    main()
