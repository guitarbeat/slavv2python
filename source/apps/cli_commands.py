"""Command handlers for the SLAVV CLI."""

from __future__ import annotations

import logging
import os
import sys

from .cli_analysis_service import calculate_exported_network_stats
from .cli_export_service import save_network_export
from .cli_exported_network import _load_exported_results
from .cli_info_service import load_info_lines
from .cli_reporting import build_analysis_output_lines
from .cli_run_service import (
    build_exportable_network,
    build_run_completion_lines,
    filter_export_formats,
    format_run_event_line,
    resolve_effective_run_dir,
    update_run_export_task,
)
from .cli_shared import (
    _DETAILED_LOG_FORMAT,
    _SIMPLE_LOG_FORMAT,
    _build_export_artifacts,
    _build_pipeline_parameters,
    _configure_logging,
    _expand_export_formats,
    _require_existing_file,
)
from .cli_status_service import build_status_output_lines, load_status_snapshot

logger = logging.getLogger(__name__)


def _handle_info_command() -> None:
    """Print version and system information."""
    from source import __version__
    from source.utils import get_system_info

    for line in load_info_lines(version=__version__, system_info=get_system_info()):
        print(line)


def _handle_run_command(args) -> None:
    """Execute the SLAVV processing pipeline."""
    from source import SlavvPipeline
    from source.io import Network, load_tiff_volume
    from source.runtime import RunContext, build_status_lines, load_run_snapshot

    _require_existing_file(args.input, label="input file")
    _configure_logging(args.verbose, format_string=_DETAILED_LOG_FORMAT)

    logger.info("Loading volume from %s", args.input)
    image = load_tiff_volume(args.input)

    parameters = _build_pipeline_parameters(args)
    effective_run_dir = resolve_effective_run_dir(
        run_dir=args.run_dir,
        output_dir=args.output,
    )

    last_event_line = {"value": ""}

    def event_callback(event) -> None:
        line = format_run_event_line(event)
        if line != last_event_line["value"]:
            print(line)
            last_event_line["value"] = line

    processor = SlavvPipeline()
    results = processor.run(
        image,
        parameters,
        event_callback=event_callback,
        run_dir=effective_run_dir,
        stop_after=args.stop_after,
        force_rerun_from=args.force_rerun_from,
    )

    os.makedirs(args.output, exist_ok=True)
    network_obj = build_exportable_network(results, network_factory=Network)

    export_formats, export_warnings = filter_export_formats(
        _expand_export_formats(args.export),
        results,
    )
    run_snapshot = load_run_snapshot(effective_run_dir) if effective_run_dir else None
    for warning_line in export_warnings:
        logger.warning(warning_line)

    for fmt in export_formats:
        save_network_export(
            fmt,
            output_dir=args.output,
            network_obj=network_obj,
            results=results,
            run_snapshot=run_snapshot,
            run_dir=effective_run_dir,
        )

    snapshot = update_run_export_task(
        effective_run_dir=effective_run_dir,
        export_formats=export_formats,
        output_dir=args.output,
        snapshot_loader=load_run_snapshot,
        context_factory=RunContext.from_existing,
        artifact_builder=_build_export_artifacts,
    )

    for line in build_run_completion_lines(
            effective_run_dir=effective_run_dir,
            output_dir=args.output,
            snapshot=snapshot,
            status_line_builder=build_status_lines,
    ):
        print(line)


def _handle_status_command(args) -> None:
    """Render run status from a run directory."""
    from source.runtime import build_status_lines, load_run_snapshot

    _configure_logging(args.verbose, format_string=_SIMPLE_LOG_FORMAT)

    snapshot = load_status_snapshot(
        args.run_dir,
        snapshot_loader=load_run_snapshot,
    )
    if snapshot is None:
        print(f"Error: no run snapshot found in {args.run_dir}", file=sys.stderr)
        sys.exit(1)

    for line in build_status_output_lines(snapshot, status_line_builder=build_status_lines):
        print(line)


def _handle_analyze_command(args) -> None:
    """Analyze an exported network JSON file and print statistics."""
    from source.analysis import calculate_network_statistics

    _configure_logging(args.verbose, format_string=_SIMPLE_LOG_FORMAT)
    results = _load_exported_results(args.input)

    logger.info("Calculating statistics...")
    stats = calculate_exported_network_stats(
        results,
        statistics_fn=calculate_network_statistics,
    )

    for line in build_analysis_output_lines(stats):
        print(line)


def _handle_plot_command(args) -> None:
    """Generate interactive plots from exported network JSON."""
    from source.visualization.network_plots import NetworkVisualizer

    _configure_logging(args.verbose, format_string=_SIMPLE_LOG_FORMAT)
    results = _load_exported_results(args.input)

    vis = NetworkVisualizer()
    logger.info("Generating length-weighted histograms...")
    fig = vis.plot_length_weighted_histograms(
        results.get("vertices", {}),
        results.get("edges", {}),
        results.get("parameters", {}),
        number_of_bins=args.number_of_bins,
    )

    fig.write_html(args.output)
    print(f"Saved interactive plots to {args.output}")
