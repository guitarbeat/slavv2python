"""Command handlers for the SLAVV CLI."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np

from .cli_analysis_service import calculate_exported_network_stats
from .cli_export_service import save_network_export
from .cli_exported_network import _load_exported_results
from .cli_info_service import load_info_lines
from .cli_reporting import build_analysis_output_lines
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
    from slavv import __version__
    from slavv.utils import get_system_info

    for line in load_info_lines(version=__version__, system_info=get_system_info()):
        print(line)


def _handle_run_command(args) -> None:
    """Execute the SLAVV processing pipeline."""
    from slavv import SLAVVProcessor
    from slavv.io import Network, load_tiff_volume
    from slavv.runtime import RunContext, build_status_lines, load_run_snapshot

    _require_existing_file(args.input, label="input file")
    _configure_logging(args.verbose, format_string=_DETAILED_LOG_FORMAT)

    logger.info("Loading volume from %s", args.input)
    image = load_tiff_volume(args.input)

    parameters = _build_pipeline_parameters(args)
    effective_run_dir = args.run_dir or (
        None if args.checkpoint_dir else os.path.join(args.output, "_slavv_run")
    )

    last_event_line = {"value": ""}

    def event_callback(event) -> None:
        line = (
            f"[{event.stage}] stage={event.stage_progress * 100:.1f}% "
            f"overall={event.overall_progress * 100:.1f}%"
        )
        if event.detail:
            line = f"{line} - {event.detail}"
        if line != last_event_line["value"]:
            print(line)
            last_event_line["value"] = line

    processor = SLAVVProcessor()
    results = processor.process_image(
        image,
        parameters,
        event_callback=event_callback,
        run_dir=effective_run_dir,
        checkpoint_dir=args.checkpoint_dir,
        stop_after=args.stop_after,
        force_rerun_from=args.force_rerun_from,
    )

    os.makedirs(args.output, exist_ok=True)
    vertices = results.get("vertices", {})
    edges = results.get("edges", {})
    pos = np.asarray(vertices.get("positions", []))
    rad = np.asarray(vertices.get("radii_microns", []))
    conn = np.atleast_2d(np.asarray(edges.get("connections", [])))

    network_obj = Network(
        vertices=pos if pos.size > 0 else np.empty((0, 3)),
        edges=conn if conn.size > 0 else np.empty((0, 2)),
        radii=rad if rad.size > 0 else None,
    )

    export_formats = _expand_export_formats(args.export)

    if export_formats and "vertices" not in results:
        logger.warning(
            "Export requested but pipeline stopped before extracting vertices. Skipping export."
        )
        export_formats = []
    elif export_formats and "network" not in results and "edges" not in results:
        logger.warning(
            "Export requested but pipeline stopped early. Formatting output with available data only."
        )

    for fmt in export_formats:
        save_network_export(
            fmt,
            output_dir=args.output,
            network_obj=network_obj,
            results=results,
        )

    snapshot = None
    if effective_run_dir:
        snapshot = load_run_snapshot(effective_run_dir)
        if snapshot is not None:
            context = RunContext.from_existing(effective_run_dir)
            context.update_optional_task(
                "exports",
                status="completed" if export_formats else "pending",
                detail="Exported requested output formats"
                if export_formats
                else "No exports requested",
                artifacts=_build_export_artifacts(args.output, export_formats),
            )
            snapshot = context.snapshot
    elif args.checkpoint_dir:
        snapshot = load_run_snapshot(args.checkpoint_dir)

    if effective_run_dir:
        print(f"Run directory: {effective_run_dir}")
    elif args.checkpoint_dir:
        print(f"Checkpoint directory: {args.checkpoint_dir}")
    if snapshot is not None:
        print()
        for line in build_status_lines(snapshot):
            print(line)

    print(f"Done. Results in {args.output}")


def _handle_import_matlab_command(args) -> None:
    """Import MATLAB batch output as Python checkpoints."""
    from slavv.io.matlab_bridge import import_matlab_batch
    from slavv.runtime import RunContext

    _configure_logging(args.verbose, format_string=_DETAILED_LOG_FORMAT)

    if written := import_matlab_batch(
        args.batch_folder,
        args.checkpoint_dir,
        stages=args.stages,
    ):
        print(f"Imported {len(written)} stage(s) into {args.checkpoint_dir}:")
        for stage, path in written.items():
            print(f"  {stage}: {path}")
        context = RunContext(
            checkpoint_dir=args.checkpoint_dir,
            target_stage="network",
            provenance={"source": "matlab_import"},
            legacy=True,
        )
        for stage, path in written.items():
            context.complete_stage(
                stage,
                detail="Imported from MATLAB batch output",
                artifacts={"checkpoint": path},
                resumed=True,
            )
        print()
        print("You can now run the Python pipeline with:")
        print(f"  slavv run -i <image.tif> --checkpoint-dir {args.checkpoint_dir}")
    else:
        print("No MATLAB data files found. Check that the batch folder path is correct.")


def _handle_status_command(args) -> None:
    """Render run status from a run directory or legacy checkpoint directory."""
    from slavv.runtime import build_status_lines, load_legacy_run_snapshot, load_run_snapshot

    _configure_logging(args.verbose, format_string=_SIMPLE_LOG_FORMAT)

    snapshot = load_status_snapshot(
        args.run_dir,
        snapshot_loader=load_run_snapshot,
        legacy_snapshot_loader=load_legacy_run_snapshot,
    )
    if snapshot is None:
        print(
            f"Error: no run snapshot or legacy checkpoints found in {args.run_dir}", file=sys.stderr
        )
        sys.exit(1)

    for line in build_status_output_lines(snapshot, status_line_builder=build_status_lines):
        print(line)


def _handle_parity_proof_command(args) -> None:
    """Render the latest maintained network-gate proof artifact summary."""
    from slavv.apps.parity_cli import print_latest_proof_summary

    _configure_logging(args.verbose, format_string=_SIMPLE_LOG_FORMAT)
    print_latest_proof_summary(Path(args.run_dir))


def _handle_analyze_command(args) -> None:
    """Analyze an exported network JSON file and print statistics."""
    from slavv.analysis import calculate_network_statistics

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
    from slavv.visualization.network_plots import NetworkVisualizer

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
