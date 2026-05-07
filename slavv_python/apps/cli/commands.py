"""Command handlers for the SLAVV CLI."""

from __future__ import annotations

import logging
import sys

from .analysis_service import calculate_exported_network_stats
from .export_service import save_network_export
from .exported_network import _load_exported_results
from .run_service import (
    build_run_completion_lines,
    filter_export_formats,
    format_run_event_line,
    resolve_effective_run_dir,
    update_run_export_task,
)
from .shared import (
    _prepare_run_parameters,
    _require_existing_file,
    _resolve_export_artifact_paths,
)
from .status_service import build_run_status_lines

_SIMPLE_LOG_FORMAT = "%(levelname)s:%(name)s:%(message)s"


def _handle_run_command(args) -> None:
    """Execute the SLAVV processing pipeline."""
    from ... import SlavvPipeline
    from ...io import load_tiff_volume
    from ...runtime import load_run_snapshot

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    logger = logging.getLogger("slavv_python.cli.run")

    effective_run_dir = resolve_effective_run_dir(args.output, args.run_dir)
    parameters = _prepare_run_parameters(args)

    _require_existing_file(args.input)
    image = load_tiff_volume(args.input)
    pipeline = SlavvPipeline()

    def _event_callback(event):
        print(format_run_event_line(event))

    results = pipeline.run(
        image,
        parameters,
        event_callback=_event_callback,
        run_dir=effective_run_dir,
        stop_after=args.stop_after,
        force_rerun_from=args.force_rerun_from,
    )

    if not results or "network" not in results:
        logger.info("Pipeline run finished (partial).")
        return

    # Post-run export phase
    export_formats = filter_export_formats(args.export)
    if not export_formats:
        logger.info("Pipeline run completed. No additional exports requested.")
        return

    logger.info("Generating requested network exports...")
    export_artifact_paths = _resolve_export_artifact_paths(args.output, export_formats)

    update_run_export_task(effective_run_dir, export_artifact_paths)

    for fmt, path in export_artifact_paths.items():
        save_network_export(results, path, format=fmt)

    # Use the structured runtime tracking for final reporting when available
    snapshot = load_run_snapshot(effective_run_dir)
    if snapshot is not None:
        lines = build_run_completion_lines(snapshot, export_artifact_paths)
    else:
        # Fallback to simple report
        lines = [f"Run completed. Staged exports: {list(export_artifact_paths.keys())}"]

    print("\n" + "\n".join(lines))


def _handle_analyze_command(args) -> None:
    """Calculate and print statistics for an exported network."""
    from ...analysis import calculate_network_statistics

    logging.basicConfig(level=logging.INFO, format=_SIMPLE_LOG_FORMAT)
    logger = logging.getLogger("slavv_python.cli.analyze")

    results = _load_exported_results(args.input)

    logger.info("Calculating statistics...")
    stats = calculate_exported_network_stats(
        results,
        statistics_fn=calculate_network_statistics,
    )

    print(f"\nNetwork Analysis Results: {args.input}")
    print("-" * (26 + len(args.input)))
    for label, value in stats:
        print(f"{label:30} {value}")
    print("")


def _handle_status_command(args) -> None:
    """Inspect and print the status of a resumable run."""
    from ...runtime import load_run_snapshot

    logging.basicConfig(level=logging.INFO, format=_SIMPLE_LOG_FORMAT)
    logger = logging.getLogger("slavv_python.cli.status")

    snapshot = load_run_snapshot(args.run_dir)
    if snapshot is None:
        logger.error(f"No valid SLAVV run metadata found under {args.run_dir}")
        sys.exit(1)

    lines = build_run_status_lines(snapshot)
    print("\n" + "\n".join(lines) + "\n")


def _handle_info_command(args) -> None:
    """Print version and system information."""
    from ... import __version__
    from ...utils import get_system_info
    from .info_service import load_info_lines

    system_info = get_system_info()
    lines = load_info_lines(version=__version__, system_info=system_info)
    print("\n" + "\n".join(lines) + "\n")


def _handle_plot_command(args) -> None:
    """Generate interactive plots for an exported network."""
    from ...visualization import NetworkVisualizer

    logging.basicConfig(level=logging.INFO, format=_SIMPLE_LOG_FORMAT)
    logger = logging.getLogger("slavv_python.cli.plot")

    results = _load_exported_results(args.input)
    visualizer = NetworkVisualizer()

    logger.info("Compiling interactive plots...")
    fig = visualizer.create_summary_dashboard(results)

    fig.write_html(args.output)
    print(f"Saved interactive plots to {args.output}")


__all__ = [
    "_handle_analyze_command",
    "_handle_info_command",
    "_handle_plot_command",
    "_handle_run_command",
    "_handle_status_command",
]
