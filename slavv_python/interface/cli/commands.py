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


class Terminal:
    """Minimal ANSI escape sequence formatter for modernizing the CLI experience."""

    BOLD = "\033[1m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"

    @classmethod
    def header(cls, text: str) -> str:
        return f"\n{cls.BOLD}{cls.CYAN}=== {text} ==={cls.RESET}\n"

    @classmethod
    def success(cls, text: str) -> str:
        return f"{cls.GREEN}\u2714 {text}{cls.RESET}"

    @classmethod
    def error(cls, text: str) -> str:
        return f"{cls.RED}\u2718 {text}{cls.RESET}"

    @classmethod
    def warn(cls, text: str) -> str:
        return f"{cls.YELLOW}\u26a0 {text}{cls.RESET}"

    @classmethod
    def label(cls, text: str) -> str:
        return f"{cls.BOLD}{text}:{cls.RESET}"


def _handle_run_command(args) -> None:
    """Execute the SLAVV processing pipeline."""
    from slavv_python.engine.state import load_run_snapshot
    from slavv_python.storage import load_tiff_volume

    from ... import SlavvPipeline

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format=f"{Terminal.CYAN}%(asctime)s{Terminal.RESET} | %(message)s",
        datefmt="%H:%M:%S",
    )

    effective_run_dir = resolve_effective_run_dir(args.output, args.run_dir)
    parameters = _prepare_run_parameters(args)

    _require_existing_file(args.input)
    print(Terminal.header(f"Starting SLAVV Pipeline Execution: {args.input}"))

    image = load_tiff_volume(args.input)
    pipeline = SlavvPipeline()

    def _event_callback(event):
        print(f"  \u251c\u2500 {format_run_event_line(event)}")

    results = pipeline.run(
        image,
        parameters,
        event_callback=_event_callback,
        run_dir=effective_run_dir,
        stop_after=args.stop_after,
        force_rerun_from=args.force_rerun_from,
    )

    if not results or "network" not in results:
        print(Terminal.warn("Pipeline run finished (partial completion)."))
        return

    export_formats = filter_export_formats(args.export)
    if not export_formats:
        print(Terminal.success("Pipeline run completed. No exports requested."))
        return

    print(f"\n{Terminal.BOLD}Generating requested network exports...{Terminal.RESET}")
    export_artifact_paths = _resolve_export_artifact_paths(args.output, export_formats)
    update_run_export_task(effective_run_dir, export_artifact_paths)

    for fmt, path in export_artifact_paths.items():
        save_network_export(results, path, format=fmt)
        print(f"  \u251c\u2500 Saved: {path}")

    snapshot = load_run_snapshot(effective_run_dir)
    if snapshot is not None:
        lines = build_run_completion_lines(snapshot, export_artifact_paths)
    else:
        lines = [f"Run completed. Staged exports: {list(export_artifact_paths.keys())}"]

    print(Terminal.header("Execution Summary"))
    print("\n".join(f"  {line}" for line in lines))
    print(Terminal.success("Process completed successfully.\n"))


def _handle_analyze_command(args) -> None:
    """Calculate and print statistics for an exported network."""
    from slavv_python.analytics import calculate_network_statistics

    print(Terminal.header("Network Analysis"))
    print(f"Loading exported results from {Terminal.CYAN}{args.input}{Terminal.RESET}...\n")
    results = _load_exported_results(args.input)

    stats = calculate_exported_network_stats(
        results,
        statistics_fn=calculate_network_statistics,
    )

    print(f"{Terminal.BOLD}Results:{Terminal.RESET}")
    for label, value in stats:
        print(f"  {Terminal.label(label):<40} {value}")
    print("\n")


def _handle_status_command(args) -> None:
    """Inspect and print the status of a resumable run."""
    from slavv_python.engine.state import load_run_snapshot

    snapshot = load_run_snapshot(args.run_dir)
    if snapshot is None:
        print(Terminal.error(f"No valid SLAVV run metadata found under {args.run_dir}"))
        sys.exit(1)

    print(Terminal.header(f"Run Status: {args.run_dir}"))
    lines = build_run_status_lines(snapshot)
    print("\n".join(f"  {line}" for line in lines) + "\n")


def _handle_info_command(args) -> None:
    """Print version and system information."""
    from slavv_python.utils import get_system_info

    from ... import __version__
    from .info_service import load_info_lines

    system_info = get_system_info()
    lines = load_info_lines(version=__version__, system_info=system_info)

    print(Terminal.header("SLAVV System Information"))
    print("\n".join(f"  {line}" for line in lines) + "\n")


def _handle_plot_command(args) -> None:
    """Generate interactive plots for an exported network."""
    from ...visualization import NetworkVisualizer

    print(Terminal.header("Interactive Plot Generation"))
    print(f"Loading network from {Terminal.CYAN}{args.input}{Terminal.RESET}...")

    results = _load_exported_results(args.input)
    visualizer = NetworkVisualizer()

    print("Compiling interactive dashboard...")
    fig = visualizer.create_summary_dashboard(results)

    fig.write_html(args.output)
    print(Terminal.success(f"Saved interactive plots to {args.output}\n"))


__all__ = [
    "_handle_analyze_command",
    "_handle_info_command",
    "_handle_plot_command",
    "_handle_run_command",
    "_handle_status_command",
]
