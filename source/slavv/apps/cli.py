"""
Command-line interface for SLAVV.

Usage:
    slavv run -i volume.tif -o results/ --export csv json
    slavv info
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import cast

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)
_DETAILED_LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
_SIMPLE_LOG_FORMAT = "%(asctime)s %(message)s"
_EXPORT_FILE_NAMES = {
    "csv": "network.csv",
    "json": "network.json",
    "casx": "network.casx",
    "vmv": "network.vmv",
    "mat": "network.mat",
}


def _build_cli_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="slavv",
        description="SLAVV - Segmentation-Less, Automated, Vascular Vectorization",
    )
    parser.add_argument("--version", action="store_true", help="Print version and exit")
    subparsers = parser.add_subparsers(dest="command")

    # --- slavv run --------------------------------------------------------
    run_parser = subparsers.add_parser("run", help="Run the SLAVV pipeline on a TIFF volume")
    run_parser.add_argument("-i", "--input", required=True, help="Path to input TIFF file")
    run_parser.add_argument(
        "-o",
        "--output",
        default="./slavv_output",
        help="Output directory (default: ./slavv_output)",
    )
    run_parser.add_argument(
        "--run-dir",
        default=None,
        help="Structured run directory for resumable status tracking",
    )
    run_parser.add_argument(
        "--checkpoint-dir", default=None, help="Checkpoint directory for resume support"
    )
    run_parser.add_argument(
        "--energy-method",
        choices=["hessian", "frangi", "sato"],
        default="hessian",
        help="Energy computation method (default: hessian)",
    )
    run_parser.add_argument(
        "--edge-method",
        choices=["tracing", "watershed"],
        default="tracing",
        help="Edge extraction method (default: tracing)",
    )
    run_parser.add_argument(
        "--vessel-radius",
        type=float,
        default=1.5,
        help="Smallest vessel radius in microns (default: 1.5)",
    )
    run_parser.add_argument(
        "--microns-per-voxel",
        type=float,
        nargs=3,
        default=[1.0, 1.0, 1.0],
        metavar=("Y", "X", "Z"),
        help="Voxel size in microns (default: 1.0 1.0 1.0)",
    )
    run_parser.add_argument(
        "--export",
        nargs="+",
        choices=["csv", "json", "mat", "casx", "vmv", "all"],
        default=[],
        help="Export formats (can specify multiple)",
    )
    run_parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")

    run_parser.add_argument(
        "--stop-after",
        choices=["energy", "vertices", "edges", "network"],
        default=None,
        help="Stop the pipeline early after completing this stage.",
    )
    run_parser.add_argument(
        "--force-rerun-from",
        choices=["energy", "vertices", "edges", "network"],
        default=None,
        help="Ignore checkpoints and force recalculation starting from this stage.",
    )

    # --- slavv analyze ----------------------------------------------------
    analyze_parser = subparsers.add_parser("analyze", help="Analyze an exported network JSON file")
    analyze_parser.add_argument("-i", "--input", required=True, help="Path to input network.json")
    analyze_parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")

    # --- slavv plot -------------------------------------------------------
    plot_parser = subparsers.add_parser(
        "plot", help="Generate plots from an exported network JSON file"
    )
    plot_parser.add_argument("-i", "--input", required=True, help="Path to input network.json")
    plot_parser.add_argument(
        "-o", "--output", default="plots.html", help="Output HTML file path (default: plots.html)"
    )
    plot_parser.add_argument(
        "--number-of-bins", type=int, default=50, help="Number of bins for histograms"
    )
    plot_parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")

    # --- slavv info -------------------------------------------------------
    subparsers.add_parser("info", help="Print version and system information")

    # --- slavv import-matlab ----------------------------------------------
    imp_parser = subparsers.add_parser(
        "import-matlab", help="Import a MATLAB batch_* folder as Python checkpoints"
    )
    imp_parser.add_argument(
        "-b",
        "--batch-folder",
        required=True,
        help="Path to the MATLAB batch_* folder (or parent directory)",
    )
    imp_parser.add_argument(
        "-c",
        "--checkpoint-dir",
        required=True,
        help="Output directory for Python checkpoint pickles",
    )
    imp_parser.add_argument(
        "--stages",
        nargs="+",
        choices=["energy", "vertices", "edges", "network"],
        default=None,
        help="Which stages to import (default: all available)",
    )
    imp_parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")

    # --- slavv status ------------------------------------------------------
    status_parser = subparsers.add_parser("status", help="Inspect the status of a resumable run")
    status_parser.add_argument(
        "--run-dir",
        required=True,
        help="Run directory or legacy checkpoint directory containing run metadata",
    )
    status_parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")

    return parser


def _build_pipeline_parameters(args: argparse.Namespace) -> dict:
    """Convert parsed CLI arguments to a SLAVV parameters dict."""
    return {
        "energy_method": args.energy_method,
        "edge_method": args.edge_method,
        "radius_of_smallest_vessel_in_microns": args.vessel_radius,
        "microns_per_voxel": list(args.microns_per_voxel),
    }


def _handle_info_command() -> None:
    """Print version and system information."""
    from slavv import __version__
    from slavv.utils import get_system_info

    print(f"slavv {__version__}")
    print()

    info = get_system_info()
    for key, value in info.items():
        print(f"  {key}: {value}")


def _configure_logging(verbose: bool, *, format_string: str) -> None:
    """Configure command logging with the requested verbosity."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format=format_string)


def _require_existing_file(path: str, *, label: str = "file") -> None:
    """Exit with a consistent error if a required file path is missing."""
    if not os.path.isfile(path):
        print(f"Error: {label} not found: {path}", file=sys.stderr)
        sys.exit(1)


def _expand_export_formats(export_formats: list[str]) -> list[str]:
    """Normalize CLI export selections into concrete formats."""
    return ["csv", "json", "casx", "vmv", "mat"] if "all" in export_formats else export_formats


def _save_network_export(
    format_type: str,
    *,
    output_dir: str,
    network_obj,
    results: dict,
) -> str | None:
    """Persist one export format and return the written path when successful."""
    export_path = os.path.join(output_dir, _EXPORT_FILE_NAMES[format_type])

    if format_type == "mat":
        try:
            from slavv.visualization import NetworkVisualizer

            vis = NetworkVisualizer()
            vis._export_mat(
                results.get("vertices", {}),
                results.get("edges", {}),
                results.get("network", {}),
                results.get("parameters", {}),
                export_path,
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


def _build_export_artifacts(output_dir: str, export_formats: list[str]) -> dict[str, str]:
    """Build run-state artifact paths for requested exports."""
    return {
        fmt: os.path.join(output_dir, _EXPORT_FILE_NAMES[fmt])
        for fmt in export_formats
        if fmt != "csv"
    }


def _load_exported_results(input_path: str) -> dict:
    """Validate and load exported JSON results for analyze/plot commands."""
    _require_existing_file(input_path)
    logger.info("Loading network from %s", input_path)
    return _load_exported_network_json(input_path)


def _handle_run_command(args: argparse.Namespace) -> None:
    """Execute the SLAVV processing pipeline."""
    from slavv import SLAVVProcessor
    from slavv.io import load_tiff_volume
    from slavv.runtime import RunContext, build_status_lines, load_run_snapshot

    # Validate input
    _require_existing_file(args.input, label="input file")

    # Setup logging
    _configure_logging(args.verbose, format_string=_DETAILED_LOG_FORMAT)

    # Load volume
    logger.info("Loading volume from %s", args.input)
    image = load_tiff_volume(args.input)

    # Build parameters
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

    # Run pipeline
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

    # Save outputs
    os.makedirs(args.output, exist_ok=True)

    # We need to construct a Network dataclass object for the exporters
    # instead of passing the raw dictionary.
    from slavv.io import Network

    vertices = results.get("vertices", {})
    edges = results.get("edges", {})

    # Convert lists/arrays to properly formed numpy arrays for Network
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
        _save_network_export(
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


def _handle_import_matlab_command(args: argparse.Namespace) -> None:
    """Import MATLAB batch output as Python checkpoints."""
    from slavv.io.matlab_bridge import import_matlab_batch
    from slavv.runtime import RunContext

    _configure_logging(args.verbose, format_string=_DETAILED_LOG_FORMAT)

    written = import_matlab_batch(
        args.batch_folder,
        args.checkpoint_dir,
        stages=args.stages,
    )

    if written:
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


def _handle_status_command(args: argparse.Namespace) -> None:
    """Render run status from a run directory or legacy checkpoint directory."""
    from slavv.runtime import RunContext, build_status_lines, load_run_snapshot

    _configure_logging(args.verbose, format_string=_SIMPLE_LOG_FORMAT)

    snapshot = load_run_snapshot(args.run_dir)
    if snapshot is None:
        legacy_context = RunContext(
            checkpoint_dir=args.run_dir,
            target_stage="network",
            legacy=True,
        )
        snapshot = legacy_context.snapshot

    for line in build_status_lines(snapshot):
        print(line)


def _normalize_exported_edge_connections(raw_connections: object) -> np.ndarray:
    """Normalize exported edge connections into a 2-column integer array."""
    connections = np.asarray(raw_connections, dtype=int)
    if connections.size == 0:
        return np.empty((0, 2), dtype=int)
    connections = np.atleast_2d(connections)
    if connections.shape[1] < 2:
        return np.empty((0, 2), dtype=int)
    return cast("np.ndarray", np.asarray(connections[:, :2], dtype=int))


def _build_strands_from_edge_connections(
    edge_connections: np.ndarray, *, vertex_count: int
) -> list[list[int]]:
    """Reconstruct strand-like paths from exported edge connections."""
    if vertex_count == 0 or edge_connections.size == 0:
        return []

    graph = nx.Graph()
    graph.add_nodes_from(range(vertex_count))
    for origin_idx, destination_idx in edge_connections:
        origin = int(origin_idx)
        destination = int(destination_idx)
        if origin == destination:
            continue
        if not (0 <= origin < vertex_count and 0 <= destination < vertex_count):
            continue
        graph.add_edge(origin, destination)

    strands: list[list[int]] = []
    visited_edges: set[tuple[int, int]] = set()

    for origin, destination in graph.edges():
        edge = tuple(sorted((origin, destination)))
        if edge in visited_edges:
            continue

        strand = [origin, destination]
        visited_edges.add(edge)

        current = destination
        while graph.degree(current) == 2:
            neighbors = list(graph.neighbors(current))
            next_node = neighbors[0] if neighbors[1] == strand[-2] else neighbors[1]
            next_edge = tuple(sorted((current, next_node)))
            if next_edge in visited_edges:
                break
            strand.append(next_node)
            visited_edges.add(next_edge)
            current = next_node

        current = origin
        while graph.degree(current) == 2:
            neighbors = list(graph.neighbors(current))
            next_node = neighbors[0] if neighbors[1] == strand[1] else neighbors[1]
            next_edge = tuple(sorted((current, next_node)))
            if next_edge in visited_edges:
                break
            strand.insert(0, next_node)
            visited_edges.add(next_edge)
            current = next_node

        strands.append(strand)

    return strands


def _infer_image_shape_from_vertices(vertex_positions: np.ndarray) -> tuple[int, int, int]:
    """Infer a minimal positive image shape from exported vertex positions."""
    if vertex_positions.size == 0:
        return (1, 1, 1)
    maxima = np.max(vertex_positions, axis=0)
    inferred_axes = [max(1, int(np.ceil(float(axis_max))) + 1) for axis_max in maxima[:3]]
    while len(inferred_axes) < 3:
        inferred_axes.append(1)
    return (inferred_axes[0], inferred_axes[1], inferred_axes[2])


def _load_exported_network_json(path: str) -> dict:
    """Load exported JSON and rebuild the stats inputs expected by analysis helpers."""
    import json

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    vertices = data.get("vertices", {})
    edges = data.get("edges", {})
    vertex_positions = np.asarray(vertices.get("positions", []), dtype=float)
    edge_connections = _normalize_exported_edge_connections(edges.get("connections", []))
    vertex_radii = np.asarray(vertices.get("radii_microns", []), dtype=float)
    if len(vertex_radii) != len(vertex_positions):
        vertex_radii = np.zeros(len(vertex_positions), dtype=float)

    strands = _build_strands_from_edge_connections(
        edge_connections,
        vertex_count=len(vertex_positions),
    )
    graph = nx.Graph()
    graph.add_nodes_from(range(len(vertex_positions)))
    graph.add_edges_from(edge_connections.tolist())
    bifurcations = np.fromiter(
        (node for node, degree in graph.degree() if degree > 2),
        dtype=int,
    )

    return {
        "vertices": {
            "positions": vertex_positions,
            "radii_microns": vertex_radii,
        },
        "edges": {"connections": edge_connections},
        "network": {
            "strands": strands,
            "bifurcations": bifurcations,
        },
        "parameters": data.get("parameters", {}),
        "image_shape": tuple(data.get("image_shape", _infer_image_shape_from_vertices(vertex_positions))),
    }


def _handle_analyze_command(args: argparse.Namespace) -> None:
    """Analyze an exported network JSON file and print statistics."""
    from slavv.analysis import calculate_network_statistics

    _configure_logging(args.verbose, format_string=_SIMPLE_LOG_FORMAT)
    results = _load_exported_results(args.input)

    logger.info("Calculating statistics...")
    stats = calculate_network_statistics(
        results["network"]["strands"],
        results["network"]["bifurcations"],
        results["vertices"]["positions"],
        results["vertices"]["radii_microns"],
        results["parameters"].get("microns_per_voxel", [1.0, 1.0, 1.0]),
        results["image_shape"],
    )

    topological_metrics = (
        ("Vertices", stats.get("num_vertices", 0)),
        ("Edges", stats.get("num_edges", 0)),
        ("Strands", stats.get("num_strands", 0)),
        ("Bifurcations", stats.get("num_bifurcations", 0)),
        ("Connected Components", stats.get("num_connected_components", 0)),
        ("Endpoints", stats.get("num_endpoints", 0)),
        ("Mean Degree", stats.get("mean_degree", 0.0)),
        ("Clustering Coefficient", stats.get("clustering_coefficient", 0.0)),
    )
    geometric_metrics = (
        ("Total Edge Length", f"{float(stats.get('total_length', 0.0)):.2f} um"),
        ("Mean Strand Length", f"{float(stats.get('mean_strand_length', 0.0)):.2f} um"),
        ("Mean Edge Length", f"{float(stats.get('mean_edge_length', 0.0)):.2f} um"),
        ("Mean Edge Radius", f"{float(stats.get('mean_edge_radius', 0.0)):.2f} um"),
        ("Mean Radius", f"{float(stats.get('mean_radius', 0.0)):.2f} um"),
        ("Volume Fraction", f"{float(stats.get('volume_fraction', 0.0)):.4f}"),
        ("Bifurcation Density", f"{float(stats.get('bifurcation_density', 0.0)):.2f} /mm^3"),
    )

    print("\n--- Network Statistics ---\n")
    print("Topological Features:")
    for label, value in topological_metrics:
        if isinstance(value, float):
            print(f"  {label}: {value:.4f}")
        else:
            print(f"  {label}: {value}")

    print("\nGeometric Features (Aggregates):")
    for label, value in geometric_metrics:
        print(f"  {label}: {value}")


def _handle_plot_command(args: argparse.Namespace) -> None:
    """Generate interactive plots from exported network JSON."""
    from slavv.visualization.network_plots import NetworkVisualizer

    _configure_logging(args.verbose, format_string=_SIMPLE_LOG_FORMAT)
    results = _load_exported_results(args.input)

    vis = NetworkVisualizer()
    logger.info("Generating length-weighted histograms...")

    # We're passing results which lacks 'traces'. The visualizer `plot_length_weighted_histograms`
    # computes euclidean distance if traces are missing.
    fig = vis.plot_length_weighted_histograms(
        results.get("vertices", {}),
        results.get("edges", {}),
        results.get("parameters", {}),
        number_of_bins=args.number_of_bins,
    )

    fig.write_html(args.output)
    print(f"Saved interactive plots to {args.output}")


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
    elif args.command == "analyze":
        _handle_analyze_command(args)
    elif args.command == "plot":
        _handle_plot_command(args)
    else:
        parser.print_help()
        sys.exit(0 if args.command is None else 1)


if __name__ == "__main__":
    main()
