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

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
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

    return parser


def _args_to_parameters(args: argparse.Namespace) -> dict:
    """Convert parsed CLI arguments to a SLAVV parameters dict."""
    return {
        "energy_method": args.energy_method,
        "edge_method": args.edge_method,
        "radius_of_smallest_vessel_in_microns": args.vessel_radius,
        "microns_per_voxel": list(args.microns_per_voxel),
    }


def _cmd_info() -> None:
    """Print version and system information."""
    from slavv import __version__
    from slavv.utils import get_system_info

    print(f"slavv {__version__}")
    print()

    info = get_system_info()
    for key, value in info.items():
        print(f"  {key}: {value}")


def _cmd_run(args: argparse.Namespace) -> None:
    """Execute the SLAVV processing pipeline."""
    from slavv import SLAVVProcessor
    from slavv.io import load_tiff_volume, save_network_to_csv, save_network_to_json

    # Validate input
    if not os.path.isfile(args.input):
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    # Load volume
    logger.info("Loading volume from %s", args.input)
    image = load_tiff_volume(args.input)

    # Build parameters
    parameters = _args_to_parameters(args)

    # Run pipeline
    processor = SLAVVProcessor()
    results = processor.process_image(
        image,
        parameters,
        checkpoint_dir=args.checkpoint_dir,
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

    export_formats = args.export
    if "all" in export_formats:
        export_formats = ["csv", "json", "casx", "vmv", "mat"]

    for fmt in export_formats:
        if fmt == "csv":
            from slavv.io import save_network_to_csv

            path = os.path.join(args.output, "network.csv")
            save_network_to_csv(network_obj, path)
            logger.info("Saved CSV to %s", path)
        elif fmt == "json":
            from slavv.io import save_network_to_json

            path = os.path.join(args.output, "network.json")
            save_network_to_json(network_obj, path)
            logger.info("Saved JSON to %s", path)
        elif fmt == "casx":
            from slavv.io import save_network_to_casx

            path = os.path.join(args.output, "network.casx")
            save_network_to_casx(network_obj, path)
            logger.info("Saved CASX to %s", path)
        elif fmt == "vmv":
            from slavv.io import save_network_to_vmv

            path = os.path.join(args.output, "network.vmv")
            save_network_to_vmv(network_obj, path)
            logger.info("Saved VMV to %s", path)
        elif fmt == "mat":
            try:
                from slavv.visualization import NetworkVisualizer

                path = os.path.join(args.output, "network.mat")
                # Using Visualizer's internal method for .mat because it packages up the dict structure.
                vis = NetworkVisualizer()
                vis._export_mat(results, path)
                logger.info("Saved MAT to %s", path)
            except ImportError as e:
                logger.warning("Error saving MAT file: %s", e)

    print(f"Done. Results in {args.output}")


def _cmd_import_matlab(args: argparse.Namespace) -> None:
    """Import MATLAB batch output as Python checkpoints."""
    from slavv.io.matlab_bridge import import_matlab_batch

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    written = import_matlab_batch(
        args.batch_folder,
        args.checkpoint_dir,
        stages=args.stages,
    )

    if written:
        print(f"Imported {len(written)} stage(s) into {args.checkpoint_dir}:")
        for stage, path in written.items():
            print(f"  {stage}: {path}")
        print()
        print("You can now run the Python pipeline with:")
        print(f"  slavv run -i <image.tif> --checkpoint-dir {args.checkpoint_dir}")
    else:
        print("No MATLAB data files found. Check that the batch folder path is correct.")


def _load_dict_from_json(path: str) -> dict:
    import json

    with open(path, "r") as f:
        data = json.load(f)

    # Reconstruct the structure that calculate_network_statistics expects, which is the internal representation.
    # calculate_network_statistics expects `vertices` with `positions`, `edges` with `connections` and `traces`,
    # but `extract_geometric_features` will recalculate lengths if traces don't exist.
    # network.json doesn't export traces.

    # Actually, calculate_network_statistics is fine without traces, it will compute euclidean distance.
    vertices = data.get("vertices", {})
    edges = data.get("edges", {})

    return {
        "vertices": {
            "positions": np.array(vertices.get("positions", [])),
            "radii_microns": np.array(vertices.get("radii_microns", [])),
        },
        "edges": {"connections": np.array(edges.get("connections", []))},
    }


def _cmd_analyze(args: argparse.Namespace) -> None:
    """Analyze an exported network JSON file and print statistics."""
    import json
    from slavv.analysis.statistics import calculate_network_statistics

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(message)s")

    if not os.path.isfile(args.input):
        print(f"Error: file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    logger.info("Loading network from %s", args.input)
    results = _load_dict_from_json(args.input)

    logger.info("Calculating statistics...")
    stats = calculate_network_statistics(results)

    print("\n--- Network Statistics ---\n")
    print("Topological Features:")
    topo = stats.get("topological", {})
    for k, v in topo.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    print("\nGeometric Features (Aggregates):")
    geom = stats.get("geometric", {})
    if "edge_lengths_microns" in geom:
        lengths = geom["edge_lengths_microns"]
        print(f"  Total Edge Length: {np.sum(lengths):.2f} um")
        print(f"  Mean Edge Length: {np.mean(lengths):.2f} um")
    if "edge_radii_microns" in geom:
        radii = geom["edge_radii_microns"]
        print(f"  Mean Edge Radius: {np.mean(radii):.2f} um")


def _cmd_plot(args: argparse.Namespace) -> None:
    """Generate interactive plots from exported network JSON."""
    from slavv.visualization.network_plots import NetworkVisualizer

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(message)s")

    if not os.path.isfile(args.input):
        print(f"Error: file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    logger.info("Loading network from %s", args.input)
    results = _load_dict_from_json(args.input)

    vis = NetworkVisualizer()
    logger.info("Generating length-weighted histograms...")

    # We're passing results which lacks 'traces'. The visualizer `plot_length_weighted_histograms`
    # computes euclidean distance if traces are missing.
    fig = vis.plot_length_weighted_histograms(results, number_of_bins=args.number_of_bins)

    fig.write_html(args.output)
    print(f"Saved interactive plots to {args.output}")


def main(argv=None):
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.version:
        from slavv import __version__

        print(f"slavv {__version__}")
        return

    if args.command == "info":
        _cmd_info()
    elif args.command == "run":
        _cmd_run(args)
    elif args.command == "import-matlab":
        _cmd_import_matlab(args)
    elif args.command == "analyze":
        import numpy as np

        _cmd_analyze(args)
    elif args.command == "plot":
        import numpy as np

        _cmd_plot(args)
    else:
        parser.print_help()
        sys.exit(0 if args.command is None else 1)


if __name__ == "__main__":
    main()
