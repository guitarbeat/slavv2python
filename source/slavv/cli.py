"""
Command-line interface for SLAVV.

Usage:
    slavv run -i volume.tif -o results/ --export csv json
    slavv info
"""
import argparse
import logging
import os
import sys

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="slavv",
        description="SLAVV – Segmentation-Less, Automated, Vascular Vectorization",
    )
    parser.add_argument(
        "--version", action="store_true", help="Print version and exit"
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- slavv run --------------------------------------------------------
    run_parser = subparsers.add_parser(
        "run", help="Run the SLAVV pipeline on a TIFF volume"
    )
    run_parser.add_argument(
        "-i", "--input", required=True, help="Path to input TIFF file"
    )
    run_parser.add_argument(
        "-o", "--output", default="./slavv_output", help="Output directory (default: ./slavv_output)"
    )
    run_parser.add_argument(
        "--checkpoint-dir", default=None,
        help="Checkpoint directory for resume support"
    )
    run_parser.add_argument(
        "--energy-method", choices=["hessian", "frangi", "sato"],
        default="hessian", help="Energy computation method (default: hessian)"
    )
    run_parser.add_argument(
        "--edge-method", choices=["tracing", "watershed"],
        default="tracing", help="Edge extraction method (default: tracing)"
    )
    run_parser.add_argument(
        "--vessel-radius", type=float, default=1.5,
        help="Smallest vessel radius in microns (default: 1.5)"
    )
    run_parser.add_argument(
        "--microns-per-voxel", type=float, nargs=3, default=[1.0, 1.0, 1.0],
        metavar=("Y", "X", "Z"),
        help="Voxel size in microns (default: 1.0 1.0 1.0)"
    )
    run_parser.add_argument(
        "--export", nargs="+", choices=["csv", "json", "mat"],
        default=[], help="Export formats (can specify multiple)"
    )
    run_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )

    # --- slavv info -------------------------------------------------------
    subparsers.add_parser("info", help="Print version and system information")

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
    network = results.get("network", {})

    for fmt in args.export:
        if fmt == "csv":
            path = os.path.join(args.output, "network.csv")
            save_network_to_csv(network, path)
            logger.info("Saved CSV to %s", path)
        elif fmt == "json":
            path = os.path.join(args.output, "network.json")
            save_network_to_json(network, path)
            logger.info("Saved JSON to %s", path)
        elif fmt == "mat":
            try:
                from slavv.io import save_network_to_mat
                path = os.path.join(args.output, "network.mat")
                save_network_to_mat(network, path)
                logger.info("Saved MAT to %s", path)
            except ImportError:
                logger.warning("scipy required for MAT export; skipping")

    print(f"Done. Results in {args.output}")


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
    else:
        parser.print_help()
        sys.exit(0 if args.command is None else 1)


if __name__ == "__main__":
    main()
