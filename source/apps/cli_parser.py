"""Argument parser construction for the SLAVV CLI."""

from __future__ import annotations

import argparse

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
        "--energy-storage-format",
        choices=["auto", "npy", "zarr"],
        default="auto",
        help=(
            "Storage format for resumable energy arrays (default: auto). "
            "'zarr' is useful for larger persisted energy volumes."
        ),
    )
    run_parser.add_argument(
        "--energy-method",
        choices=["hessian", "frangi", "sato", "simpleitk_objectness", "cupy_hessian"],
        default="hessian",
        help=(
            "Energy computation method (default: hessian). "
            "'simpleitk_objectness' is experimental and spacing-aware; "
            "'cupy_hessian' is experimental and requires an NVIDIA GPU."
        ),
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

    analyze_parser = subparsers.add_parser("analyze", help="Analyze an exported network JSON file")
    analyze_parser.add_argument("-i", "--input", required=True, help="Path to input network.json")
    analyze_parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")

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

    subparsers.add_parser("info", help="Print version and system information")

    status_parser = subparsers.add_parser("status", help="Inspect the status of a resumable run")
    status_parser.add_argument(
        "--run-dir",
        required=True,
        help="Run directory containing run metadata",
    )
    status_parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    return parser
