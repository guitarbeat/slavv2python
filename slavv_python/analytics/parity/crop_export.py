"""Export the tier-M center crop from full 180709_E for parity pre-gate.

ROI definition: docs/reference/workflow/PARITY_PRE_GATE.md (64 x 256 x 256, Z x Y x X).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tifffile

# Tier M — center crop of the 64 x 512 x 512 (Z, Y, X) canonical volume
CROP_Z_SLICE = slice(0, 64)
CROP_Y_SLICE = slice(128, 384)
CROP_X_SLICE = slice(128, 384)
EXPECTED_FULL_SHAPE = (64, 512, 512)
CROP_SHAPE = (64, 256, 256)
DEFAULT_OUTPUT_NAME = "180709_E_crop_M.tif"

DEFAULT_SOURCE = (
    Path("workspace")
    / "datasets"
    / "771eb62fd1322cf59e24f056aff2692b3375b94ce6dc9b25744428d4dbf1e353"
    / "01_Input"
    / "180709_E.tif"
)


def export_crop_m(source: Path, output: Path) -> np.ndarray:
    """Load full volume, validate shape, crop, and write TIFF."""
    if not source.is_file():
        raise FileNotFoundError(f"source volume not found: {source}")

    volume = tifffile.imread(source)
    if volume.shape != EXPECTED_FULL_SHAPE:
        raise ValueError(
            f"expected full shape {EXPECTED_FULL_SHAPE} (Z,Y,X), "
            f"got {volume.shape} from {source}"
        )

    cropped = np.ascontiguousarray(volume[CROP_Z_SLICE, CROP_Y_SLICE, CROP_X_SLICE])
    if cropped.shape != CROP_SHAPE:
        raise ValueError(
            f"crop shape mismatch: expected {CROP_SHAPE}, got {cropped.shape}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(output, cropped)
    return cropped


def write_roi_metadata(output: Path) -> Path:
    """Write a .roi.json sidecar file alongside the TIFF."""
    meta_path = output.with_suffix(output.suffix + ".roi.json")
    payload = {
        "tier": "M",
        "dataset_id": "180709_E_crop_M",
        "axis_order": "ZYX",
        "full_shape": list(EXPECTED_FULL_SHAPE),
        "crop_shape": list(CROP_SHAPE),
        "slices": {
            "z": [CROP_Z_SLICE.start, CROP_Z_SLICE.stop],
            "y": [CROP_Y_SLICE.start, CROP_Y_SLICE.stop],
            "x": [CROP_X_SLICE.start, CROP_X_SLICE.stop],
        },
        "source_doc": "docs/reference/workflow/PARITY_PRE_GATE.md",
    }
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return meta_path


def build_export_crop_parser() -> argparse.ArgumentParser:
    """Build the argument parser for export-crop."""
    parser = argparse.ArgumentParser(
        description="Export 180709_E tier-M center crop TIFF for parity pre-gate."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Full 180709_E.tif (Z x Y x X).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("workspace/scratch/180709_E_crop_M") / DEFAULT_OUTPUT_NAME,
        help="Output TIFF path.",
    )
    parser.add_argument(
        "--write-metadata",
        action="store_true",
        help="Write <output>.roi.json with slice bounds next to the TIFF.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entrypoint for the export-crop command."""
    args = build_export_crop_parser().parse_args(argv)
    cropped = export_crop_m(
        args.source.expanduser().resolve(),
        args.output.expanduser().resolve(),
    )
    print(f"wrote {args.output} shape={cropped.shape} dtype={cropped.dtype}")
    if args.write_metadata:
        meta = write_roi_metadata(args.output.expanduser().resolve())
        print(f"wrote {meta}")
    return 0


__all__ = [
    "CROP_SHAPE",
    "CROP_X_SLICE",
    "CROP_Y_SLICE",
    "CROP_Z_SLICE",
    "DEFAULT_OUTPUT_NAME",
    "DEFAULT_SOURCE",
    "EXPECTED_FULL_SHAPE",
    "build_export_crop_parser",
    "export_crop_m",
    "main",
    "write_roi_metadata",
]
