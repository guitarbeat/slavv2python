"""Deterministic TIFF samples that enter the normal SLAVV processing path."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import numpy as np
import tifffile

from slavv_python.utils.synthetic import (
    generate_ladder_rung_volume,
    generate_synthetic_vessel_volume,
)


@dataclass(frozen=True)
class SampleTiff:
    """A generated TIFF payload and its user-facing provenance."""

    sample_id: str
    name: str
    description: str
    tiff_bytes: bytes
    shape_zyx: tuple[int, int, int]


SAMPLE_TIFF_OPTIONS = {
    "straight_vessel": "Straight vessel (32 x 64 x 64)",
    "y_junction_32": "Y-junction (32³)",
    "double_junction_32": "Double junction (32³)",
}


def _sample_volume(sample_id: str) -> tuple[np.ndarray, str]:
    if sample_id == "y_junction_32":
        return (
            generate_ladder_rung_volume(sample_id),
            "A central vessel with one branch; fast enough for an interactive full-pipeline run.",
        )
    if sample_id == "double_junction_32":
        return (
            generate_ladder_rung_volume(sample_id),
            "A central vessel with opposing branches at two depths.",
        )
    if sample_id == "straight_vessel":
        return (
            generate_synthetic_vessel_volume(
                shape=(32, 64, 64),
                vessel_radius=3.0,
            ),
            "A single straight vessel for a quick end-to-end processing check.",
        )
    known = ", ".join(SAMPLE_TIFF_OPTIONS)
    raise ValueError(f"Unknown sample TIFF {sample_id!r}; expected one of: {known}")


def build_sample_tiff(sample_id: str) -> SampleTiff:
    """Generate a real TIFF byte stream for the selected deterministic sample."""
    volume, description = _sample_volume(sample_id)
    buffer = BytesIO()
    tifffile.imwrite(buffer, volume.astype(np.float32), photometric="minisblack")
    return SampleTiff(
        sample_id=sample_id,
        name=f"slavv_sample_{sample_id}.tif",
        description=description,
        tiff_bytes=buffer.getvalue(),
        shape_zyx=tuple(int(value) for value in volume.shape),
    )


__all__ = ["SAMPLE_TIFF_OPTIONS", "SampleTiff", "build_sample_tiff"]
