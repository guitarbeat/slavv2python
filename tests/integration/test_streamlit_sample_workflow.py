"""Generated Streamlit samples must traverse the real SLAVV pipeline."""

from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING

from slavv_python.engine import SlavvPipeline
from slavv_python.interface.streamlit.state.sample_data import build_sample_tiff
from slavv_python.storage import load_tiff_volume

if TYPE_CHECKING:
    from pathlib import Path


def test_default_sample_runs_through_full_pipeline(tmp_path: Path) -> None:
    sample = build_sample_tiff("straight_vessel")
    image = load_tiff_volume(BytesIO(sample.tiff_bytes))
    results = SlavvPipeline().run(
        image,
        {
            "pipeline_profile": "paper",
            "radius_of_smallest_vessel_in_microns": 1.0,
            "radius_of_largest_vessel_in_microns": 5.0,
            "scales_per_octave": 1.0,
        },
        run_dir=str(tmp_path / "sample_run"),
    )

    assert len(results["vertices"]["positions"]) > 0
    assert len(results["edges"]["traces"]) > 0
    assert len(results["network"]["strands"]) > 0
