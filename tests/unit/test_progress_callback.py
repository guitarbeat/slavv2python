import pathlib
import sys
import numpy as np

# Add source path for imports
from slavv.core.pipeline import SLAVVProcessor


def test_process_image_reports_progress():
    processor = SLAVVProcessor()
    image = np.zeros((5, 5, 5), dtype=np.float32)
    image[2, 2, 2] = 1.0
    params = {
        'radius_of_smallest_vessel_in_microns': 1.0,
        'radius_of_largest_vessel_in_microns': 2.0,
        'scales_per_octave': 1.0,
        'microns_per_voxel': [1.0, 1.0, 1.0],
    }
    calls = []

    def cb(fraction, stage):
        calls.append((fraction, stage))

    processor.process_image(image, params, progress_callback=cb)

    stages = [s for _, s in calls]
    assert stages == ['start', 'preprocess', 'energy', 'vertices', 'edges', 'network']
    assert calls[0][0] == 0.0
    assert calls[-1][0] == 1.0
