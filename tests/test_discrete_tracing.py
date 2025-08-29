import pathlib
import sys

import numpy as np

# Add source path for imports
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / 'slavv-streamlit' / 'src'))

from vectorization_core import SLAVVProcessor


def test_discrete_tracing_steps_snap_to_voxels():
    proc = SLAVVProcessor()
    # Energy decreases along +x (axis 1) so tracing proceeds without energy rise.
    energy = np.tile(-np.arange(5)[None, :, None], (5, 1, 5)).astype(float)
    start_pos = np.array([2.0, 0.0, 2.0])  # (y, x, z)
    direction = np.array([0.0, 1.0, 0.0])
    trace = proc._trace_edge(
        energy,
        start_pos,
        direction,
        step_size=1.0,
        max_energy=0.0,
        vertex_positions=np.empty((0, 3)),
        vertex_scales=np.empty((0,), dtype=int),
        lumen_radius_pixels=np.array([1.0]),
        lumen_radius_microns=np.array([1.0]),
        max_steps=4,
        microns_per_voxel=np.ones(3),
        energy_sign=-1.0,
        discrete_steps=True,
    )
    coords = np.asarray(trace)
    assert np.allclose(coords, np.round(coords))
