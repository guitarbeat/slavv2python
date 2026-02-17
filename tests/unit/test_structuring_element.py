import pathlib
import sys

import numpy as np

# Add source path for imports
from slavv.core import SLAVVProcessor


def test_anisotropic_structuring_element():
    core = SLAVVProcessor()
    strel_iso = core._spherical_structuring_element(1, np.array([1.0, 1.0, 1.0]))
    strel_aniso = core._spherical_structuring_element(1, np.array([1.0, 1.0, 2.0]))
    assert strel_iso.shape == (3, 3, 3)
    assert strel_iso[1, 1, 2]
    assert not strel_aniso[1, 1, 2]
