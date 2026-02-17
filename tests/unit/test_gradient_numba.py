import pathlib
import sys

import numpy as np

<<<<<<< HEAD
from slavv.core import SLAVVProcessor
=======
from slavv.core.pipeline import SLAVVProcessor
>>>>>>> 3e500a60f45114343cdca16b13c837e3d0f1d578

def test_compute_gradient_linear_field():
    proc = SLAVVProcessor()
    # Create linear energy field: 2*y + 3*x + 4*z
    y, x, z = np.indices((5, 5, 5))
    energy = 2 * y + 3 * x + 4 * z
    pos = np.array([2.0, 2.0, 2.0])
    grad = proc._compute_gradient(energy, pos, np.array([1.0, 1.0, 1.0]))
    assert np.allclose(grad, np.array([2.0, 3.0, 4.0]))
