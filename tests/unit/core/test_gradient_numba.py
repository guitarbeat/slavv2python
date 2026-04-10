import numpy as np

from slavv.core.edge_primitives import compute_gradient


def test_compute_gradient_linear_field():
    # Create linear energy field: 2*y + 3*x + 4*z
    y, x, z = np.indices((5, 5, 5))
    energy = 2 * y + 3 * x + 4 * z
    pos = np.array([2.0, 2.0, 2.0])
    grad = compute_gradient(energy, pos, np.array([1.0, 1.0, 1.0]))
    assert np.allclose(grad, np.array([2.0, 3.0, 4.0]))
