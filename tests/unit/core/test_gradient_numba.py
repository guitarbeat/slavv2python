import numpy as np

from slavv_python.core import energy as energy_module
from slavv_python.core.edge_primitives import compute_gradient


def test_compute_gradient_linear_field():
    # Create linear energy field: 2*y + 3*x + 4*z
    y, x, z = np.indices((5, 5, 5))
    energy = 2 * y + 3 * x + 4 * z
    pos = np.array([2.0, 2.0, 2.0])
    grad = compute_gradient(energy, pos, np.array([1.0, 1.0, 1.0]))
    assert np.allclose(grad, np.array([2.0, 3.0, 4.0]))


def test_compute_gradient_fast_linear_field():
    y, x, z = np.indices((5, 5, 5))
    energy = 2 * y + 3 * x + 4 * z
    inv_mpv_2x = np.array([0.5, 0.5, 0.5], dtype=np.float64)

    grad = energy_module.compute_gradient_fast(energy.astype(np.float64), 2, 2, 2, inv_mpv_2x)

    assert np.allclose(grad, np.array([2.0, 3.0, 4.0]))


def test_numba_acceleration_flag_matches_gradient_helper_shape():
    assert isinstance(energy_module.is_numba_acceleration_enabled(), bool)
    if energy_module.is_numba_acceleration_enabled():
        assert callable(energy_module.compute_gradient_impl)
        assert callable(energy_module.compute_gradient_fast)
