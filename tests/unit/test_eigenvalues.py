
import numpy as np
import pytest
from slavv.core.energy import solve_symmetric_eigenvalues_3x3

def test_solve_symmetric_eigenvalues_3x3_random():
    """Test against numpy.linalg.eigvalsh with random matrices."""
    np.random.seed(42)
    shape = (10, 10, 10)

    Hxx = np.random.randn(*shape).astype(np.float32)
    Hxy = np.random.randn(*shape).astype(np.float32)
    Hxz = np.random.randn(*shape).astype(np.float32)
    Hyy = np.random.randn(*shape).astype(np.float32)
    Hyz = np.random.randn(*shape).astype(np.float32)
    Hzz = np.random.randn(*shape).astype(np.float32)

    # Run analytical solver
    l3, l2, l1 = solve_symmetric_eigenvalues_3x3(Hxx, Hxy, Hxz, Hyy, Hyz, Hzz)
    eigs_ana = np.stack([l3, l2, l1], axis=-1)

    # Run numpy solver
    H = np.zeros((*shape, 3, 3), dtype=np.float32)
    H[..., 0, 0] = Hxx
    H[..., 0, 1] = Hxy
    H[..., 0, 2] = Hxz
    H[..., 1, 0] = Hxy
    H[..., 1, 1] = Hyy
    H[..., 1, 2] = Hyz
    H[..., 2, 0] = Hxz
    H[..., 2, 1] = Hyz
    H[..., 2, 2] = Hzz

    eigs_np = np.linalg.eigvalsh(H)

    # Verify shape
    assert eigs_ana.shape == (*shape, 3)

    # Verify values (allow some tolerance for float precision vs double precision internal differences)
    # The analytical solver uses float32 operations if inputs are float32, while eigvalsh might use double.
    np.testing.assert_allclose(eigs_ana, eigs_np, rtol=1e-4, atol=1e-4)

def test_solve_symmetric_eigenvalues_3x3_diagonal():
    """Test with diagonal matrices (multiple eigenvalues)."""
    shape = (5, 5)
    Hxx = np.full(shape, 1.0, dtype=np.float32)
    Hyy = np.full(shape, 2.0, dtype=np.float32)
    Hzz = np.full(shape, 3.0, dtype=np.float32)
    Hxy = np.zeros(shape, dtype=np.float32)
    Hxz = np.zeros(shape, dtype=np.float32)
    Hyz = np.zeros(shape, dtype=np.float32)

    l3, l2, l1 = solve_symmetric_eigenvalues_3x3(Hxx, Hxy, Hxz, Hyy, Hyz, Hzz)

    # Should be sorted 1.0, 2.0, 3.0
    assert np.all(l3 == 1.0)
    assert np.all(l2 == 2.0)
    assert np.all(l1 == 3.0)

def test_solve_symmetric_eigenvalues_3x3_singular():
    """Test with singular matrices (zero eigenvalues)."""
    shape = (5, 5)
    # Matrix with one zero eigenvalue: e.g. diag(1, 1, 0)
    Hxx = np.ones(shape, dtype=np.float32)
    Hyy = np.ones(shape, dtype=np.float32)
    Hzz = np.zeros(shape, dtype=np.float32)
    Hxy = np.zeros(shape, dtype=np.float32)
    Hxz = np.zeros(shape, dtype=np.float32)
    Hyz = np.zeros(shape, dtype=np.float32)

    l3, l2, l1 = solve_symmetric_eigenvalues_3x3(Hxx, Hxy, Hxz, Hyy, Hyz, Hzz)

    # Sorted: 0, 1, 1
    # Relax tolerance to 1e-4 due to float32 precision in analytical solver
    np.testing.assert_allclose(l3, 0.0, atol=1e-4)
    np.testing.assert_allclose(l2, 1.0, atol=1e-4)
    np.testing.assert_allclose(l1, 1.0, atol=1e-4)

def test_solve_symmetric_eigenvalues_3x3_isotropic():
    """Test with isotropic matrices (triple eigenvalues)."""
    shape = (5, 5)
    val = 2.5
    Hxx = np.full(shape, val, dtype=np.float32)
    Hyy = np.full(shape, val, dtype=np.float32)
    Hzz = np.full(shape, val, dtype=np.float32)
    Hxy = np.zeros(shape, dtype=np.float32)
    Hxz = np.zeros(shape, dtype=np.float32)
    Hyz = np.zeros(shape, dtype=np.float32)

    l3, l2, l1 = solve_symmetric_eigenvalues_3x3(Hxx, Hxy, Hxz, Hyy, Hyz, Hzz)

    np.testing.assert_allclose(l3, val, atol=1e-5)
    np.testing.assert_allclose(l2, val, atol=1e-5)
    np.testing.assert_allclose(l1, val, atol=1e-5)
