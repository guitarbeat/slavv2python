
import numpy as np
import pytest
from slavv.core.energy import solve_symmetric_eigenvalues_3x3

def test_random_symmetric_matrices():
    """Test against random symmetric matrices."""
    np.random.seed(42)
    shape = (10, 10)
    # Generate random symmetric matrices
    Hxx = np.random.randn(*shape).astype(np.float32)
    Hxy = np.random.randn(*shape).astype(np.float32)
    Hxz = np.random.randn(*shape).astype(np.float32)
    Hyy = np.random.randn(*shape).astype(np.float32)
    Hyz = np.random.randn(*shape).astype(np.float32)
    Hzz = np.random.randn(*shape).astype(np.float32)

    # Expected from eigvalsh
    # Construct full (..., 3, 3) matrix
    H = np.zeros(shape + (3, 3), dtype=np.float32)
    H[..., 0, 0] = Hxx
    H[..., 0, 1] = Hxy
    H[..., 0, 2] = Hxz
    H[..., 1, 0] = Hxy
    H[..., 1, 1] = Hyy
    H[..., 1, 2] = Hyz
    H[..., 2, 0] = Hxz
    H[..., 2, 1] = Hyz
    H[..., 2, 2] = Hzz

    expected = np.linalg.eigvalsh(H)
    # eigvalsh returns ascending order
    # solve_symmetric_eigenvalues_3x3 should return sorted eigenvalues
    # but let's check what we implemented. We implemented ascending sort in the benchmark.
    # The actual implementation in energy.py will return descending (l1 >= l2 >= l3) to match current usage.

    l1, l2, l3 = solve_symmetric_eigenvalues_3x3(Hxx, Hxy, Hxz, Hyy, Hyz, Hzz)

    # l1, l2, l3 are expected to be descending: l1 >= l2 >= l3
    # expected from eigvalsh is ascending: e0 <= e1 <= e2

    np.testing.assert_allclose(l1, expected[..., 2], rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(l2, expected[..., 1], rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(l3, expected[..., 0], rtol=1e-4, atol=1e-5)

def test_diagonal_matrices():
    """Test diagonal matrices (eigenvalues are diagonal elements)."""
    shape = (5, 5)
    Hxx = np.full(shape, 3.0, dtype=np.float32)
    Hyy = np.full(shape, 1.0, dtype=np.float32)
    Hzz = np.full(shape, 2.0, dtype=np.float32)
    Hxy = np.zeros(shape, dtype=np.float32)
    Hxz = np.zeros(shape, dtype=np.float32)
    Hyz = np.zeros(shape, dtype=np.float32)

    l1, l2, l3 = solve_symmetric_eigenvalues_3x3(Hxx, Hxy, Hxz, Hyy, Hyz, Hzz)

    # Expected: 3, 2, 1
    np.testing.assert_allclose(l1, 3.0, rtol=1e-5)
    np.testing.assert_allclose(l2, 2.0, rtol=1e-5)
    np.testing.assert_allclose(l3, 1.0, rtol=1e-5)

def test_triple_root():
    """Test matrices with triple roots (spherical tensor)."""
    shape = (5, 5)
    val = 2.5
    Hxx = np.full(shape, val, dtype=np.float32)
    Hyy = np.full(shape, val, dtype=np.float32)
    Hzz = np.full(shape, val, dtype=np.float32)
    Hxy = np.zeros(shape, dtype=np.float32)
    Hxz = np.zeros(shape, dtype=np.float32)
    Hyz = np.zeros(shape, dtype=np.float32)

    l1, l2, l3 = solve_symmetric_eigenvalues_3x3(Hxx, Hxy, Hxz, Hyy, Hyz, Hzz)

    np.testing.assert_allclose(l1, val, rtol=1e-5)
    np.testing.assert_allclose(l2, val, rtol=1e-5)
    np.testing.assert_allclose(l3, val, rtol=1e-5)

def test_singular_matrix():
    """Test singular matrices (zero eigenvalue)."""
    # Rank 1 matrix from vector [1, 0, 0] -> eigenvalue 1, 0, 0
    shape = (5, 5)
    Hxx = np.ones(shape, dtype=np.float32)
    Hyy = np.zeros(shape, dtype=np.float32)
    Hzz = np.zeros(shape, dtype=np.float32)
    Hxy = np.zeros(shape, dtype=np.float32)
    Hxz = np.zeros(shape, dtype=np.float32)
    Hyz = np.zeros(shape, dtype=np.float32)

    l1, l2, l3 = solve_symmetric_eigenvalues_3x3(Hxx, Hxy, Hxz, Hyy, Hyz, Hzz)

    np.testing.assert_allclose(l1, 1.0, atol=1e-3)
    np.testing.assert_allclose(l2, 0.0, atol=1e-3)
    np.testing.assert_allclose(l3, 0.0, atol=1e-3)

def test_double_root():
    """Test double root case."""
    # Diagonal 2, 2, 1
    shape = (5, 5)
    Hxx = np.full(shape, 2.0, dtype=np.float32)
    Hyy = np.full(shape, 2.0, dtype=np.float32)
    Hzz = np.full(shape, 1.0, dtype=np.float32)
    Hxy = np.zeros(shape, dtype=np.float32)
    Hxz = np.zeros(shape, dtype=np.float32)
    Hyz = np.zeros(shape, dtype=np.float32)

    l1, l2, l3 = solve_symmetric_eigenvalues_3x3(Hxx, Hxy, Hxz, Hyy, Hyz, Hzz)

    # Relaxed tolerance for float32 analytical solution near multiple roots
    np.testing.assert_allclose(l1, 2.0, atol=1e-3)
    np.testing.assert_allclose(l2, 2.0, atol=1e-3)
    np.testing.assert_allclose(l3, 1.0, atol=1e-3)
