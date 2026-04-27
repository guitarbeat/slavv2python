import numpy as np
from source.core.energy import spherical_structuring_element


def test_anisotropic_structuring_element():
    strel_iso = spherical_structuring_element(1, np.array([1.0, 1.0, 1.0]))
    strel_aniso = spherical_structuring_element(1, np.array([1.0, 1.0, 2.0]))
    assert strel_iso.shape == (3, 3, 3)
    assert strel_iso[1, 1, 2]
    assert not strel_aniso[1, 1, 2]
