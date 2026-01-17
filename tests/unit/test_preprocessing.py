import sys
import pathlib
import numpy as np

# Ensure module path
from src.slavv.vectorization_core import preprocess_image


def test_preprocess_normalizes_to_unit_range():
    img = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    out = preprocess_image(img, {})
    assert np.isclose(out.min(), 0.0)
    assert np.isclose(out.max(), 1.0)


def test_preprocess_bandpass_removes_axial_gradient():
    # Linear gradient along z to simulate axial banding
    img = np.zeros((4, 4, 4), dtype=np.float32)
    for z in range(4):
        img[:, :, z] = z
    without = preprocess_image(img, {})
    out = preprocess_image(img, {"bandpass_window": 1.0})
    # Bandpass filtering should reduce axial variance compared to no filtering
    var_without = np.var(without, axis=2).mean()
    var_with = np.var(out, axis=2).mean()
    assert var_with < var_without * 0.1
