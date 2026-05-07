from pstats import Stats

import numpy as np

# Ensure module path
from slavv_python.utils import preprocess_image, weighted_ks_test
from slavv_python.utils.profiling import profile_process_image


def test_preprocess_normalizes_to_unit_range():
    img = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    out = preprocess_image(img, {})
    assert np.isclose(out.min(), 0.0)
    assert np.isclose(out.max(), 1.0)


def test_preprocess_constant_image_returns_zero_float32_volume():
    img = np.full((2, 2, 2), 7.0, dtype=np.float64)

    out = preprocess_image(img, {})

    assert out.dtype == np.float32
    assert np.allclose(out, 0.0)


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
    assert np.isclose(out.min(), 0.0)
    assert out.max() <= 1.0


def test_weighted_ks_unweighted():
    x = np.array([0, 1])
    y = np.array([0, 2])
    stat = weighted_ks_test(x, y)
    assert np.isclose(stat, 0.5)


def test_weighted_ks_with_weights():
    x = np.array([0, 1])
    y = np.array([0, 2])
    w2 = np.array([0.1, 0.9])
    stat = weighted_ks_test(x, y, weights2=w2)
    assert np.isclose(stat, 0.9)


def test_profile_process_image_returns_stats():
    volume = np.zeros((3, 3, 3), dtype=float)
    stats = profile_process_image(volume, {})
    assert isinstance(stats, Stats)
    assert any(func[0].endswith("pipeline.py") and func[2] == "run" for func in stats.stats)
