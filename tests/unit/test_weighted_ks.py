import numpy as np

import sys
import pathlib



from slavv.utils import weighted_ks_test

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
