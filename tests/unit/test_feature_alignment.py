import sys
import pathlib
import numpy as np

from slavv.analysis.ml_curator import MLCurator
from slavv.utils import calculate_path_length


def test_vertex_edge_feature_enrichment():
    curator = MLCurator()
    vertices = {
        'positions': np.array([[1, 1, 1], [3, 3, 3]]),
        'energies': np.array([1.0, 2.0]),
        'scales': np.array([1.0, 2.0]),
        'radii_pixels': np.array([1.0, 2.0]),
    }
    energy_data = {'energy': np.ones((5, 5, 5))}
    image_shape = (5, 5, 5)

    v_feats = curator.extract_vertex_features(vertices, energy_data, image_shape)
    # radius-to-scale ratio
    assert np.isclose(v_feats[0][3], 1.0)
    # energy ratio to local mean
    assert np.isclose(v_feats[1][13], 2.0)

    edges = {
        'traces': [np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2]])],
        'connections': [(0, 1)],
    }
    e_feats = curator.extract_edge_features(edges, vertices, energy_data)
    assert np.isclose(e_feats[0][11], 1.0)  # start radius
    assert np.isclose(e_feats[0][12], 2.0)  # end radius
    assert np.isclose(e_feats[0][14], calculate_path_length(edges['traces'][0]) / 1.5)
    assert np.isclose(e_feats[0][15], -1.0)
