import pathlib
import sys
import numpy as np

# Add source path for imports
<<<<<<< HEAD
from slavv.analysis import crop_vertices, crop_edges, crop_vertices_by_mask
=======
from slavv.analysis.geometry import crop_vertices, crop_edges, crop_vertices_by_mask
>>>>>>> 3e500a60f45114343cdca16b13c837e3d0f1d578


def test_crop_vertices_and_edges():
    vertices = np.array([[1, 1, 1], [5, 5, 5], [9, 9, 9]], dtype=float)
    bounds = ((0, 5), (0, 5), (0, 5))
    cropped_vertices, mask = crop_vertices(vertices, bounds)
    assert cropped_vertices.shape[0] == 2
    assert mask.tolist() == [True, True, False]

    edges = np.array([[0, 1], [1, 2]])
    cropped_edges, edge_mask = crop_edges(edges, mask)
    assert cropped_edges.shape[0] == 1
    assert edge_mask.tolist() == [True, False]


def test_crop_vertices_by_mask():
    vertices = np.array([[0, 0, 0], [2, 2, 0], [4, 4, 0]], dtype=float)
    mask_volume = np.zeros((5, 5, 1), dtype=bool)
    mask_volume[0:3, 0:3, 0] = True
    cropped_vertices, mask = crop_vertices_by_mask(vertices, mask_volume)
    assert cropped_vertices.shape[0] == 2
    assert mask.tolist() == [True, True, False]
