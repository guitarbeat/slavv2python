import numpy as np
from source.utils import calculate_path_length


def test_calculate_path_length_handles_empty_singleton_and_repeated_points():
    assert calculate_path_length(np.empty((0, 3), dtype=float)) == 0.0
    assert calculate_path_length(np.array([[1.0, 2.0, 3.0]], dtype=float)) == 0.0

    path = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 3.0, 4.0],
        ],
        dtype=float,
    )

    assert np.isclose(calculate_path_length(path), 5.0)


def test_calculate_path_length_accumulates_multisegment_3d_distance():
    path = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 3.0, 4.0],
            [12.0, 3.0, 4.0],
            [12.0, 6.0, 8.0],
        ],
        dtype=float,
    )

    assert np.isclose(calculate_path_length(path), 22.0)
