import numpy as np


def calculate_path_length(path: np.ndarray) -> float:
    """Calculate the total length of a polyline path.

    Parameters
    ----------
    path : np.ndarray
        Array of points with shape ``(N, D)`` where ``D`` is typically 3
        (``y, x, z``). Coordinates are interpreted in the same units as the
        input path (e.g., voxels or microns).

    Returns
    -------
    float
        Sum of Euclidean distances between successive points. Returns ``0.0``
        for degenerate paths with fewer than two points.
    """
    if len(path) < 2:
        return 0.0

    distances = np.linalg.norm(np.diff(path, axis=0), axis=1)
    return float(np.sum(distances))


