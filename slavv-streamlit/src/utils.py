import numpy as np

def calculate_path_length(path: np.ndarray) -> float:
    """Calculate total length of a path"""
    if len(path) < 2:
        return 0.0
    
    distances = np.linalg.norm(np.diff(path, axis=0), axis=1)
    return np.sum(distances)


