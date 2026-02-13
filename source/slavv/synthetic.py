"""Synthetic data generation for testing and demos."""
import numpy as np

def generate_synthetic_vessel_volume(
    shape: tuple[int, int, int] = (64, 64, 64),
    vessel_radius: float = 5.0,
    background_val: float = 0.0,
    vessel_val: float = 1.0
) -> np.ndarray:
    """Generate a 3D volume with a simple synthetic vessel.

    Creates a vertical tubular vessel centered in the volume.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Dimensions of the volume (z, y, x).
    vessel_radius : float
        Radius of the vessel cylinder in pixels.
    background_val : float
        Intensity value for the background.
    vessel_val : float
        Intensity value for the vessel structure.

    Returns
    -------
    np.ndarray
        3D float32 array containing the synthetic data.
    """
    image = np.full(shape, background_val, dtype=np.float32)
    
    # Coordinates grid
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    
    # Vertical vessel centered in X-Y plane
    cy, cx = shape[1] // 2, shape[2] // 2
    mask = ((x - cx)**2 + (y - cy)**2) <= vessel_radius**2
    
    # Broadcast and assign
    image[np.broadcast_to(mask, image.shape)] = vessel_val
    
    return image
