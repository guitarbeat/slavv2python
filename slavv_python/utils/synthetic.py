from __future__ import annotations

import numpy as np


def generate_synthetic_vessel_volume(
    shape: tuple[int, int, int] = (64, 64, 64),
    vessel_radius: float = 5.0,
    background_val: float = 0.0,
    vessel_val: float = 1.0,
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
    _z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]

    # Vertical vessel centered in X-Y plane
    cy, cx = shape[1] // 2, shape[2] // 2
    mask = ((x - cx) ** 2 + (y - cy) ** 2) <= vessel_radius**2

    # Broadcast and assign
    image[np.broadcast_to(mask, image.shape)] = vessel_val

    return image


def generate_synthetic_y_junction_volume(
    shape: tuple[int, int, int] = (64, 64, 64),
    trunk_radius: float = 5.0,
    branch_radius: float = 4.0,
    background_val: float = 0.0,
    vessel_val: float = 1.0,
) -> np.ndarray:
    """Generate a 3D volume with a vertical trunk and horizontal branch (Y-junction).

    The trunk runs along Z through the volume center. The branch extends along +X from
    the trunk mid-plane, meeting the trunk at the center voxel column.
    """
    image = generate_synthetic_vessel_volume(
        shape=shape,
        vessel_radius=trunk_radius,
        background_val=background_val,
        vessel_val=vessel_val,
    )

    _z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]
    cy, cx = shape[1] // 2, shape[2] // 2
    z_mid = shape[0] // 2
    z_band = np.abs(_z - z_mid) <= 1
    branch_mask = z_band & ((y - cy) ** 2 <= branch_radius**2) & (x >= cx)
    image[branch_mask] = vessel_val
    return image
