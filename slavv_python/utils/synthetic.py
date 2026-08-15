from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

# Fixed ladder rung ids (KTD1). Order is escalation order for the operator script.
LADDER_RUNG_IDS: tuple[str, ...] = (
    "y_junction_32",
    "double_junction_32",
    "asymmetric_y_48",
    "y_junction_64",
)


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


def generate_synthetic_double_junction_volume(
    shape: tuple[int, int, int] = (32, 32, 32),
    trunk_radius: float = 3.0,
    branch_radius: float = 3.0,
    background_val: float = 0.0,
    vessel_val: float = 1.0,
) -> np.ndarray:
    """Y-junction plus a second opposite-side branch (topology step, ~32³).

    Adds a -X branch at a second Z plane so the volume has two junction sites
    versus the single-junction baseline.
    """
    image = generate_synthetic_y_junction_volume(
        shape=shape,
        trunk_radius=trunk_radius,
        branch_radius=branch_radius,
        background_val=background_val,
        vessel_val=vessel_val,
    )
    _z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]
    cy, cx = shape[1] // 2, shape[2] // 2
    z2 = max(2, shape[0] // 4)
    z_band = np.abs(_z - z2) <= 1
    second_branch = z_band & ((y - cy) ** 2 <= branch_radius**2) & (x <= cx)
    image[second_branch] = vessel_val
    return image


def generate_synthetic_asymmetric_y_volume(
    shape: tuple[int, int, int] = (48, 48, 48),
    trunk_radius: float = 4.0,
    branch_radius: float = 2.5,
    junction_y_offset: int = 6,
    background_val: float = 0.0,
    vessel_val: float = 1.0,
) -> np.ndarray:
    """Asymmetric-radii Y with an offset junction on ~48³ (geometry asymmetry)."""
    image = np.full(shape, background_val, dtype=np.float32)
    _z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]
    cy = shape[1] // 2 + int(junction_y_offset)
    cx = shape[2] // 2
    trunk = ((x - cx) ** 2 + (y - cy) ** 2) <= trunk_radius**2
    image[np.broadcast_to(trunk, image.shape)] = vessel_val
    z_mid = shape[0] // 2
    z_band = np.abs(_z - z_mid) <= 1
    branch_mask = z_band & ((y - cy) ** 2 <= branch_radius**2) & (x >= cx)
    image[branch_mask] = vessel_val
    return image


def _rung_y_junction_32(**kwargs: Any) -> np.ndarray:
    return generate_synthetic_y_junction_volume(
        shape=(32, 32, 32),
        trunk_radius=3.0,
        branch_radius=3.0,
        **kwargs,
    )


def _rung_double_junction_32(**kwargs: Any) -> np.ndarray:
    return generate_synthetic_double_junction_volume(
        shape=(32, 32, 32),
        trunk_radius=3.0,
        branch_radius=3.0,
        **kwargs,
    )


def _rung_asymmetric_y_48(**kwargs: Any) -> np.ndarray:
    return generate_synthetic_asymmetric_y_volume(
        shape=(48, 48, 48),
        trunk_radius=4.0,
        branch_radius=2.5,
        junction_y_offset=6,
        **kwargs,
    )


def _rung_y_junction_64(**kwargs: Any) -> np.ndarray:
    return generate_synthetic_y_junction_volume(
        shape=(64, 64, 64),
        trunk_radius=4.0,
        branch_radius=3.0,
        **kwargs,
    )


_LADDER_BUILDERS: dict[str, Callable[..., np.ndarray]] = {
    "y_junction_32": _rung_y_junction_32,
    "double_junction_32": _rung_double_junction_32,
    "asymmetric_y_48": _rung_asymmetric_y_48,
    "y_junction_64": _rung_y_junction_64,
}

# Max dimension per named rung (soft size-cap checks use this without building).
LADDER_RUNG_MAX_DIM: dict[str, int] = {
    "y_junction_32": 32,
    "double_junction_32": 32,
    "asymmetric_y_48": 48,
    "y_junction_64": 64,
}


def generate_ladder_rung_volume(
    rung_id: str,
    *,
    background_val: float = 0.0,
    vessel_val: float = 1.0,
) -> np.ndarray:
    """Build a fixed named ladder-rung volume (deterministic paint; no search)."""
    try:
        builder = _LADDER_BUILDERS[rung_id]
    except KeyError as exc:
        known = ", ".join(LADDER_RUNG_IDS)
        raise ValueError(f"unknown ladder rung {rung_id!r}; known: {known}") from exc
    return builder(background_val=background_val, vessel_val=vessel_val)
