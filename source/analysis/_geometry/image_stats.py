from __future__ import annotations

import numpy as np


def calculate_image_stats(
    image: np.ndarray,
    mask: np.ndarray,
) -> tuple[float, float, float, float]:
    """Calculate mean and standard deviation of an image inside and outside a mask."""
    img_arr = np.asarray(image, dtype=float)
    mask_arr = np.asarray(mask, dtype=bool)
    if img_arr.shape != mask_arr.shape:
        raise ValueError("Image and mask must have the same shape")

    inside_vals = img_arr[mask_arr]
    outside_vals = img_arr[~mask_arr]

    mean_in = float(np.mean(inside_vals)) if inside_vals.size > 0 else 0.0
    std_in = float(np.std(inside_vals)) if inside_vals.size > 0 else 0.0
    mean_out = float(np.mean(outside_vals)) if outside_vals.size > 0 else 0.0
    std_out = float(np.std(outside_vals)) if outside_vals.size > 0 else 0.0

    return mean_in, std_in, mean_out, std_out
