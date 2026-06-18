"""
Image preprocessing functions for source.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.ndimage as ndi


from slavv_python.pipeline.energy.policy import EnergyPolicy


def preprocess_image(image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    """Normalize intensities and optionally correct axial banding.

    Parameters
    ----------
    image:
        Raw input image array (y, x, z).
    params:
        Processing parameters. Uses ``bandpass_window`` to control the
        axial Gaussian smoothing window for band removal.

    Returns:
    -------
    np.ndarray
        Preprocessed image with intensities potentially scaled to ``[0, 1]``.
    """
    policy = EnergyPolicy.from_params(params)
    img = image.astype(policy.precision)

    # Policy controls whether raw intensity scale is preserved (Exact) or normalized (Paper).
    if policy.intensity_scaling:
        # Scale intensities to [0, 1] for the native paper workflow.
        img -= img.min()
        max_val = img.max()
        if max_val > 0:
            img /= max_val

    # Simple axial band correction inspired by `fix_intensity_bands.m`
    band_window = params.get("bandpass_window", 0.0)
    if band_window > 0:
        background = ndi.gaussian_filter(img, sigma=(0, 0, band_window))
        img = np.clip(img - background, 0, 1)

    return img
