"""
Image preprocessing functions for source.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.ndimage as ndi


def preprocess_image(image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    """Normalize intensities and optionally correct axial banding.

    Parameters
    ----------
    image:
        Raw input image array (y, x, z).
    params:
        Processing parameters. Uses ``bandpass_window`` to control the
        axial Gaussian smoothing window for band removal.

    Returns
    -------
    np.ndarray
        Preprocessed image with intensities scaled to ``[0, 1]``.
    """
    is_exact = bool(params.get("comparison_exact_network", False))
    
    if is_exact:
        img = image.astype(np.float64)
    else:
        img = image.astype(np.float32)

    # Exact-route parity runs must preserve raw intensity scale to match MATLAB
    # oracle energy (see crop_M_exact prove-exact mismatch investigation).
    if not is_exact:
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
