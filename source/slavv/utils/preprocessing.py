"""
Image preprocessing functions for SLAVV.
"""
from typing import Dict, Any
import numpy as np
import scipy.ndimage as ndi


def preprocess_image(image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
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
    img = image.astype(np.float32)

    # Scale intensities to [0, 1]
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
