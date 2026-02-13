import cProfile
import pstats
from typing import Any, Dict, Optional

import numpy as np

from .core import SLAVVProcessor


def profile_process_image(image: np.ndarray, parameters: Optional[Dict[str, Any]] = None) -> pstats.Stats:
    """Profile the SLAVV pipeline on a 3D volume.

    Parameters
    ----------
    image : np.ndarray
        Input 3D volume (y, x, z).
    parameters : dict, optional
        Processing parameters passed to :class:`SLAVVProcessor`. Defaults to an empty
        dictionary, which uses MATLAB-equivalent defaults.

    Returns
    -------
    pstats.Stats
        Profiling statistics for ``process_image``.
    """
    if parameters is None:
        parameters = {}

    profiler = cProfile.Profile()
    processor = SLAVVProcessor()
    profiler.runcall(processor.process_image, image, parameters)
    return pstats.Stats(profiler)
