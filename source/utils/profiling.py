from __future__ import annotations

import cProfile
import pstats
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

# from ..core import SlavvPipeline  # Moved inside to avoid circular dependency


def profile_process_image(
    image: np.ndarray, parameters: dict[str, Any] | None = None
) -> pstats.Stats:
    """Profile the SLAVV pipeline on a 3D volume.

    Parameters
    ----------
    image : np.ndarray
        Input 3D volume (y, x, z).
    parameters : dict, optional
        Processing parameters passed to :class:`source.core.SlavvPipeline`.
        Defaults to an empty dictionary.

    Returns
    -------
    pstats.Stats
        Profiling statistics for ``run``.
    """
    if parameters is None:
        parameters = {}

    from ..core import SlavvPipeline

    profiler = cProfile.Profile()
    processor = SlavvPipeline()
    profiler.runcall(processor.run, image, parameters)
    return pstats.Stats(profiler)
