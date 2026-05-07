from __future__ import annotations

import cProfile
import os
import pstats
from typing import TYPE_CHECKING, Any

import psutil

if TYPE_CHECKING:
    import numpy as np


def get_process_memory_usage() -> float:
    """Return the current process memory usage (RSS) in megabytes."""
    process = psutil.Process(os.getpid())
    return float(process.memory_info().rss) / (1024.0 * 1024.0)


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
