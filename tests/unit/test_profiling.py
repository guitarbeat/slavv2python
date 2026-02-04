import pathlib
import sys

import numpy as np
from pstats import Stats

# Add source path for imports
try:
    from slavv.profiling import profile_process_image
except ImportError:
    from src.slavv.profiling import profile_process_image


def test_profile_process_image_returns_stats():
    volume = np.zeros((3, 3, 3), dtype=float)
    stats = profile_process_image(volume, {})
    assert isinstance(stats, Stats)
    assert any('process_image' in func[2] for func in stats.stats)
