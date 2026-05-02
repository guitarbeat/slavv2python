from pstats import Stats

import numpy as np

from source.utils.profiling import profile_process_image


def test_profile_process_image_returns_stats():
    volume = np.zeros((3, 3, 3), dtype=float)
    stats = profile_process_image(volume, {})
    assert isinstance(stats, Stats)
    assert any(func[0].endswith("pipeline.py") and func[2] == "run" for func in stats.stats)
