"""Network plotting package: deep ``NetworkVisualizer`` facade + plot modules.

Prefer ``NetworkVisualizer`` for application code (owns theme + Stage Result
normalization). Sibling modules hold plot/export implementation detail.
"""

from __future__ import annotations

from slavv_python.visualization.network_plots.helpers import NETWORK_COLOR_SCHEMES
from slavv_python.visualization.network_plots.visualizer import (
    NetworkVisualizer,
    processing_payload,
    stage_payload,
)

__all__ = [
    "NETWORK_COLOR_SCHEMES",
    "NetworkVisualizer",
    "processing_payload",
    "stage_payload",
]
