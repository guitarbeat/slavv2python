"""Visualization package for vascular networks and curation UIs."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    from slavv_python.visualization.network_plots import (
        NetworkVisualizer,
        processing_payload,
        stage_payload,
    )

    __all__ = [
        "NetworkVisualizer",
        "processing_payload",
        "stage_payload",
    ]
except ImportError as exc:
    logger.warning("Visualization module unavailable (missing dependencies): %s", exc)
    NetworkVisualizer = None  # type: ignore[assignment,misc]
    processing_payload = None  # type: ignore[assignment,misc]
    stage_payload = None  # type: ignore[assignment,misc]
    __all__ = []
