"""
Visualization modules for SLAVV.

This subpackage contains visualization tools for vascular networks:
- network_plots: Main NetworkVisualizer class and plotting utilities
"""
import logging

logger = logging.getLogger(__name__)

try:
    from .network_plots import NetworkVisualizer
    __all__ = ["NetworkVisualizer"]
except ImportError as e:
    logger.warning(f"Visualization module unavailable (missing dependencies): {e}")
    NetworkVisualizer = None
    __all__ = []
