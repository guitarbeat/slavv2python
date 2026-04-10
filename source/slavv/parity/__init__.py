"""Parity and comparison tools for SLAVV.

This package contains MATLAB/Python parity helpers for validation,
comparison, reporting, and visualization.
"""

from __future__ import annotations

from .comparison import load_parameters as load_parameters
from .comparison import orchestrate_comparison as orchestrate_comparison
from .comparison import run_standalone_comparison as run_standalone_comparison
from .comparison_plots import plot_count_comparison as plot_count_comparison
from .comparison_plots import plot_radius_distributions as plot_radius_distributions
from .comparison_plots import set_plot_style as set_plot_style
from .environment_checks import Validator as Validator
from .metrics import compare_edges as compare_edges
from .metrics import compare_networks as compare_networks
from .metrics import compare_results as compare_results
from .metrics import compare_vertices as compare_vertices
from .metrics import match_vertices as match_vertices
from .reporting import generate_summary as generate_summary

__all__ = [
    "Validator",
    "compare_edges",
    "compare_networks",
    "compare_results",
    "compare_vertices",
    "generate_summary",
    "load_parameters",
    "match_vertices",
    "orchestrate_comparison",
    "plot_count_comparison",
    "plot_radius_distributions",
    "run_standalone_comparison",
    "set_plot_style",
]
