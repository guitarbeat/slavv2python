"""
Evaluation and comparison tools for SLAVV.

This package contains utilities for validation, comparison, and visualization
useful for evaluating the Python implementation against MATLAB baselines.
"""

from __future__ import annotations

from .comparison import load_parameters as load_parameters
from .comparison import orchestrate_comparison as orchestrate_comparison
from .comparison import run_standalone_comparison as run_standalone_comparison
from .metrics import compare_edges as compare_edges
from .metrics import compare_networks as compare_networks
from .metrics import compare_results as compare_results
from .metrics import compare_vertices as compare_vertices
from .metrics import match_vertices as match_vertices
from .reporting import generate_summary as generate_summary
from .setup_checks import Validator as Validator
from .viz import plot_count_comparison as plot_count_comparison
from .viz import plot_radius_distributions as plot_radius_distributions
from .viz import set_plot_style as set_plot_style
