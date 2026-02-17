"""
Evaluation and comparison tools for SLAVV.

This package contains utilities for validation, comparison, and visualization
useful for evaluating the Python implementation against MATLAB baselines.
"""

from .comparison import (
    load_parameters as load_parameters,
    orchestrate_comparison as orchestrate_comparison,
    run_standalone_comparison as run_standalone_comparison
)

from .metrics import (
    match_vertices as match_vertices,
    compare_vertices as compare_vertices,
    compare_edges as compare_edges,
    compare_networks as compare_networks,
    compare_results as compare_results
)

from .reporting import (
    generate_summary as generate_summary
)

from .viz import (
    set_plot_style as set_plot_style,
    plot_count_comparison as plot_count_comparison,
    plot_radius_distributions as plot_radius_distributions
)

from .setup_checks import Validator as Validator
