"""
Evaluation and comparison tools for SLAVV.

This package contains utilities for validation, comparison, and visualization
useful for evaluating the Python implementation against MATLAB baselines.
"""

from .comparison import (
    load_parameters,
    orchestrate_comparison,
    run_standalone_comparison
)

from .metrics import (
    match_vertices,
    compare_vertices,
    compare_edges,
    compare_networks,
    compare_results
)

from .reporting import (
    generate_summary
)

from .viz import (
    set_plot_style,
    plot_count_comparison,
    plot_radius_distributions
)

from .setup_checks import Validator
