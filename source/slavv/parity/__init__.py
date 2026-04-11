"""Parity and comparison tools for SLAVV."""

from __future__ import annotations

from importlib import import_module

_EXPORT_MAP = {
    "Validator": (".environment_checks", "Validator"),
    "compare_edges": (".metrics", "compare_edges"),
    "compare_networks": (".metrics", "compare_networks"),
    "compare_results": (".metrics", "compare_results"),
    "compare_vertices": (".metrics", "compare_vertices"),
    "generate_summary": (".reporting", "generate_summary"),
    "load_parameters": (".comparison", "load_parameters"),
    "match_vertices": (".metrics", "match_vertices"),
    "orchestrate_comparison": (".comparison", "orchestrate_comparison"),
    "plot_count_comparison": (".comparison_plots", "plot_count_comparison"),
    "plot_radius_distributions": (".comparison_plots", "plot_radius_distributions"),
    "run_standalone_comparison": (".comparison", "run_standalone_comparison"),
    "set_plot_style": (".comparison_plots", "set_plot_style"),
}

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


def __getattr__(name: str):
    try:
        module_name, attribute_name = _EXPORT_MAP[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
