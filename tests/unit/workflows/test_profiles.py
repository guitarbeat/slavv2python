from __future__ import annotations

from slavv_python.utils import validate_parameters
from slavv_python.workflows import (
    apply_pipeline_profile,
    get_pipeline_profile_defaults,
)


def test_get_pipeline_profile_defaults_returns_paper_defaults():
    defaults = get_pipeline_profile_defaults("paper")

    assert defaults["pipeline_profile"] == "paper"
    assert defaults["energy_projection_mode"] == "paper"
    assert defaults["edge_method"] == "tracing"


def test_apply_pipeline_profile_preserves_explicit_overrides():
    resolved = apply_pipeline_profile(
        {
            "pipeline_profile": "paper",
            "energy_projection_mode": "matlab",
            "number_of_edges_per_vertex": 6,
        }
    )

    assert resolved["pipeline_profile"] == "paper"
    assert resolved["energy_projection_mode"] == "matlab"
    assert resolved["number_of_edges_per_vertex"] == 6


def test_validate_parameters_applies_profile_defaults_before_validation():
    validated = validate_parameters({"pipeline_profile": "paper"})

    assert validated["pipeline_profile"] == "paper"
    assert validated["energy_projection_mode"] == "paper"
    assert validated["edge_method"] == "tracing"
