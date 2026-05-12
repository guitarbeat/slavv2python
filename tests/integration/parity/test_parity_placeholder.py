"""
Integration parity tests — placeholder.

Full integration parity tests are tracked in:
    docs/reference/core/MATLAB_METHOD_IMPLEMENTATION_PLAN.md

These tests require a promoted oracle and a completed `prove-exact` run.
Use the parity_experiment.py CLI workflow to exercise parity proofs.
"""

import pytest


@pytest.mark.integration
@pytest.mark.skip(reason="Parity integration tests not yet implemented — see MATLAB_METHOD_IMPLEMENTATION_PLAN.md")
def test_parity_integration_placeholder():
    """Placeholder: parity integration tests will live here."""
    pass
