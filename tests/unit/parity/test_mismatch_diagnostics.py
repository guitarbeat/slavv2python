from __future__ import annotations

import numpy as np

from slavv_python.analytics.parity.proof.mismatch_diagnostics import (
    build_mismatch_diagnostics,
    persist_mismatch_diagnostics,
)


def test_numeric_diagnostic_uses_fortran_order_and_scale_context(tmp_path):
    matlab = {
        "energy": np.zeros((2, 2, 1), dtype=np.float64),
        "scale_indices": np.zeros((2, 2, 1), dtype=np.int16),
    }
    python = {
        "energy": np.zeros((2, 2, 1), dtype=np.float64),
        "scale_indices": np.zeros((2, 2, 1), dtype=np.int16),
    }
    python["energy"][1, 0, 0] = -2.0
    python["scale_indices"][1, 0, 0] = 4

    diagnosis = build_mismatch_diagnostics("energy", matlab, python, {})

    first = diagnosis["fields"][0]
    assert first["first_coordinate"] == [1, 0, 0]
    assert first["first_fortran_linear_index"] == 1
    assert diagnosis["energy_context"]["winner_scale_disagrees"] is True

    paths = persist_mismatch_diagnostics(
        tmp_path,
        report={"first_failing_stage": "energy"},
        matlab_artifacts={"energy": matlab},
        python_artifacts={"energy": python},
        params={},
    )
    assert paths[0] is not None
    assert paths[0].is_file()
    assert paths[1] is not None
    assert paths[1].is_file()


def test_energy_diagnostic_includes_ulp_stats_for_scale_agreeing_drift() -> None:
    matlab_energy = np.array([-20.37433178324523], dtype=np.float64)
    python_energy = np.array([-20.374331783245218], dtype=np.float64)
    matlab = {
        "energy": matlab_energy.reshape(1, 1, 1),
        "scale_indices": np.array([90], dtype=np.int16).reshape(1, 1, 1),
    }
    python = {
        "energy": python_energy.reshape(1, 1, 1),
        "scale_indices": np.array([90], dtype=np.int16).reshape(1, 1, 1),
    }

    diagnosis = build_mismatch_diagnostics("energy", matlab, python, {})
    energy_field = next(field for field in diagnosis["fields"] if field["field"] == "energy")

    assert energy_field["max_ulp"] == 3
    assert energy_field["ulp_histogram"]["3"] == 1
    assert diagnosis["energy_context"]["winner_scale_disagrees"] is False
    assert diagnosis["energy_scale_agreeing_mismatch"]["mismatch_count"] == 1
    assert diagnosis["energy_scale_agreeing_mismatch"]["max_ulp"] == 3
