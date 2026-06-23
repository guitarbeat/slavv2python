from __future__ import annotations

import numpy as np

from slavv_python.analytics.parity.mismatch_diagnostics import (
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
