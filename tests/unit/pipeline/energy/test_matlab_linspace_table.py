"""Table-driven regression for MATLAB R2019a crop linspace meshes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from tests.support.parity_harness import linspace_table_path

import slavv_python.pipeline.energy as energy_pkg
from slavv_python.pipeline.energy.exact_mesh import _matlab_zero_based_linspace

OVERRIDES_PATH = Path(energy_pkg.__file__).resolve().parent / "matlab_linspace_overrides.json"


def _table_available() -> bool:
    return linspace_table_path() is not None


@pytest.mark.unit
@pytest.mark.parity
@pytest.mark.skipif(
    not _table_available(), reason="MATLAB linspace probe table fixture not available"
)
def test_crop_linspace_table_matches_matlab_for_all_contexts() -> None:
    table_path = linspace_table_path()
    assert table_path is not None
    payload = json.loads(table_path.read_text(encoding="utf-8"))
    for row in payload["rows"]:
        offset = int(row["offset"])
        stride = int(row["stride"])
        count = int(row["count"])
        local_start = int(row["local_start"])
        expected = np.asarray(row["mesh_0based"], dtype=np.float64)
        actual = _matlab_zero_based_linspace(offset, stride, count, local_start)
        assert actual.shape == expected.shape
        assert np.array_equal(actual, expected), (
            f"mesh mismatch octave={row['octave']} chunk={row['chunk_idx']} "
            f"axis={row['axis']} key=({offset},{stride},{count},{local_start})"
        )


@pytest.mark.unit
@pytest.mark.parity
@pytest.mark.skipif(
    not _table_available(), reason="MATLAB linspace probe table fixture not available"
)
def test_linspace_override_keys_are_subset_of_probe_table() -> None:
    table_path = linspace_table_path()
    assert table_path is not None
    table_payload = json.loads(table_path.read_text(encoding="utf-8"))
    table_keys = {
        f"{row['offset']},{row['stride']},{row['count']},{row['local_start']}"
        for row in table_payload["rows"]
    }

    override_keys = set(json.loads(OVERRIDES_PATH.read_text(encoding="utf-8")))
    orphan_keys = sorted(override_keys - table_keys)
    assert not orphan_keys, f"override keys missing from probe table: {orphan_keys[:5]}"
