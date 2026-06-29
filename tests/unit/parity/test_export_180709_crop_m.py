"""Tests for 180709_E tier-M crop export."""

from __future__ import annotations

import json

import numpy as np
import pytest
import tifffile

from slavv_python.analytics.parity.probes.crop_export import export_crop_m, write_roi_metadata


@pytest.mark.unit
def test_export_crop_m_writes_expected_shape(tmp_path):
    source = tmp_path / "180709_E.tif"
    full = np.arange(64 * 512 * 512, dtype=np.uint16).reshape(64, 512, 512)
    tifffile.imwrite(source, full)

    output = tmp_path / "180709_E_crop_M.tif"
    cropped = export_crop_m(source, output)

    assert cropped.shape == (64, 256, 256)
    assert output.is_file()
    loaded = tifffile.imread(output)
    np.testing.assert_array_equal(loaded, cropped)
    np.testing.assert_array_equal(cropped, full[:, 128:384, 128:384])


@pytest.mark.unit
def test_write_roi_metadata(tmp_path):
    output = tmp_path / "180709_E_crop_M.tif"
    output.write_bytes(b"placeholder")
    meta_path = write_roi_metadata(output)
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    assert payload["tier"] == "M"
    assert payload["dataset_id"] == "180709_E_crop_M"
    assert payload["crop_shape"] == [64, 256, 256]
