from __future__ import annotations

import numpy as np
from dev.tests.support.payload_builders import build_energy_result, build_processing_results

from slavv.models import (
    EdgeSet,
    EnergyResult,
    NetworkResult,
    PipelineResult,
    VertexSet,
    normalize_pipeline_result,
)


def test_energy_result_roundtrip_preserves_extra_fields():
    payload = build_energy_result(overrides={"energy_sign": -1.0, "custom": "value"})

    result = EnergyResult.from_dict(payload)
    restored = result.to_dict()

    assert result.image_shape == payload["image_shape"]
    assert np.array_equal(restored["energy"], payload["energy"])
    assert restored["energy_sign"] == -1.0
    assert restored["custom"] == "value"


def test_component_models_roundtrip_processing_payloads():
    payload = build_processing_results()

    vertices = VertexSet.from_dict(payload["vertices"])
    edges = EdgeSet.from_dict(payload["edges"])
    network = NetworkResult.from_dict(payload["network"])

    assert np.array_equal(vertices.to_dict()["positions"], payload["vertices"]["positions"])
    assert np.array_equal(edges.to_dict()["connections"], payload["edges"]["connections"])
    assert np.array_equal(
        network.to_dict()["vertex_degrees"],
        payload["network"]["vertex_degrees"],
    )


def test_vertex_set_uses_legacy_radii_as_fallback_for_normalized_fields():
    payload = {
        "positions": np.array([[0.0, 0.0, 0.0]], dtype=float),
        "radii": np.array([1.5], dtype=np.float32),
    }

    vertices = VertexSet.from_dict(payload)
    restored = vertices.to_dict()

    assert np.array_equal(restored["radii_microns"], np.array([1.5], dtype=np.float32))
    assert np.array_equal(restored["radii_pixels"], np.array([1.5], dtype=np.float32))
    assert np.array_equal(restored["radii"], np.array([1.5], dtype=np.float32))


def test_pipeline_result_roundtrip_preserves_nested_payloads_and_extra_fields():
    payload = build_processing_results(overrides={"metadata": {"source": "unit-test"}})

    result = PipelineResult.from_dict(payload)
    restored = result.to_dict()

    assert np.array_equal(
        restored["energy_data"]["energy"],
        payload["energy_data"]["energy"],
    )
    assert np.array_equal(
        restored["vertices"]["radii_microns"],
        payload["vertices"]["radii_microns"],
    )
    assert restored["metadata"] == {"source": "unit-test"}


def test_normalize_pipeline_result_accepts_mapping_like_payloads():
    payload = build_processing_results(overrides={"metadata": {"source": "mapping"}})

    result = normalize_pipeline_result(payload)

    assert isinstance(result, PipelineResult)
    assert np.array_equal(result.vertices.to_dict()["positions"], payload["vertices"]["positions"])
    assert result.extra["metadata"] == {"source": "mapping"}
