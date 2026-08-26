"""Browser manual-curation state and projection coverage."""

from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest

pytest.importorskip("streamlit")

from slavv_python.interface.streamlit.state.manual_curation import (
    CurationSessionError,
    curate_manual_selection,
    materialize_curation_session,
    new_curation_session,
    serialize_curation_session,
    validate_curation_session,
)
from slavv_python.interface.streamlit.state.sample_data import build_sample_tiff
from slavv_python.interface.streamlit.views.manual_curation import build_manual_review_figure
from slavv_python.storage.loaders import load_tiff_volume
from tests.support.payload_builders import build_processing_results


@pytest.mark.unit
def test_generated_sample_is_a_real_deterministic_tiff() -> None:
    first = build_sample_tiff("y_junction_32")
    second = build_sample_tiff("y_junction_32")

    assert first.tiff_bytes == second.tiff_bytes
    assert first.name.endswith(".tif")
    loaded = load_tiff_volume(BytesIO(first.tiff_bytes))
    assert loaded.shape == (32, 32, 32)
    assert np.isfinite(loaded).all()
    assert float(loaded.max()) > float(loaded.min())


@pytest.mark.unit
def test_manual_vertex_rejection_drops_incident_edge_and_remaps_connections() -> None:
    results = build_processing_results()
    vertices, edges = curate_manual_selection(
        results["vertices"],
        results["edges"],
        rejected_vertex_ids=[2],
    )

    assert len(vertices["positions"]) == 2
    assert len(edges["traces"]) == 1
    assert np.asarray(edges["connections"]).tolist() == [[0, 1]]


@pytest.mark.unit
def test_manual_edge_rejection_preserves_vertices() -> None:
    results = build_processing_results()
    vertices, edges = curate_manual_selection(
        results["vertices"],
        results["edges"],
        rejected_edge_ids=[1],
    )

    assert len(vertices["positions"]) == 3
    assert len(edges["traces"]) == 1
    assert np.asarray(edges["connections"]).tolist() == [[0, 1]]


@pytest.mark.unit
def test_manual_projection_exposes_keep_and_reject_states() -> None:
    results = build_processing_results()
    figure = build_manual_review_figure(
        results,
        axis=2,
        depth_range=(0, 3),
        rejected_vertex_ids={2},
        rejected_edge_ids=set(),
    )

    assert figure.layout.paper_bgcolor == "#071111"
    assert any(trace.name == "Keep" for trace in figure.data)
    assert any(trace.name == "Reject" for trace in figure.data)
    assert any(trace.name == "Reject" and trace.mode == "lines" for trace in figure.data)


@pytest.mark.unit
def test_manual_projection_sidebar_display_options_control_overlays() -> None:
    results = build_processing_results()
    figure = build_manual_review_figure(
        results,
        axis=2,
        depth_range=(0, 3),
        rejected_vertex_ids={2},
        rejected_edge_ids=set(),
        show_edges=False,
        show_vertex_labels=False,
        focus_rejected=True,
        contrast="High",
    )

    scatter_traces = [trace for trace in figure.data if trace.type == "scatter"]
    assert scatter_traces
    assert all(trace.name == "Reject" for trace in scatter_traces)
    assert all(trace.mode == "markers" for trace in scatter_traces)


@pytest.mark.unit
def test_curation_session_round_trip_materializes_additions_and_cascade() -> None:
    results = build_processing_results()
    shape = np.asarray(results["energy_data"]["energy"]).shape
    session = new_curation_session(
        results["vertices"], results["edges"], image_shape=shape, dataset_name="fixture"
    )
    session.vertex_truth[2] = False
    session.added_vertices.append(
        {
            "position": [1.0, 1.0, 1.0],
            "energy": -4.0,
            "scale": 0,
            "radii_pixels": [1.0],
            "radius_microns": 1.0,
        }
    )
    session.vertex_truth.append(True)
    session.vertex_deleted.append(False)
    session.added_edges.append(
        {
            "connections": [0, 3],
            "trace": [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            "energy": -3.0,
        }
    )
    session.edge_truth.append(True)
    session.edge_deleted.append(False)

    replayed = validate_curation_session(
        serialize_curation_session(session),
        expected_signature=session.baseline_signature,
        baseline_vertex_count=3,
        baseline_edge_count=2,
        image_shape=shape,
    )
    vertices, edges = materialize_curation_session(
        results["vertices"], results["edges"], replayed
    )

    assert len(vertices["positions"]) == 3
    assert np.asarray(edges["connections"]).tolist() == [[0, 1], [0, 2]]
    assert len(edges["traces"]) == 2


@pytest.mark.unit
def test_curation_preserves_bridge_vertices_and_remaps_seed_endpoints() -> None:
    results = build_processing_results()
    edges = dict(results["edges"])
    edges["traces"] = [np.array([[0, 0, 0], [2, 2, 2]], dtype=np.float32)]
    edges["connections"] = np.array([[0, 3]], dtype=np.int32)
    edges["energies"] = np.array([-1.0], dtype=np.float32)
    edges["bridge_vertex_positions"] = np.array([[2, 2, 2]], dtype=np.float32)
    edges["bridge_vertex_scales"] = np.array([0], dtype=np.int16)
    edges["bridge_vertex_energies"] = np.array([-1.0], dtype=np.float32)
    shape = np.asarray(results["energy_data"]["energy"]).shape
    session = new_curation_session(
        results["vertices"], edges, image_shape=shape, dataset_name="bridge"
    )
    session.vertex_truth[1] = False

    vertices, curated_edges = materialize_curation_session(
        results["vertices"], edges, session
    )

    assert len(vertices["positions"]) == 2
    assert np.asarray(curated_edges["connections"]).tolist() == [[0, 2]]
    assert np.asarray(curated_edges["bridge_vertex_positions"]).tolist() == [[2, 2, 2]]
    assert np.asarray(curated_edges["bridge_vertex_scales"]).tolist() == [0]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(schema_version=99), "Unsupported curation schema"),
        (lambda value: value.update(baseline_signature="another-run"), "different pipeline"),
        (
            lambda value: value.update(
                added_vertices=[
                    {
                        "position": [999, 0, 0],
                        "energy": -1,
                        "scale": 0,
                        "radii_pixels": [1],
                        "radius_microns": 1,
                    }
                ],
                vertex_truth=value["vertex_truth"] + [True],
                vertex_deleted=value["vertex_deleted"] + [False],
            ),
            "outside the image volume",
        ),
    ],
)
def test_curation_import_rejects_incompatible_or_unsafe_payloads(mutation, message) -> None:
    results = build_processing_results()
    shape = np.asarray(results["energy_data"]["energy"]).shape
    session = new_curation_session(
        results["vertices"], results["edges"], image_shape=shape, dataset_name="fixture"
    )
    raw = session.to_dict()
    mutation(raw)

    with pytest.raises(CurationSessionError, match=message):
        validate_curation_session(
            raw,
            expected_signature=session.baseline_signature,
            baseline_vertex_count=3,
            baseline_edge_count=2,
            image_shape=shape,
        )
