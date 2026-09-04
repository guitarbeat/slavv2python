"""Browser-native manual curation workspace for SLAVV."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from slavv_python.interface.streamlit.components.matlab_curator import matlab_curator
from slavv_python.interface.streamlit.curation_trust_labels import (
    trust_claim_chrome_visible,
)
from slavv_python.interface.streamlit.navigation import switch_to
from slavv_python.interface.streamlit.services.curation import apply_curated_results
from slavv_python.interface.streamlit.state.manual_curation import (
    CurationSessionError,
    build_curation_baseline_signature,
    materialize_curation_session,
    new_curation_session,
    validate_curation_session,
)
from slavv_python.schema.app_run import AppRunState
from slavv_python.storage.loaders import load_tiff_volume

if TYPE_CHECKING:
    from collections.abc import MutableMapping

ACCENT = "#35D0BA"
REJECT = "#FF6B6B"
INK = "#DCE7E5"


def clear_manual_review(session_state: MutableMapping[str, Any]) -> None:
    """Clear pending browser-review decisions without changing pipeline results."""
    session_state["manual_rejected_vertices"] = set()
    session_state["manual_rejected_edges"] = set()
    session_state["manual_review_generation"] = (
        int(session_state.get("manual_review_generation", 0)) + 1
    )
    session_state.pop("matlab_curator_session", None)
    session_state.pop("matlab_curator_payload_cache", None)
    session_state["matlab_curator_generation"] = (
        int(session_state.get("matlab_curator_generation", 0)) + 1
    )


def _projection_plane(axis: int) -> tuple[int, int, str, str]:
    remaining = [index for index in range(3) if index != axis]
    labels = ("Y", "X", "Z")
    return remaining[0], remaining[1], labels[remaining[1]], labels[remaining[0]]


def build_manual_review_figure(
    results: Mapping[str, Any] | AppRunState,
    *,
    axis: int,
    depth_range: tuple[int, int],
    rejected_vertex_ids: set[int],
    rejected_edge_ids: set[int],
    show_edges: bool = True,
    show_vertex_labels: bool = True,
    focus_rejected: bool = False,
    contrast: str = "Balanced",
) -> go.Figure:
    """Build a MATLAB-inspired projection with explicit keep/reject state."""
    payload = AppRunState.from_value(results).to_dict()
    energy_payload = payload["energy_data"]
    volume = np.asarray(
        energy_payload.get("original", -np.asarray(energy_payload["energy"], dtype=float)),
        dtype=float,
    )
    low, high = depth_range
    slicer = [slice(None)] * 3
    slicer[axis] = slice(low, high + 1)
    projection = np.max(volume[tuple(slicer)], axis=axis)
    finite = projection[np.isfinite(projection)]
    if finite.size:
        contrast_percentiles = {
            "Soft": (0.5, 100.0),
            "Balanced": (2.0, 99.5),
            "High": (8.0, 98.0),
        }
        p_low, p_high = np.percentile(finite, contrast_percentiles.get(contrast, (2, 99.5)))
        projection = np.clip((projection - p_low) / max(p_high - p_low, 1e-12), 0.0, 1.0)

    vertical_axis, horizontal_axis, x_label, y_label = _projection_plane(axis)
    fig = go.Figure(
        go.Heatmap(
            z=projection,
            colorscale=[[0.0, "#071111"], [0.35, "#183332"], [1.0, "#E8F4F1"]],
            showscale=False,
            hoverinfo="skip",
        )
    )

    edges = payload["edges"]
    connections = np.asarray(edges.get("connections", []), dtype=int).reshape(-1, 2)
    for edge_id, trace_value in enumerate(edges.get("traces", [])):
        trace = np.asarray(trace_value, dtype=float)
        if not len(trace) or not np.any((trace[:, axis] >= low) & (trace[:, axis] <= high)):
            continue
        incident_to_rejected_vertex = edge_id < len(connections) and any(
            int(endpoint) in rejected_vertex_ids for endpoint in connections[edge_id]
        )
        rejected = edge_id in rejected_edge_ids or incident_to_rejected_vertex
        if not show_edges or (focus_rejected and not rejected):
            continue
        fig.add_trace(
            go.Scatter(
                x=trace[:, horizontal_axis],
                y=trace[:, vertical_axis],
                mode="lines",
                line={"color": REJECT if rejected else ACCENT, "width": 4 if rejected else 2.5},
                opacity=0.95,
                name="Reject" if rejected else "Keep",
                legendgroup="reject" if rejected else "keep",
                showlegend=False,
                hovertemplate=f"Edge {edge_id}<extra></extra>",
            )
        )

    vertices = payload["vertices"]
    positions = np.asarray(vertices.get("positions", []), dtype=float).reshape(-1, 3)
    energies = np.asarray(vertices.get("energies", np.zeros(len(positions))), dtype=float)
    for rejected, color, label in ((False, ACCENT, "Keep"), (True, REJECT, "Reject")):
        indices = [
            index
            for index, position in enumerate(positions)
            if low <= position[axis] <= high
            and ((index in rejected_vertex_ids) is rejected)
            and (not focus_rejected or rejected)
        ]
        if not indices:
            continue
        selected = positions[indices]
        fig.add_trace(
            go.Scatter(
                x=selected[:, horizontal_axis],
                y=selected[:, vertical_axis],
                mode="markers+text" if show_vertex_labels else "markers",
                text=[str(index) for index in indices] if show_vertex_labels else None,
                textposition="top center",
                textfont={"color": INK, "size": 11},
                marker={
                    "color": color,
                    "size": 11 if rejected else 9,
                    "line": {"color": "#071111", "width": 1.5},
                },
                customdata=np.column_stack([indices, energies[indices]]),
                name=label,
                hovertemplate="Vertex %{customdata[0]:.0f}<br>Energy %{customdata[1]:.3f}<extra></extra>",
            )
        )

    fig.update_layout(
        height=620,
        margin={"l": 8, "r": 8, "t": 40, "b": 8},
        paper_bgcolor="#071111",
        plot_bgcolor="#071111",
        font={"color": INK},
        title={"text": f"{x_label}{y_label} projection · slices {low}-{high}", "x": 0.02},
        legend={"orientation": "h", "x": 0.01, "y": 1.02},
        xaxis={"title": x_label, "showgrid": False, "zeroline": False, "constrain": "domain"},
        yaxis={
            "title": y_label,
            "showgrid": False,
            "zeroline": False,
            "scaleanchor": "x",
            "autorange": "reversed",
        },
        hovermode="closest",
    )
    return fig


def _vertex_review_frame(vertices: Mapping[str, Any]) -> pd.DataFrame:
    positions = np.asarray(vertices.get("positions", []), dtype=float).reshape(-1, 3)
    energies = np.asarray(vertices.get("energies", np.zeros(len(positions))), dtype=float)
    radii = np.asarray(vertices.get("radii_microns", np.zeros(len(positions))), dtype=float)
    return pd.DataFrame(
        {
            "ID": np.arange(len(positions)),
            "Energy": energies,
            "Radius (µm)": radii,
            "Y": positions[:, 0],
            "X": positions[:, 1],
            "Z": positions[:, 2],
        }
    )


def _edge_review_frame(edges: Mapping[str, Any]) -> pd.DataFrame:
    connections = np.asarray(edges.get("connections", []), dtype=int).reshape(-1, 2)
    energies = np.asarray(edges.get("energies", np.zeros(len(connections))), dtype=float)
    traces = edges.get("traces", [])
    return pd.DataFrame(
        {
            "ID": np.arange(len(connections)),
            "From": connections[:, 0],
            "To": connections[:, 1],
            "Energy": energies,
            "Points": [len(trace) for trace in traces],
        }
    )


def render_manual_empty_state() -> bool:
    """Direct users to produce the real Edge Set required by curation."""
    st.subheader("Curation requires processed vertices and edges")
    st.write(
        "Run an uploaded TIFF or a built-in sample through Processing first. "
        "Curation uses the vertices and edges produced by that pipeline run."
    )
    if st.button("Open Processing", type="primary", icon=":material/arrow_forward:"):
        switch_to("processing")
    return False


def _normalize_display_volume(volume: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    """Convert a scientific volume to a stable, transfer-efficient display buffer."""
    values = np.asarray(volume, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size:
        lower, upper = np.percentile(finite, (0.5, 99.5))
    else:
        lower, upper = 0.0, 1.0
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        lower = float(finite.min()) if finite.size else 0.0
        upper = float(finite.max()) if finite.size else 1.0
    if upper <= lower:
        upper = lower + 1.0
    normalized = np.nan_to_num((values - lower) / (upper - lower), nan=0.0, posinf=1.0, neginf=0.0)
    return np.asarray(np.clip(normalized, 0.0, 1.0) * 255.0, dtype=np.uint8), (
        float(lower),
        float(upper),
    )


def _resolve_original_volume(
    app_run: AppRunState,
    energy_payload: Mapping[str, Any],
) -> tuple[np.ndarray, bool, str | None]:
    """Resolve intensity data using the documented Energy/session/manifest precedence."""
    energy = np.asarray(energy_payload["energy"], dtype=np.float32)
    embedded = energy_payload.get("original")
    if embedded is not None:
        volume = np.asarray(embedded)
        if volume.shape == energy.shape:
            return volume, True, None

    session_volume = st.session_state.get("curation_source_volume")
    if session_volume is not None:
        volume = np.asarray(session_volume)
        if volume.shape == energy.shape:
            return volume, True, None

    candidates: list[Path] = []
    source_path = app_run.extra.get("source_path")
    if source_path:
        candidates.append(Path(str(source_path)).expanduser())
    if app_run.run_dir:
        candidates.append(Path(app_run.run_dir) / "01_Input" / "volume.tif")
    for candidate in candidates:
        try:
            if candidate.is_file():
                volume = np.asarray(load_tiff_volume(candidate))
                if volume.shape == energy.shape:
                    return volume, True, None
        except (OSError, RuntimeError, TypeError, ValueError):
            continue

    return (
        -energy,
        False,
        "Original intensity is unavailable. Showing an Energy-derived projection; "
        "intensity histogram and cranium Crop are disabled.",
    )


def _prepare_curator_data(app_run: AppRunState, signature: str) -> dict[str, Any]:
    """Build the typed component input and cache the encoded canonical volume per run."""
    cached = st.session_state.get("matlab_curator_payload_cache")
    if isinstance(cached, dict) and cached.get("signature") == signature:
        return cached["data"]

    payload = app_run.to_dict()
    energy_payload = payload["energy_data"]
    energy = np.asarray(energy_payload["energy"], dtype=np.float32)
    if energy.ndim != 3 or any(dimension <= 0 for dimension in energy.shape):
        raise CurationSessionError("MATLAB-style curation requires a non-empty 3D Energy volume")
    shape = tuple(int(value) for value in energy.shape)
    source_volume, original_available, degraded_reason = _resolve_original_volume(
        app_run, energy_payload
    )
    display, display_range = _normalize_display_volume(source_volume)
    scales = np.asarray(energy_payload.get("scale_indices", []), dtype=np.int16)
    scale_available = scales.shape == energy.shape
    radii_pixels = np.asarray(energy_payload.get("lumen_radius_pixels", []), dtype=np.float32)
    radii_microns = np.asarray(
        energy_payload.get("lumen_radius_microns", []), dtype=np.float32
    ).reshape(-1)
    add_vertex_available = bool(scale_available and radii_pixels.size and radii_microns.size)
    spacing_raw = np.asarray(
        app_run.pipeline.parameters.get("microns_per_voxel", [1.0, 1.0, 1.0]),
        dtype=float,
    ).reshape(-1)
    spacing = (
        spacing_raw[:3]
        if spacing_raw.size >= 3 and np.isfinite(spacing_raw[:3]).all()
        else np.ones(3, dtype=float)
    )
    vertices = payload["vertices"]
    edges = payload["edges"]
    # The MATLAB-facing public buffer is [Y,X,Z]. Cornerstone consumes an
    # X-fast scalar buffer, equivalent to a contiguous [Z,Y,X] transpose.
    cornerstone_volume = np.ascontiguousarray(display.transpose(2, 0, 1)).reshape(-1)
    data: dict[str, Any] = {
        "volumeKey": signature,
        "sessionRevision": int(st.session_state.get("matlab_curator_generation", 0)),
        "displayVolume": np.ascontiguousarray(display).reshape(-1),
        "cornerstoneVolume": cornerstone_volume,
        "energyVolume": np.ascontiguousarray(energy).reshape(-1),
        "scaleVolume": (
            np.ascontiguousarray(scales).reshape(-1)
            if scale_available
            else np.empty(0, dtype=np.int16)
        ),
        "shape": list(shape),
        "spacing": spacing.astype(float).tolist(),
        "displayRange": list(display_range),
        "originalAvailable": original_available,
        "addVertexAvailable": add_vertex_available,
        "degradedReason": degraded_reason,
        "showTrustClaim": trust_claim_chrome_visible(degraded_reason),
        "vertices": {
            "positions": np.asarray(vertices.get("positions", []), dtype=float)
            .reshape(-1, 3)
            .tolist(),
            "energies": np.asarray(vertices.get("energies", []), dtype=float).reshape(-1).tolist(),
            "scales": np.asarray(vertices.get("scales", []), dtype=int).reshape(-1).tolist(),
            "radii_pixels": np.asarray(vertices.get("radii_pixels", []), dtype=float).tolist(),
            "radii_microns": np.asarray(vertices.get("radii_microns", []), dtype=float)
            .reshape(-1)
            .tolist(),
        },
        "edges": {
            "traces": [
                np.asarray(trace, dtype=float).reshape(-1, 3).tolist()
                for trace in edges.get("traces", [])
            ],
            "connections": np.asarray(edges.get("connections", []), dtype=int)
            .reshape(-1, 2)
            .tolist(),
            "energies": np.asarray(edges.get("energies", []), dtype=float).reshape(-1).tolist(),
        },
        "lumenRadiiPixels": radii_pixels.tolist(),
        "lumenRadiiMicrons": radii_microns.tolist(),
    }
    st.session_state["matlab_curator_payload_cache"] = {
        "signature": signature,
        "data": data,
    }
    return data


def _component_trigger(result: Any, name: str) -> Any | None:
    if result is None:
        return None
    if isinstance(result, Mapping):
        return result.get(name)
    return getattr(result, name, None)


def render_browser_manual_curation(results: Mapping[str, Any] | AppRunState) -> None:
    """Render the MATLAB-faithful two-stage curator and commit validated edits."""
    app_run = AppRunState.from_value(results)
    payload = app_run.to_dict()
    energy_shape = tuple(int(value) for value in np.asarray(payload["energy_data"]["energy"]).shape)
    signature = build_curation_baseline_signature(
        payload["vertices"], payload["edges"], energy_shape
    )
    baseline_vertex_count = len(np.asarray(payload["vertices"].get("positions", [])).reshape(-1, 3))
    baseline_edge_count = len(np.asarray(payload["edges"].get("connections", [])).reshape(-1, 2))
    dataset_name = app_run.dataset_name or str(st.session_state.get("dataset_name", "Current run"))

    raw_session = st.session_state.get("matlab_curator_session")
    try:
        if raw_session is None:
            session = new_curation_session(
                payload["vertices"],
                payload["edges"],
                image_shape=energy_shape,
                dataset_name=dataset_name,
            )
        else:
            session = validate_curation_session(
                raw_session,
                expected_signature=signature,
                baseline_vertex_count=baseline_vertex_count,
                baseline_edge_count=baseline_edge_count,
                image_shape=energy_shape,
            )
    except CurationSessionError:
        session = new_curation_session(
            payload["vertices"],
            payload["edges"],
            image_shape=energy_shape,
            dataset_name=dataset_name,
        )
        st.session_state.pop("matlab_curator_session", None)

    flash = st.session_state.pop("matlab_curator_flash", None)
    if flash:
        st.success(str(flash))
    st.caption(
        "MATLAB-equivalent order: curate vertices, continue to edges, then apply. "
        "Pointer and slider interactions stay in the browser; only Load, Save, and Apply "
        "cross the Python boundary."
    )
    if app_run.read_only or st.session_state.get("run_read_only", False):
        st.info(
            "This reopened run is read-only on disk. Curation changes the browser session and "
            "can be exported, but source checkpoints are never overwritten."
        )
    with st.sidebar:
        st.subheader("Curator session")
        st.caption(f"Dataset · {dataset_name}")
        st.caption(f"Baseline · {signature[:12]}")
        st.caption("Vertex → Edge → Network")
        if st.button(
            "Reset curation session",
            icon=":material/restart_alt:",
            width="stretch",
        ):
            clear_manual_review(st.session_state)
            st.rerun()

    try:
        component_data = _prepare_curator_data(app_run, signature)
    except (CurationSessionError, MemoryError, ValueError) as exc:
        st.error(f"The curator could not prepare this volume: {exc}")
        return
    # Refresh the revision outside the static payload cache so an imported
    # same-baseline session is picked up immediately by React.
    component_data = {
        **component_data,
        "sessionRevision": int(st.session_state.get("matlab_curator_generation", 0)),
    }
    result = matlab_curator(
        data=component_data,
        session=session.to_dict(),
        key=f"matlab_curator_{signature}",
    )

    load_value = _component_trigger(result, "load")
    save_value = _component_trigger(result, "save")
    apply_value = _component_trigger(result, "apply")
    if load_value is not None:
        try:
            loaded = validate_curation_session(
                load_value,
                expected_signature=signature,
                baseline_vertex_count=baseline_vertex_count,
                baseline_edge_count=baseline_edge_count,
                image_shape=energy_shape,
            )
        except CurationSessionError as exc:
            st.error(f"Curation file rejected: {exc}")
        else:
            st.session_state["matlab_curator_session"] = loaded.to_dict()
            st.session_state["matlab_curator_generation"] = (
                int(st.session_state.get("matlab_curator_generation", 0)) + 1
            )
            st.session_state["matlab_curator_flash"] = "Curation session loaded and validated."
            st.rerun()

    commit_value = apply_value if apply_value is not None else save_value
    if commit_value is not None:
        try:
            committed = validate_curation_session(
                commit_value,
                expected_signature=signature,
                baseline_vertex_count=baseline_vertex_count,
                baseline_edge_count=baseline_edge_count,
                image_shape=energy_shape,
            )
            curated_vertices, curated_edges = materialize_curation_session(
                payload["vertices"], payload["edges"], committed
            )
            baseline, current = apply_curated_results(
                st.session_state,
                curated_vertices,
                curated_edges,
                curation_mode="MATLAB-faithful browser curator",
            )
        except (CurationSessionError, TypeError, ValueError) as exc:
            st.error(f"Curation could not be applied: {exc}")
        else:
            clear_manual_review(st.session_state)
            action = "saved and applied" if save_value is not None else "applied"
            st.session_state["matlab_curator_flash"] = (
                f"Curation {action}. Network rebuilt: "
                f"{baseline['Vertices']} → {current['Vertices']} vertices, "
                f"{baseline['Edges']} → {current['Edges']} edges."
            )
            st.rerun()


__all__ = [
    "build_manual_review_figure",
    "clear_manual_review",
    "render_browser_manual_curation",
    "render_manual_empty_state",
]
