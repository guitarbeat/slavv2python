"""Focused app-facing helpers for exports and share reports."""

from __future__ import annotations

import os
import tempfile
import zipfile
from typing import TYPE_CHECKING, Any, cast

import streamlit as st

from .share_report import build_share_report_html, record_share_event
from source.apps.state.processing import build_processing_run_dir
from source.models import normalize_pipeline_result
from source.runtime import RunContext
from source.visualization import NetworkVisualizer

if TYPE_CHECKING:
    from collections.abc import Mapping


@st.cache_data(show_spinner=False)
def generate_export_data(vertices, edges, network, parameters, format_type):
    """Generate export data and return as bytes."""
    results = normalize_pipeline_result(
        {
            "vertices": vertices,
            "edges": edges,
            "network": network,
            "parameters": parameters,
        }
    ).to_dict()
    visualizer = NetworkVisualizer()

    with tempfile.TemporaryDirectory() as temp_dir:
        base_path = os.path.join(temp_dir, "export")

        if format_type == "csv":
            visualizer.export_network_data(results, base_path, format="csv")
            v_path = f"{base_path}_vertices.csv"
            e_path = f"{base_path}_edges.csv"

            zip_path = os.path.join(temp_dir, "network_csv.zip")
            with zipfile.ZipFile(zip_path, "w") as zf:
                if os.path.exists(v_path):
                    zf.write(v_path, "vertices.csv")
                if os.path.exists(e_path):
                    zf.write(e_path, "edges.csv")

            if os.path.exists(zip_path):
                with open(zip_path, "rb") as file_handle:
                    return file_handle.read()
            return None

        file_path = f"{base_path}.{format_type}"
        visualizer.export_network_data(results, file_path, format=format_type)

        if os.path.exists(file_path):
            with open(file_path, "rb") as file_handle:
                return file_handle.read()
        return None


def has_full_network_results(results: Mapping[str, Any]) -> bool:
    """Return True when a full network exists and exports can be offered."""
    typed_result = normalize_pipeline_result(results)
    return (
        typed_result.vertices is not None
        and typed_result.edges is not None
        and typed_result.network is not None
    )


@st.cache_data(show_spinner=False)
def generate_share_report_data(
    processing_results: Mapping[str, Any],
    dataset_name: str,
    image_shape: tuple[int, int, int],
) -> str:
    """Generate a self-contained HTML share report."""
    typed_result = normalize_pipeline_result(processing_results)
    return cast(
        "str",
        build_share_report_html(
            typed_result.to_dict(),
            dataset_name=dataset_name,
            image_shape=image_shape,
        ),
    )


def log_share_report_prepared_once(dataset_name, report_data, results):
    """Track report preparation exactly once per report signature in a session."""
    signature = report_data["signature"]
    if st.session_state.get("share_report_prepared_signature") == signature:
        return

    st.session_state["share_report_prepared_signature"] = signature
    typed_result = normalize_pipeline_result(results)
    normalized_results = typed_result.to_dict()
    record_share_event(
        st.session_state,
        "share_report_requested",
        dataset_name,
        signature,
        extra={
            "vertices_count": len(normalized_results["vertices"].get("positions", [])),
            "edges_count": len(normalized_results["edges"].get("traces", [])),
            "strands_count": len(normalized_results["network"].get("strands", [])),
        },
    )


def build_run_task_dir(upload_bytes: bytes, validated_params: dict[str, object]) -> str:
    """Compatibility wrapper for the processing-page run-dir helper."""
    return build_processing_run_dir(upload_bytes, validated_params)


def update_run_task(
    run_dir: str | None,
    task_name: str,
    *,
    status: str,
    detail: str,
    artifacts: dict[str, str] | None = None,
) -> None:
    """Attach optional task progress to the active run."""
    if not run_dir:
        return
    context = RunContext.from_existing(run_dir)
    context.update_optional_task(
        task_name,
        status=status,
        detail=detail,
        artifacts=artifacts,
    )


__all__ = [
    "build_run_task_dir",
    "generate_export_data",
    "generate_share_report_data",
    "has_full_network_results",
    "log_share_report_prepared_once",
    "update_run_task",
]
