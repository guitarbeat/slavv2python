"""Export and share-report helpers for the Streamlit web app."""

from __future__ import annotations

import hashlib
import os
import tempfile
import zipfile
from typing import TYPE_CHECKING, Any, cast

import streamlit as st

from slavv.apps.share_report import build_share_report_html, record_share_event
from slavv.runtime import RunContext
from slavv.runtime.run_state import fingerprint_jsonable
from slavv.visualization import NetworkVisualizer

if TYPE_CHECKING:
    from collections.abc import Mapping


@st.cache_data(show_spinner=False)
def generate_export_data(vertices, edges, network, parameters, format_type):
    """Generate export data and return as bytes."""
    results = {"vertices": vertices, "edges": edges, "network": network, "parameters": parameters}
    visualizer = NetworkVisualizer()

    with tempfile.TemporaryDirectory() as temp_dir:
        base_path = os.path.join(temp_dir, "export")

        if format_type == "csv":
            visualizer.export_network_data(results, base_path, format="csv")
            v_path = base_path + "_vertices.csv"
            e_path = base_path + "_edges.csv"

            zip_path = os.path.join(temp_dir, "network_csv.zip")
            with zipfile.ZipFile(zip_path, "w") as zf:
                if os.path.exists(v_path):
                    zf.write(v_path, "vertices.csv")
                if os.path.exists(e_path):
                    zf.write(e_path, "edges.csv")

            if os.path.exists(zip_path):
                with open(zip_path, "rb") as f:
                    return f.read()
            return None

        file_path = base_path + "." + format_type
        visualizer.export_network_data(results, file_path, format=format_type)

        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                return f.read()
        return None


def _has_full_network_results(results: Mapping[str, Any]) -> bool:
    """Return True when a full network exists and exports can be offered."""
    required_keys = {"vertices", "edges", "network"}
    return required_keys.issubset(results)


@st.cache_data(show_spinner=False)
def generate_share_report_data(
    processing_results: Mapping[str, Any],
    dataset_name: str,
    image_shape: tuple[int, int, int],
) -> str:
    """Generate a self-contained HTML share report."""
    return cast(
        "str",
        build_share_report_html(
            processing_results,
            dataset_name=dataset_name,
            image_shape=image_shape,
        ),
    )


def _log_share_report_prepared_once(dataset_name, report_data, results):
    """Track report preparation exactly once per report signature in a session."""
    signature = report_data["signature"]
    if st.session_state.get("share_report_prepared_signature") == signature:
        return

    st.session_state["share_report_prepared_signature"] = signature
    record_share_event(
        st.session_state,
        "share_report_requested",
        dataset_name,
        signature,
        extra={
            "vertices_count": len(results["vertices"].get("positions", [])),
            "edges_count": len(results["edges"].get("traces", [])),
            "strands_count": len(results["network"].get("strands", [])),
        },
    )


def _build_processing_run_dir(upload_bytes: bytes, validated_params: dict[str, object]) -> str:
    """Return a stable run directory per input file and validated parameter set."""
    file_hash = hashlib.md5(upload_bytes).hexdigest()[:12]
    params_hash = fingerprint_jsonable(validated_params)[:12]
    return os.path.join(tempfile.gettempdir(), "slavv_runs", f"{file_hash}_{params_hash}")


def _update_run_task(
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
