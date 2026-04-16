"""
Experimental napari-based curator prototype.

This module keeps the same high-level contract as the existing desktop curator:
launch a blocking review surface and return curated vertex/edge dictionaries
with rejected rows stripped out after the viewer closes.
"""

from __future__ import annotations

import copy
import importlib
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_NAPARI_INSTALL_HINT = (
    "napari is required for the experimental napari curator. "
    "Install it with `pip install -e \".[napari]\"`, `pip install slavv[napari]`, "
    "or `pip install \"napari[all]>=0.5.0\"`."
)


def _load_napari_modules() -> dict[str, Any]:
    try:
        napari = importlib.import_module("napari")
        qt_widgets = importlib.import_module("qtpy.QtWidgets")
    except Exception as exc:  # pragma: no cover - exercised through runtime error surface
        raise RuntimeError(_NAPARI_INSTALL_HINT) from exc
    return {
        "napari": napari,
        "QCheckBox": qt_widgets.QCheckBox,
        "QDoubleSpinBox": qt_widgets.QDoubleSpinBox,
        "QGroupBox": qt_widgets.QGroupBox,
        "QHBoxLayout": qt_widgets.QHBoxLayout,
        "QLabel": qt_widgets.QLabel,
        "QPushButton": qt_widgets.QPushButton,
        "QVBoxLayout": qt_widgets.QVBoxLayout,
        "QWidget": qt_widgets.QWidget,
    }


def _prepare_curator_payload(
    vertices_data: dict[str, Any],
    edges_data: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    vertices_copy = copy.deepcopy(vertices_data)
    edges_copy = copy.deepcopy(edges_data) if edges_data else {}

    n_vertices = len(vertices_copy.get("positions", []))
    vertices_copy.setdefault("status", [True] * n_vertices)

    if not edges_copy:
        edges_copy = {"traces": [], "origin_indices": [], "terminal_indices": []}
    n_edges = len(edges_copy.get("traces", []))
    edges_copy.setdefault("status", [True] * n_edges)

    return vertices_copy, edges_copy


def _strip_false_status_rows(data: dict[str, Any]) -> dict[str, Any]:
    curated = copy.deepcopy(data)
    statuses = np.asarray(curated.get("status", []), dtype=bool)
    if statuses.size == 0:
        return curated
    for key in list(curated.keys()):
        values = curated[key]
        if isinstance(values, list) and len(values) == len(statuses):
            curated[key] = [value for idx, value in enumerate(values) if bool(statuses[idx])]
    return curated


def _rgba(
    statuses: np.ndarray,
    *,
    true_rgb: tuple[float, float, float],
    false_rgb: tuple[float, float, float],
    hide_false: bool,
) -> np.ndarray:
    colors = np.zeros((len(statuses), 4), dtype=float)
    for idx, keep in enumerate(statuses.astype(bool)):
        if keep:
            colors[idx, :3] = true_rgb
            colors[idx, 3] = 1.0
        else:
            colors[idx, :3] = false_rgb
            colors[idx, 3] = 0.0 if hide_false else 0.85
    return colors


def _build_linear_trace(p1: np.ndarray, p2: np.ndarray) -> list[list[int]]:
    n_steps = max(int(np.linalg.norm(p2 - p1)), 2)
    return [
        list((p1 + (p2 - p1) * (step / (n_steps - 1))).round().astype(int))
        for step in range(n_steps)
    ]


class NapariCuratorPrototype:
    """Experimental napari curator focused on point/edge review workflow."""

    def __init__(
        self,
        energy_data: np.ndarray,
        vertices_data: dict[str, Any],
        edges_data: dict[str, Any],
        modules: dict[str, Any],
    ) -> None:
        self.energy_data = energy_data
        self.vertices_data = vertices_data
        self.edges_data = edges_data
        self.modules = modules
        self.hide_false = False

        self.viewer = modules["napari"].Viewer(title="SLAVV Curator (napari prototype)")
        if getattr(self.viewer, "dims", None) is not None and self.energy_data.ndim == 3:
            self.viewer.dims.ndisplay = 3

        self.viewer.add_image(
            self.energy_data,
            name="Energy",
            rendering="mip",
            colormap="gray",
        )

        self.vertex_layer = None
        self.edge_layer = None

        self._build_controls()
        self._refresh_layers()

    def _build_controls(self) -> None:
        QWidget = self.modules["QWidget"]
        QVBoxLayout = self.modules["QVBoxLayout"]
        QHBoxLayout = self.modules["QHBoxLayout"]
        QGroupBox = self.modules["QGroupBox"]
        QLabel = self.modules["QLabel"]
        QPushButton = self.modules["QPushButton"]
        QCheckBox = self.modules["QCheckBox"]
        QDoubleSpinBox = self.modules["QDoubleSpinBox"]

        panel = QWidget()
        layout = QVBoxLayout(panel)

        instructions = QLabel(
            "Select vertices or edges in napari, then use the buttons below to "
            "toggle status, threshold vertices, or add an edge from two selected vertices."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        vertex_group = QGroupBox("Vertices")
        vertex_layout = QVBoxLayout(vertex_group)
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setDecimals(4)
        self.threshold_spin.setRange(-1_000_000.0, 1_000_000.0)
        energies = np.asarray(self.vertices_data.get("energies", []), dtype=float)
        if energies.size:
            self.threshold_spin.setRange(float(np.min(energies)), float(np.max(energies)))
            self.threshold_spin.setValue(float(np.max(energies)))
        threshold_row = QWidget()
        threshold_layout = QHBoxLayout(threshold_row)
        threshold_layout.addWidget(QLabel("Energy threshold"))
        threshold_layout.addWidget(self.threshold_spin)
        threshold_apply = QPushButton("Apply threshold")
        threshold_apply.clicked.connect(self._apply_threshold)
        threshold_layout.addWidget(threshold_apply)
        vertex_layout.addWidget(threshold_row)

        toggle_vertices = QPushButton("Toggle selected vertices")
        toggle_vertices.clicked.connect(self._toggle_selected_vertices)
        vertex_layout.addWidget(toggle_vertices)
        layout.addWidget(vertex_group)

        edge_group = QGroupBox("Edges")
        edge_layout = QVBoxLayout(edge_group)
        toggle_edges = QPushButton("Toggle selected edges")
        toggle_edges.clicked.connect(self._toggle_selected_edges)
        edge_layout.addWidget(toggle_edges)
        add_edge = QPushButton("Add edge from 2 selected vertices")
        add_edge.clicked.connect(self._add_edge_from_selection)
        edge_layout.addWidget(add_edge)
        layout.addWidget(edge_group)

        self.hide_false_checkbox = QCheckBox("Hide rejected vertices and edges")
        self.hide_false_checkbox.stateChanged.connect(self._toggle_hide_false)
        layout.addWidget(self.hide_false_checkbox)

        save_button = QPushButton("Save Curations and Close")
        save_button.clicked.connect(self._save_and_close)
        layout.addWidget(save_button)

        layout.addStretch(1)
        self.viewer.window.add_dock_widget(panel, area="right", name="Curator Controls")

    def _remove_layer_if_present(self, name: str) -> None:
        if name in self.viewer.layers:
            self.viewer.layers.remove(name)

    def _refresh_layers(self) -> None:
        self._remove_layer_if_present("Vertices")
        self._remove_layer_if_present("Edges")

        positions = np.asarray(self.vertices_data.get("positions", []), dtype=float)
        vertex_status = np.asarray(self.vertices_data.get("status", []), dtype=bool)
        vertex_colors = _rgba(
            vertex_status,
            true_rgb=(0.1, 0.4, 1.0),
            false_rgb=(0.9, 0.1, 0.1),
            hide_false=self.hide_false,
        )
        self.vertex_layer = self.viewer.add_points(
            positions,
            name="Vertices",
            size=8,
            face_color=vertex_colors,
            edge_color="white",
            properties={
                "vertex_index": np.arange(len(positions), dtype=int),
                "status": vertex_status.astype(int),
            },
        )

        traces = [np.asarray(trace, dtype=float) for trace in self.edges_data.get("traces", [])]
        edge_status = np.asarray(self.edges_data.get("status", []), dtype=bool)
        edge_colors = _rgba(
            edge_status,
            true_rgb=(0.0, 0.8, 1.0),
            false_rgb=(1.0, 0.2, 0.2),
            hide_false=self.hide_false,
        )
        self.edge_layer = self.viewer.add_shapes(
            traces,
            shape_type="path",
            name="Edges",
            edge_color=edge_colors,
            edge_width=3,
            face_color=[0.0, 0.0, 0.0, 0.0],
            properties={
                "edge_index": np.arange(len(traces), dtype=int),
                "status": edge_status.astype(int),
            },
        )

    def _toggle_selected_vertices(self) -> None:
        for idx in sorted(self.vertex_layer.selected_data):
            self.vertices_data["status"][int(idx)] = not self.vertices_data["status"][int(idx)]
        self._refresh_layers()

    def _toggle_selected_edges(self) -> None:
        for idx in sorted(self.edge_layer.selected_data):
            self.edges_data["status"][int(idx)] = not self.edges_data["status"][int(idx)]
        self._refresh_layers()

    def _apply_threshold(self) -> None:
        energies = np.asarray(self.vertices_data.get("energies", []), dtype=float)
        if energies.size == 0:
            self.viewer.status = "No vertex energies available for thresholding."
            return
        threshold = float(self.threshold_spin.value())
        self.vertices_data["status"] = list(energies <= threshold)
        self._refresh_layers()

    def _toggle_hide_false(self) -> None:
        self.hide_false = bool(self.hide_false_checkbox.isChecked())
        self._refresh_layers()

    def _add_edge_from_selection(self) -> None:
        selected = sorted(int(idx) for idx in self.vertex_layer.selected_data)
        if len(selected) != 2:
            self.viewer.status = "Select exactly two vertices to add an edge."
            return
        v1, v2 = selected
        p1 = np.asarray(self.vertices_data["positions"][v1], dtype=float)
        p2 = np.asarray(self.vertices_data["positions"][v2], dtype=float)
        trace = _build_linear_trace(p1, p2)
        self.edges_data.setdefault("traces", []).append(trace)
        self.edges_data.setdefault("status", []).append(True)
        self.edges_data.setdefault("origin_indices", []).append(v1)
        self.edges_data.setdefault("terminal_indices", []).append(v2)
        self.viewer.status = f"Added edge between vertices {v1} and {v2}."
        self._refresh_layers()

    def _save_and_close(self) -> None:
        self.viewer.window.close()


def run_curator_napari(
    energy_data: np.ndarray,
    vertices_data: dict[str, Any],
    edges_data: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    modules = _load_napari_modules()
    vertices_copy, edges_copy = _prepare_curator_payload(vertices_data, edges_data)
    controller = NapariCuratorPrototype(energy_data, vertices_copy, edges_copy, modules)
    modules["napari"].run()
    return _strip_false_status_rows(controller.vertices_data), _strip_false_status_rows(
        controller.edges_data
    )
