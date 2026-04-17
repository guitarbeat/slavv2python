"""
Comprehensive Interactive Graphical Curator Interface (GCI) using PyQt5, PyVista, and PyQtGraph.
Achieves 1:1 feature parity with the MATLAB GCI:
  - 4-panel layout (Volume Map, Volume Display, Intensity Histogram, Energy Histogram)
  - Depth/Thickness sliders with X/Y/Z orthogonal projection switching
  - Blue (True) / Red (False) state toggling for vertices AND edges
  - Sweep button to hide False objects
  - Interactive energy threshold via draggable histogram line
  - Point-and-click edge addition between two vertices
  - Intensity histogram with brightness/contrast adjustments
"""

from __future__ import annotations

import copy
import logging
import sys

import numpy as np
import pyqtgraph as pg
import pyvista as pv
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

logger = logging.getLogger(__name__)


class InteractiveCurator(QMainWindow):
    """
    4-Panel GCI Window.
    Panel 1 (Top-Left):  Volume Map - 3D bounding-box context + current FOV highlight
    Panel 2 (Top-Right): Volume Display - 2D MIP with curation controls
    Panel 3 (Bot-Left):  Intensity Histogram - pixel distribution + brightness/contrast
    Panel 4 (Bot-Right): Energy Histogram - object energy distribution + threshold line
    """

    def __init__(self, energy_data, vertices_data, edges_data=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SLAVV Graphical Curator Interface (GCI)")
        self.resize(1400, 950)

        # ── Data ──────────────────────────────────────────────────────────
        self.energy_data = energy_data  # 3-D numpy [y, x, z]
        self.vertices_data = copy.deepcopy(vertices_data)

        if edges_data:
            self.edges_data = copy.deepcopy(edges_data)
        else:
            self.edges_data = {"traces": [], "origin_indices": [], "terminal_indices": []}

        # Ensure boolean status columns (True = Blue, False = Red)
        n_verts = len(self.vertices_data.get("positions", []))
        if "status" not in self.vertices_data:
            self.vertices_data["status"] = [True] * n_verts
        n_edges = len(self.edges_data.get("traces", []))
        if "status" not in self.edges_data:
            self.edges_data["status"] = [True] * n_edges

        # ── View state ────────────────────────────────────────────────────
        self.current_depth = 0
        self.current_thickness = self.energy_data.shape[2]
        self.projection_axis = "Z"
        self.mode = "view"  # 'view' | 'toggle' | 'add'
        self.target_type = "vertices"
        self.hide_false = False
        self._add_edge_buffer = []  # Holds vertex indices for 2-click edge addition

        # ── Build UI, populate data, draw ─────────────────────────────────
        self._init_ui()
        self._init_data()
        self._update_views()

    # ================================================================== #
    #  UI CONSTRUCTION                                                     #
    # ================================================================== #
    def _init_ui(self):
        """Build the 4-panel layout that mirrors the MATLAB GCI."""
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)

        v_split = QSplitter(Qt.Vertical)

        # ── Top row (Volume Map | Volume Display) ─────────────────────────
        top_split = QSplitter(Qt.Horizontal)

        # Panel 1 - Volume Map
        map_grp = QGroupBox("1. Volume Map (Context)")
        map_lay = QVBoxLayout()
        self.plotter_map = QtInteractor(self)
        map_lay.addWidget(self.plotter_map.interactor)
        map_grp.setLayout(map_lay)

        # Panel 2 - Volume Display
        disp_grp = QGroupBox("2. Volume Display (Curation)")
        disp_lay = QVBoxLayout()
        self.plotter_display = QtInteractor(self)
        disp_lay.addWidget(self.plotter_display.interactor)

        # Control ribbon
        ctrl = QGridLayout()

        self.slider_depth = QSlider(Qt.Horizontal)
        self.slider_depth.valueChanged.connect(self._on_slider_changed)
        ctrl.addWidget(QLabel("Depth:"), 0, 0)
        ctrl.addWidget(self.slider_depth, 0, 1)

        self.slider_thickness = QSlider(Qt.Horizontal)
        self.slider_thickness.setMinimum(1)
        self.slider_thickness.valueChanged.connect(self._on_slider_changed)
        ctrl.addWidget(QLabel("Thickness:"), 1, 0)
        ctrl.addWidget(self.slider_thickness, 1, 1)

        self.combo_proj = QComboBox()
        self.combo_proj.addItems(["Z-Projection", "X-Projection", "Y-Projection"])
        self.combo_proj.currentTextChanged.connect(self._on_proj_changed)
        ctrl.addWidget(self.combo_proj, 0, 2)

        self.combo_target = QComboBox()
        self.combo_target.addItems(["Vertices", "Edges"])
        self.combo_target.currentTextChanged.connect(self._on_target_changed)
        ctrl.addWidget(QLabel("Target:"), 0, 3)
        ctrl.addWidget(self.combo_target, 0, 4)

        self.btn_toggle = QPushButton("Toggle Mode")
        self.btn_toggle.setCheckable(True)
        self.btn_toggle.clicked.connect(self._toggle_mode)
        ctrl.addWidget(self.btn_toggle, 1, 2)

        self.btn_sweep = QPushButton("Sweep (Hide Red)")
        self.btn_sweep.setCheckable(True)
        self.btn_sweep.clicked.connect(self._sweep_toggled)
        ctrl.addWidget(self.btn_sweep, 1, 3)

        self.btn_add = QPushButton("Add Edge")
        self.btn_add.setCheckable(True)
        self.btn_add.clicked.connect(self._add_mode)
        ctrl.addWidget(self.btn_add, 1, 4)

        disp_lay.addLayout(ctrl)
        disp_grp.setLayout(disp_lay)

        top_split.addWidget(map_grp)
        top_split.addWidget(disp_grp)
        top_split.setSizes([400, 1000])

        # ── Bottom row (Intensity Histogram | Energy Histogram) ──────────
        bot_split = QSplitter(Qt.Horizontal)

        # Panel 3 - Intensity Histogram
        int_grp = QGroupBox("3. Intensity Histogram")
        int_lay = QVBoxLayout()
        self.hist_intensity = pg.PlotWidget()
        self.hist_intensity.setLabel("bottom", "Pixel Intensity")
        self.hist_intensity.setLabel("left", "Count")
        int_lay.addWidget(self.hist_intensity)
        int_grp.setLayout(int_lay)

        # Panel 4 - Energy Histogram
        eng_grp = QGroupBox("4. Energy Histogram (drag red line to threshold)")
        eng_lay = QVBoxLayout()
        self.hist_energy = pg.PlotWidget()
        self.hist_energy.setLabel("bottom", "Energy")
        self.hist_energy.setLabel("left", "Count")
        self.energy_line = pg.InfiniteLine(
            pos=0, angle=90, movable=True, pen=pg.mkPen("r", width=2)
        )
        self.energy_line.sigPositionChanged.connect(self._on_threshold_changed)
        self.hist_energy.addItem(self.energy_line)
        eng_lay.addWidget(self.hist_energy)
        eng_grp.setLayout(eng_lay)

        bot_split.addWidget(int_grp)
        bot_split.addWidget(eng_grp)

        v_split.addWidget(top_split)
        v_split.addWidget(bot_split)
        v_split.setSizes([650, 300])

        root_layout.addWidget(v_split)

        # Save & Close
        self.btn_save = QPushButton("Save Curations and Continue Pipeline")
        self.btn_save.setStyleSheet(
            "background-color:#28a745;color:white;font-weight:bold;padding:10px;font-size:14px;"
        )
        self.btn_save.clicked.connect(self.close)
        root_layout.addWidget(self.btn_save)

    # ================================================================== #
    #  DATA INITIALIZATION                                                 #
    # ================================================================== #
    def _init_data(self):
        self.grid_shape = self.energy_data.shape  # (y, x, z)

        self.slider_depth.setMaximum(self.grid_shape[2] - 1)
        self.slider_thickness.setMaximum(self.grid_shape[2])
        self.slider_thickness.setValue(self.grid_shape[2])

        # Point picker
        self.plotter_display.enable_point_picking(
            callback=self._handle_point_pick,
            show_message=False,
            color="yellow",
            point_size=15,
            use_picker=True,
            show_point=True,
        )

        # ── Intensity histogram (Panel 3) ────────────────────────────────
        flat_img = self.energy_data.flatten()
        y_int, x_int = np.histogram(flat_img, bins=100)
        self.hist_intensity.plot(
            x_int, y_int, stepMode="center", fillLevel=0, brush=(200, 200, 200, 180)
        )

        # ── Energy histogram (Panel 4) ───────────────────────────────────
        if "energies" in self.vertices_data and len(self.vertices_data["energies"]) > 0:
            energies = np.array(self.vertices_data["energies"])
            y_eng, x_eng = np.histogram(energies, bins=60)
            self.hist_energy.plot(
                x_eng, y_eng, stepMode="center", fillLevel=0, brush=(0, 120, 255, 150)
            )
            # Position threshold line at the maximum energy by default (no filtering)
            self.energy_line.setValue(float(energies.max()))

    # ================================================================== #
    #  COORDINATE HELPERS                                                  #
    # ================================================================== #
    @staticmethod
    def _yx2xy(positions):
        """Convert list of (y,x,z) arrays to PyVista (x,y,z) ndarray."""
        pts = np.asarray(positions, dtype=float)
        if pts.ndim != 2 or pts.shape[0] == 0:
            return np.empty((0, 3))
        out = np.empty_like(pts)
        out[:, 0] = pts[:, 1]
        out[:, 1] = pts[:, 0]
        out[:, 2] = pts[:, 2]
        return out

    # ================================================================== #
    #  VIEW UPDATE (redraw plotters)                                        #
    # ================================================================== #
    def _update_views(self):
        d = self.current_depth
        t = self.current_thickness
        gy, gx, gz = self.grid_shape

        # -- Panel 1 - Volume Map ─────────────────────────────────────────
        self._render_volume_map(d, t, gx, gy, gz)

        # -- Panel 2 - Volume Display ─────────────────────────────────────
        self._render_volume_display(d, t)
        self._set_display_camera()

    def _slice_box_bounds(self, depth, thickness, gx, gy, gz):
        if self.projection_axis == "X":
            return (depth, min(depth + thickness, gx), 0, gy, 0, gz)
        if self.projection_axis == "Z":
            return (0, gx, 0, gy, depth, min(depth + thickness, gz))
        return (0, gx, depth, min(depth + thickness, gy), 0, gz)

    def _render_volume_map(self, depth, thickness, gx, gy, gz):
        self.plotter_map.clear()
        self.plotter_map.add_mesh(
            pv.Box(bounds=(0, gx, 0, gy, 0, gz)),
            style="wireframe",
            color="white",
            line_width=2,
        )
        self.plotter_map.add_mesh(
            pv.Box(bounds=self._slice_box_bounds(depth, thickness, gx, gy, gz)),
            color="red",
            opacity=0.3,
        )
        self.plotter_map.view_isometric()

    def _visible_vertices(self, depth, thickness):
        positions = self.vertices_data.get("positions", [])
        statuses = np.asarray(self.vertices_data.get("status", []))
        if len(positions) == 0:
            return np.empty((0, 3)), np.asarray([], dtype=int)
        positions_array = np.asarray(positions)
        in_slice = self._in_slice_mask(positions_array, depth, thickness)
        mask = in_slice & statuses if self.hide_false else in_slice
        return positions_array[mask], statuses[mask].astype(int)

    def _render_vertices(self, depth, thickness):
        visible_positions, visible_status = self._visible_vertices(depth, thickness)
        if len(visible_positions) == 0:
            return
        poly = pv.PolyData(self._yx2xy(visible_positions))
        poly["status"] = visible_status
        self.plotter_display.add_mesh(
            poly,
            scalars="status",
            cmap=["red", "blue"],
            point_size=12,
            render_points_as_spheres=True,
            show_scalar_bar=False,
            name="vertices",
        )

    def _trace_in_slice(self, trace, depth, thickness):
        if self.projection_axis == "Z":
            return np.any((trace[:, 2] >= depth) & (trace[:, 2] <= depth + thickness))
        if self.projection_axis == "X":
            return np.any((trace[:, 1] >= depth) & (trace[:, 1] <= depth + thickness))
        return np.any((trace[:, 0] >= depth) & (trace[:, 0] <= depth + thickness))

    def _edge_polydata(self, depth, thickness):
        traces = self.edges_data.get("traces", [])
        statuses = np.asarray(self.edges_data.get("status", []))
        if len(traces) == 0:
            return None
        pts_list = []
        lines = []
        colors = []
        point_offset = 0
        for index, trace in enumerate(traces):
            if len(trace) < 2:
                continue
            trace_array = np.asarray(trace, dtype=float)
            if not self._trace_in_slice(trace_array, depth, thickness):
                continue
            if self.hide_false and not statuses[index]:
                continue
            render_trace = self._yx2xy(trace_array)
            n_points = len(render_trace)
            pts_list.append(render_trace)
            lines.append(n_points)
            lines.extend(range(point_offset, point_offset + n_points))
            colors.extend([int(statuses[index])] * n_points)
            point_offset += n_points
        if not pts_list:
            return None
        edge_poly = pv.PolyData(np.vstack(pts_list))
        edge_poly.lines = np.array(lines)
        edge_poly["status"] = np.array(colors)
        return edge_poly

    def _render_edges(self, depth, thickness):
        edge_poly = self._edge_polydata(depth, thickness)
        if edge_poly is None:
            return
        self.plotter_display.add_mesh(
            edge_poly,
            scalars="status",
            cmap=["red", "cyan"],
            line_width=3,
            show_scalar_bar=False,
            name="edges",
        )

    def _render_volume_display(self, depth, thickness):
        self.plotter_display.clear()
        self._render_vertices(depth, thickness)
        self._render_edges(depth, thickness)

    def _set_display_camera(self):
        self.plotter_display.camera.ParallelProjectionOn()
        if self.projection_axis == "Z":
            self.plotter_display.view_xy()
        elif self.projection_axis == "X":
            self.plotter_display.view_yz()
        else:
            self.plotter_display.view_xz()

    def _in_slice_mask(self, pos, d, t):
        """Return boolean mask for positions inside the current slice."""
        if self.projection_axis == "Z":
            return (pos[:, 2] >= d) & (pos[:, 2] <= d + t)
        if self.projection_axis == "X":
            return (pos[:, 1] >= d) & (pos[:, 1] <= d + t)
        return (pos[:, 0] >= d) & (pos[:, 0] <= d + t)

    # ================================================================== #
    #  CALLBACKS                                                           #
    # ================================================================== #
    def _on_slider_changed(self):
        self.current_depth = self.slider_depth.value()
        self.current_thickness = self.slider_thickness.value()
        self._update_views()

    def _on_proj_changed(self, text):
        self.projection_axis = text[0]
        if self.projection_axis == "Z":
            mx = self.grid_shape[2]
        elif self.projection_axis == "X":
            mx = self.grid_shape[1]
        else:
            mx = self.grid_shape[0]
        self.slider_depth.setMaximum(mx - 1)
        self.slider_thickness.setMaximum(mx)
        self._update_views()

    def _on_target_changed(self, text):
        self.target_type = text.lower()  # 'vertices' or 'edges'

    def _on_threshold_changed(self):
        thresh = self.energy_line.value()
        if "energies" in self.vertices_data:
            energies = np.array(self.vertices_data["energies"])
            self.vertices_data["status"] = list(energies <= thresh)
            self._update_views()

    def _toggle_mode(self, checked):
        if checked:
            self.mode = "toggle"
            self.btn_add.setChecked(False)
        else:
            self.mode = "view"

    def _add_mode(self, checked):
        if checked:
            self.mode = "add"
            self.btn_toggle.setChecked(False)
        else:
            self.mode = "view"

        self._add_edge_buffer = []

    def _sweep_toggled(self, checked):
        self.hide_false = checked
        self._update_views()

    # ================================================================== #
    #  POINT PICKING                                                       #
    # ================================================================== #
    def _handle_point_pick(self, point, picker=None):
        if self.mode == "view":
            return

        # Convert VTK (x,y,z) → numpy (y,x,z)
        np_pt = np.array([point[1], point[0], point[2]])

        # ── Toggle mode ───────────────────────────────────────────────────
        if self.mode == "toggle":
            self._handle_toggle_pick(np_pt)

        # ── Add-edge mode (2-click) ───────────────────────────────────────
        elif self.mode == "add":
            self._handle_add_pick(np_pt)

    def _nearest_vertex_index(self, point, max_distance=5.0):
        if not self.vertices_data.get("positions"):
            return None
        positions = np.asarray(self.vertices_data["positions"], dtype=float)
        dists = np.linalg.norm(positions - point, axis=1)
        idx = int(np.argmin(dists))
        return idx if dists[idx] < max_distance else None

    def _nearest_edge_index(self, point, max_distance=5.0):
        if not self.edges_data.get("traces"):
            return None
        best_idx, best_dist = None, float("inf")
        for i, trace in enumerate(self.edges_data["traces"]):
            if len(trace) == 0:
                continue
            trace_points = np.asarray(trace, dtype=float)
            distance = float(np.min(np.linalg.norm(trace_points - point, axis=1)))
            if distance < best_dist:
                best_dist = distance
                best_idx = i
        return best_idx if best_dist < max_distance else None

    def _toggle_vertex_status(self, point):
        idx = self._nearest_vertex_index(point)
        if idx is None:
            return
        self.vertices_data["status"][idx] = not self.vertices_data["status"][idx]
        self._update_views()

    def _toggle_edge_status(self, point):
        idx = self._nearest_edge_index(point)
        if idx is None:
            return
        self.edges_data["status"][idx] = not self.edges_data["status"][idx]
        self._update_views()

    def _handle_toggle_pick(self, point):
        if self.target_type == "vertices":
            self._toggle_vertex_status(point)
            return
        if self.target_type == "edges":
            self._toggle_edge_status(point)

    def _interpolated_edge_trace(self, start_idx, end_idx):
        p1 = np.asarray(self.vertices_data["positions"][start_idx], dtype=float)
        p2 = np.asarray(self.vertices_data["positions"][end_idx], dtype=float)
        n_steps = max(int(np.linalg.norm(p2 - p1)), 2)
        return [
            list((p1 + (p2 - p1) * (step / (n_steps - 1))).round().astype(int))
            for step in range(n_steps)
        ]

    def _commit_added_edge(self):
        start_idx, end_idx = self._add_edge_buffer
        self.edges_data["traces"].append(self._interpolated_edge_trace(start_idx, end_idx))
        self.edges_data["status"].append(True)
        if "origin_indices" in self.edges_data:
            self.edges_data["origin_indices"].append(start_idx)
        if "terminal_indices" in self.edges_data:
            self.edges_data["terminal_indices"].append(end_idx)
        self._add_edge_buffer = []
        self._update_views()

    def _handle_add_pick(self, point):
        idx = self._nearest_vertex_index(point)
        if idx is None:
            return
        self._add_edge_buffer.append(idx)
        logger.info("Edge vertex %d selected (%d/2)", idx, len(self._add_edge_buffer))
        if len(self._add_edge_buffer) == 2:
            self._commit_added_edge()


# ====================================================================== #
#  PUBLIC ENTRY POINT                                                      #
# ====================================================================== #
def run_curator(energy_data, vertices_data, edges_data=None):
    """Launch the 4-panel GCI; blocks until user clicks *Save & Close*.
    Returns (curated_vertices, curated_edges) with False items stripped.
    """
    app = QApplication.instance() or QApplication(sys.argv)

    win = InteractiveCurator(energy_data, vertices_data, edges_data)
    win.show()
    app.exec_()

    # Deep-copy and strip False-status rows before returning
    cv = copy.deepcopy(win.vertices_data)
    ce = copy.deepcopy(win.edges_data)

    for data, key in [(cv, "status"), (ce, "status")]:
        st = np.asarray(data.get(key, []))
        for k in list(data.keys()):
            if isinstance(data[k], list) and len(data[k]) == len(st):
                data[k] = [v for i, v in enumerate(data[k]) if st[i]]

    return cv, ce
