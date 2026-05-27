"""Public vertex extraction facade for SLAVV."""

from __future__ import annotations

from slavv_python.processing.stages.vertices.extraction import extract_vertices
from slavv_python.processing.stages.vertices.manager import VertexManager
from slavv_python.processing.stages.vertices.painting import (
    paint_vertex_center_image,
    paint_vertex_image,
)
from slavv_python.processing.stages.vertices.resumable import extract_vertices_resumable

__all__ = [
    "VertexManager",
    "extract_vertices",
    "extract_vertices_resumable",
    "paint_vertex_center_image",
    "paint_vertex_image",
]
