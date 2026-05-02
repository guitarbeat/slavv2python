from __future__ import annotations

from source.core._vertices.extraction import extract_vertices as legacy_extract_vertices
from source.core._vertices.painting import paint_vertex_image as legacy_paint_vertex_image
from source.core._vertices.payloads import build_vertices_result as legacy_build_vertices_result
from source.core.vertices import extract_vertices, paint_vertex_image
from source.core.vertices_internal.vertex_extraction import (
    extract_vertices as internal_extract_vertices,
)
from source.core.vertices_internal.vertex_painting import (
    paint_vertex_image as internal_paint_vertex_image,
)
from source.core.vertices_internal.vertex_results import (
    build_vertices_result as internal_build_vertices_result,
)


def test_grouped_vertex_import_surfaces_resolve_consistently():
    assert extract_vertices is internal_extract_vertices is legacy_extract_vertices
    assert paint_vertex_image is internal_paint_vertex_image is legacy_paint_vertex_image
    assert internal_build_vertices_result is legacy_build_vertices_result
