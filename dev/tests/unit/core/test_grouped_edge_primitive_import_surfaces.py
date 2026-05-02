from __future__ import annotations

from source.core._edge_primitives.directions import (
    generate_edge_directions as legacy_generate_edge_directions,
)
from source.core._edge_primitives.metrics import compute_gradient as legacy_compute_gradient
from source.core._edge_primitives.terminals import (
    _finalize_traced_edge as legacy_finalize_traced_edge,
)
from source.core._edge_primitives.tracing import trace_edge as legacy_trace_edge
from source.core.edge_primitives import (
    _finalize_traced_edge,
    compute_gradient,
    generate_edge_directions,
    trace_edge,
)
from source.core.edges_internal.edge_tracing import trace_edge as internal_trace_edge
from source.core.edges_internal.terminal_lookup import (
    _finalize_traced_edge as internal_finalize_traced_edge,
)
from source.core.edges_internal.trace_directions import (
    generate_edge_directions as internal_generate_edge_directions,
)
from source.core.edges_internal.trace_metrics import compute_gradient as internal_compute_gradient


def test_grouped_edge_primitive_import_surfaces_resolve_consistently():
    assert trace_edge is internal_trace_edge is legacy_trace_edge
    assert generate_edge_directions is internal_generate_edge_directions
    assert generate_edge_directions is legacy_generate_edge_directions
    assert compute_gradient is internal_compute_gradient is legacy_compute_gradient
    assert _finalize_traced_edge is internal_finalize_traced_edge is legacy_finalize_traced_edge
