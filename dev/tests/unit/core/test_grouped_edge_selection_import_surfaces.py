from __future__ import annotations

from source.core._edge_selection.cleanup import (
    clean_edges_cycles_python as legacy_clean_edges_cycles_python,
)
from source.core._edge_selection.conflict_painting import (
    _choose_edges_matlab_style as legacy_choose_edges_matlab_style,
)
from source.core._edge_selection.payloads import (
    build_selected_edges_result as legacy_build_selected_edges_result,
)
from source.core.edge_selection import _choose_edges_matlab_style
from source.core.edges_internal.edge_cleanup import clean_edges_cycles_python
from source.core.edges_internal.edge_selection import (
    _choose_edges_matlab_style as internal_choose_edges_matlab_style,
)
from source.core.edges_internal.edge_selection_payloads import (
    build_selected_edges_result as internal_build_selected_edges_result,
)


def test_grouped_edge_selection_import_surfaces_resolve_consistently():
    assert _choose_edges_matlab_style is internal_choose_edges_matlab_style
    assert _choose_edges_matlab_style is legacy_choose_edges_matlab_style
    assert clean_edges_cycles_python is legacy_clean_edges_cycles_python
    assert internal_build_selected_edges_result is legacy_build_selected_edges_result
