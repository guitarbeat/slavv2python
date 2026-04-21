"""Tests for CLI run-command helpers."""

from __future__ import annotations

from types import SimpleNamespace

from slavv.apps.cli_run_service import (
    build_exportable_network,
    build_run_completion_lines,
    filter_export_formats,
    format_run_event_line,
    resolve_effective_run_dir,
)


def test_resolve_effective_run_dir_prefers_explicit_run_dir():
    assert (
        resolve_effective_run_dir(
            run_dir="custom-run",
            output_dir="out",
        )
        == "custom-run"
    )


def test_resolve_effective_run_dir_uses_output_when_not_legacy():
    assert (
        resolve_effective_run_dir(
            run_dir=None,
            output_dir="out",
        )
        == "out/_slavv_run"
    )


def test_format_run_event_line_includes_detail():
    line = format_run_event_line(
        SimpleNamespace(
            stage="edges",
            stage_progress=0.25,
            overall_progress=0.5,
            detail="Tracing",
        )
    )

    assert line == "[edges] stage=25.0% overall=50.0% - Tracing"


def test_build_exportable_network_normalizes_empty_arrays():
    network = build_exportable_network(
        {"vertices": {}, "edges": {}},
        network_factory=lambda **kwargs: kwargs,
    )

    assert network["vertices"].shape == (0, 3)
    assert network["edges"].shape == (0, 2)
    assert network["radii"] is None


def test_filter_export_formats_skips_when_vertices_missing():
    formats, warnings = filter_export_formats(["csv"], {"energy_data": {}})

    assert formats == []
    assert warnings == [
        "Export requested but pipeline stopped before extracting vertices. Skipping export."
    ]


def test_filter_export_formats_warns_for_partial_results():
    formats, warnings = filter_export_formats(["csv"], {"vertices": {}})

    assert formats == ["csv"]
    assert warnings == [
        "Export requested but pipeline stopped early. Formatting output with available data only."
    ]


def test_build_run_completion_lines_includes_snapshot_lines():
    lines = build_run_completion_lines(
        effective_run_dir="run-dir",
        output_dir="out",
        snapshot=object(),
        status_line_builder=lambda snapshot: ["status-a", "status-b"],
    )

    assert lines == [
        "Run directory: run-dir",
        "",
        "status-a",
        "status-b",
        "Done. Results in out",
    ]
