"""Tests for maintenance scripts under ``workspace/scripts/maintenance``."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module(relative_path: str, module_name: str):
    repo_root = Path(__file__).resolve().parents[3]
    module_path = repo_root / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_extract_mapped_names_only_reads_table_entries():
    module = _load_module(
        "workspace/scripts/maintenance/check_mapped.py", "check_mapped_script_test"
    )
    content = """
Intro text mentions `foo.m` but this is not a table row.

| MATLAB File | Port Status | Notes |
| --- | --- | --- |
| `bar.m` | `Ported` | Something |
"""

    mapped = module._extract_mapped_names(content)

    assert mapped == {"bar.m"}


def test_check_mapped_targets_reference_mapping_doc():
    module = _load_module(
        "workspace/scripts/maintenance/check_mapped.py", "check_mapped_script_path_test"
    )
    repo_root = Path(__file__).resolve().parents[3]

    assert repo_root / "docs" / "reference" / "MATLAB_MAPPING.md" == module.MAPPING_PATH


def test_check_mapped_writes_appendix_to_configured_mapping_path(tmp_path):
    module = _load_module(
        "workspace/scripts/maintenance/check_mapped.py", "check_mapped_script_main_test"
    )
    mapping_path = tmp_path / "docs" / "reference" / "MATLAB_MAPPING.md"
    matlab_dir = tmp_path / "external" / "Vectorization-Public" / "source"
    mapping_path.parent.mkdir(parents=True)
    matlab_dir.mkdir(parents=True)
    mapping_path.write_text(
        "\n".join(
            [
                "# Mapping",
                "",
                "| MATLAB File | Port Status | Notes |",
                "| --- | --- | --- |",
                "| `ported.m` | `Ported` | Already covered |",
            ]
        ),
        encoding="utf-8",
    )
    (matlab_dir / "ported.m").write_text("function out = ported()\nout = 1;\n", encoding="utf-8")
    (matlab_dir / "extra_script.m").write_text("disp('hi');\n", encoding="utf-8")
    module.MAPPING_PATH = mapping_path
    module.MATLAB_DIR = matlab_dir

    module.main()

    updated = mapping_path.read_text(encoding="utf-8")
    assert "## Appendix: Unmapped MATLAB Files" in updated
    assert "| `extra_script.m` | `Skipped` | Upstream-only or archival helper |" in updated
    assert updated.count("`ported.m`") == 1


def test_is_matlab_function_detects_case_and_spacing_variants(tmp_path):
    module = _load_module(
        "workspace/scripts/maintenance/find_matlab_scripts.py",
        "find_matlab_scripts_test",
    )
    function_file = tmp_path / "function_variant.m"
    function_file.write_text(
        "% comment line\n   FUNCTION   [out] = my_fn(x)\nout = x;\n",
        encoding="utf-8",
    )
    script_file = tmp_path / "plain_script.m"
    script_file.write_text("x = 42;\n", encoding="utf-8")

    assert module._is_matlab_function(function_file) is True
    assert module._is_matlab_function(script_file) is False
