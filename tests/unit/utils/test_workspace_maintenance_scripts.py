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
