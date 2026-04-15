"""Tests for ``dev/scripts/maintenance/find_matlab_script_files.py``."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_workspace_module(relative_path: str, module_name: str):
    repo_root = Path(__file__).resolve().parents[4]
    module_path = repo_root / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_is_matlab_function_detects_case_and_spacing_variants(tmp_path):
    module = _load_workspace_module(
        "dev/scripts/maintenance/find_matlab_script_files.py",
        "find_matlab_script_files_function_test",
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


def test_find_matlab_script_files_scans_all_configured_roots(tmp_path):
    module = _load_workspace_module(
        "dev/scripts/maintenance/find_matlab_script_files.py",
        "find_matlab_script_files_scan_test",
    )
    upstream_root = tmp_path / "external" / "Vectorization-Public"
    workspace_root = tmp_path / "workspace"
    upstream_root.mkdir(parents=True)
    workspace_root.mkdir(parents=True)

    (upstream_root / "script_a.m").write_text("disp('upstream script');\n", encoding="utf-8")
    (workspace_root / "script_b.m").write_text("x = 42;\n", encoding="utf-8")
    (workspace_root / "function_c.m").write_text(
        "function out = function_c()\nout = 1;\n",
        encoding="utf-8",
    )
    module.MATLAB_SEARCH_ROOTS = (upstream_root, workspace_root)

    result = module.find_matlab_script_files()

    assert [path.name for path in result] == ["script_a.m", "script_b.m"]


def test_main_prints_relative_script_paths(tmp_path, capsys):
    module = _load_workspace_module(
        "dev/scripts/maintenance/find_matlab_script_files.py",
        "find_matlab_script_files_main_test",
    )
    module.REPO_ROOT = tmp_path
    module.MATLAB_SEARCH_ROOTS = (tmp_path / "dev",)
    module.MATLAB_SEARCH_ROOTS[0].mkdir(parents=True)
    (module.MATLAB_SEARCH_ROOTS[0] / "script_only.m").write_text("disp('hi');\n", encoding="utf-8")

    module.main()

    captured = capsys.readouterr()
    assert "--- MATLAB SCRIPTS FOUND ---" in captured.out
    assert "dev\\script_only.m" in captured.out or "dev/script_only.m" in captured.out
