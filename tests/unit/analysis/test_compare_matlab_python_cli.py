from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_cli_module():
    repo_root = Path(__file__).resolve().parents[3]
    module_path = repo_root / "workspace" / "scripts" / "cli" / "compare_matlab_python.py"
    spec = importlib.util.spec_from_file_location("compare_matlab_python_cli_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cli_module():
    return _load_cli_module()


def _write_input_file(tmp_path):
    input_file = tmp_path / "input.tif"
    input_file.write_bytes(b"fake")
    return input_file


def _run_cli(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["compare_matlab_python.py", *argv])


@pytest.mark.parametrize(
    ("extra_args", "expected_message"),
    [
        (["--skip-matlab", "--skip-python"], "cannot be used together"),
        ([], "--matlab-path is required unless --skip-matlab is set"),
    ],
)
def test_cli_rejects_invalid_flag_combinations(
    cli_module, tmp_path, monkeypatch, capsys, extra_args, expected_message
):
    input_file = _write_input_file(tmp_path)
    _run_cli(monkeypatch, ["--input", str(input_file), *extra_args])

    assert cli_module.main() == 2
    assert expected_message in capsys.readouterr().out


def test_cli_allows_skip_matlab_without_matlab_path(cli_module, tmp_path, monkeypatch):
    input_file = _write_input_file(tmp_path)
    params_file = tmp_path / "params.json"
    params_file.write_text("{}", encoding="utf-8")
    observed = {}

    def fake_load_parameters(path):
        observed["params_path"] = path
        return {"edge_method": "tracing"}

    def fake_orchestrate(*, input_file, output_dir, matlab_path, project_root, params, **_kwargs):
        observed["input_file"] = input_file
        observed["output_dir"] = output_dir
        observed["matlab_path"] = matlab_path
        observed["project_root"] = project_root
        observed["params"] = params
        return 0

    monkeypatch.setattr(cli_module, "load_parameters", fake_load_parameters)
    monkeypatch.setattr(cli_module, "orchestrate_comparison", fake_orchestrate)
    _run_cli(
        monkeypatch,
        [
            "--input",
            str(input_file),
            "--params",
            str(params_file),
            "--skip-matlab",
        ],
    )

    assert cli_module.main() == 0
    assert observed["params_path"] == str(params_file)
    assert observed["matlab_path"] == ""
    assert observed["params"]["edge_method"] == "tracing"


def test_cli_rejects_directory_as_input(cli_module, tmp_path, monkeypatch, capsys):
    input_dir = tmp_path / "input_dir"
    input_dir.mkdir()

    _run_cli(
        monkeypatch,
        [
            "--input",
            str(input_dir),
            "--skip-matlab",
            "--params",
            str(tmp_path / "missing.json"),
        ],
    )

    assert cli_module.main() == 1
    assert "Input file not found" in capsys.readouterr().out


def test_cli_handles_parameter_load_error(cli_module, tmp_path, monkeypatch, capsys):
    input_file = _write_input_file(tmp_path)
    params_file = tmp_path / "params.json"
    params_file.write_text("{}", encoding="utf-8")

    def fake_load_parameters(_path):
        raise ValueError("invalid json")

    monkeypatch.setattr(cli_module, "load_parameters", fake_load_parameters)
    _run_cli(
        monkeypatch,
        [
            "--input",
            str(input_file),
            "--params",
            str(params_file),
            "--skip-matlab",
        ],
    )

    assert cli_module.main() == 1
    assert "Failed to load parameters" in capsys.readouterr().out
