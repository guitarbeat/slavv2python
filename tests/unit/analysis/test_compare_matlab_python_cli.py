from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_cli_module():
    repo_root = Path(__file__).resolve().parents[3]
    module_path = repo_root / "workspace" / "scripts" / "cli" / "compare_matlab_python.py"
    spec = importlib.util.spec_from_file_location("compare_matlab_python_cli_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_cli_rejects_both_skip_flags(tmp_path, monkeypatch, capsys):
    module = _load_cli_module()
    input_file = tmp_path / "input.tif"
    input_file.write_bytes(b"fake")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_matlab_python.py",
            "--input",
            str(input_file),
            "--skip-matlab",
            "--skip-python",
        ],
    )

    assert module.main() == 2
    assert "cannot be used together" in capsys.readouterr().out


def test_cli_requires_matlab_path_unless_skipping_matlab(tmp_path, monkeypatch, capsys):
    module = _load_cli_module()
    input_file = tmp_path / "input.tif"
    input_file.write_bytes(b"fake")

    monkeypatch.setattr(sys, "argv", ["compare_matlab_python.py", "--input", str(input_file)])

    assert module.main() == 2
    assert "--matlab-path is required unless --skip-matlab is set" in capsys.readouterr().out


def test_cli_allows_skip_matlab_without_matlab_path(tmp_path, monkeypatch):
    module = _load_cli_module()
    input_file = tmp_path / "input.tif"
    input_file.write_bytes(b"fake")
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

    monkeypatch.setattr(module, "load_parameters", fake_load_parameters)
    monkeypatch.setattr(module, "orchestrate_comparison", fake_orchestrate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_matlab_python.py",
            "--input",
            str(input_file),
            "--params",
            str(params_file),
            "--skip-matlab",
        ],
    )

    assert module.main() == 0
    assert observed["params_path"] == str(params_file)
    assert observed["matlab_path"] == ""
    assert observed["params"]["edge_method"] == "tracing"


def test_cli_rejects_directory_as_input(tmp_path, monkeypatch, capsys):
    module = _load_cli_module()
    input_dir = tmp_path / "input_dir"
    input_dir.mkdir()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_matlab_python.py",
            "--input",
            str(input_dir),
            "--skip-matlab",
            "--params",
            str(tmp_path / "missing.json"),
        ],
    )

    assert module.main() == 1
    assert "Input file not found" in capsys.readouterr().out


def test_cli_handles_parameter_load_error(tmp_path, monkeypatch, capsys):
    module = _load_cli_module()
    input_file = tmp_path / "input.tif"
    input_file.write_bytes(b"fake")
    params_file = tmp_path / "params.json"
    params_file.write_text("{}", encoding="utf-8")

    def fake_load_parameters(_path):
        raise ValueError("invalid json")

    monkeypatch.setattr(module, "load_parameters", fake_load_parameters)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_matlab_python.py",
            "--input",
            str(input_file),
            "--params",
            str(params_file),
            "--skip-matlab",
        ],
    )

    assert module.main() == 1
    assert "Failed to load parameters" in capsys.readouterr().out
