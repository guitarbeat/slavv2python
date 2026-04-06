from __future__ import annotations

import importlib.util
import json
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


def test_cli_requires_input_when_not_in_standalone_mode(cli_module, monkeypatch, capsys):
    _run_cli(monkeypatch, ["--skip-matlab"])

    assert cli_module.main() == 2
    assert (
        "--input is required unless standalone comparison directories are provided"
        in capsys.readouterr().out
    )


def test_cli_requires_both_standalone_directories(cli_module, tmp_path, monkeypatch, capsys):
    matlab_dir = tmp_path / "matlab_results"
    matlab_dir.mkdir()
    _run_cli(monkeypatch, ["--standalone-matlab-dir", str(matlab_dir)])

    assert cli_module.main() == 2
    assert "must be provided together" in capsys.readouterr().out


def test_cli_rejects_validate_only_in_standalone_mode(cli_module, tmp_path, monkeypatch, capsys):
    matlab_dir = tmp_path / "matlab_results"
    python_dir = tmp_path / "python_results"
    matlab_dir.mkdir()
    python_dir.mkdir()
    _run_cli(
        monkeypatch,
        [
            "--standalone-matlab-dir",
            str(matlab_dir),
            "--standalone-python-dir",
            str(python_dir),
            "--validate-only",
        ],
    )

    assert cli_module.main() == 2
    assert "standalone comparison cannot be combined" in capsys.readouterr().out


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


def test_cli_passes_comparison_depth_and_resume_latest(cli_module, tmp_path, monkeypatch):
    input_file = _write_input_file(tmp_path)
    params_file = tmp_path / "params.json"
    params_file.write_text("{}", encoding="utf-8")
    existing_run = tmp_path / "comparisons" / "20260406_120000_comparison"
    (existing_run / "99_Metadata").mkdir(parents=True)
    (existing_run / "99_Metadata" / "run_snapshot.json").write_text(
        json.dumps({"run_id": "run-1", "provenance": {"input_file": str(input_file)}}),
        encoding="utf-8",
    )
    (existing_run / "99_Metadata" / "comparison_params.normalized.json").write_text(
        json.dumps({"edge_method": "tracing"}),
        encoding="utf-8",
    )
    checkpoints_dir = existing_run / "02_Output" / "python_results" / "checkpoints"
    checkpoints_dir.mkdir(parents=True)
    (checkpoints_dir / "checkpoint_edges.pkl").write_bytes(b"checkpoint")
    observed = {}

    monkeypatch.setattr(cli_module, "load_parameters", lambda _path: {"edge_method": "tracing"})
    monkeypatch.setattr(
        cli_module,
        "list_runs",
        lambda _base_dir: [{"path": existing_run}],
    )

    def fake_orchestrate(*, output_dir, comparison_depth, **_kwargs):
        observed["output_dir"] = output_dir
        observed["comparison_depth"] = comparison_depth
        return 0

    monkeypatch.setattr(cli_module, "orchestrate_comparison", fake_orchestrate)
    _run_cli(
        monkeypatch,
        [
            "--input",
            str(input_file),
            "--params",
            str(params_file),
            "--skip-matlab",
            "--resume-latest",
            "--comparison-depth",
            "shallow",
        ],
    )

    assert cli_module.main() == 0
    assert observed["output_dir"] == existing_run
    assert observed["comparison_depth"] == "shallow"


def test_cli_resume_latest_falls_back_to_fresh_run_root_when_required_artifacts_are_missing(
    cli_module, tmp_path, monkeypatch, capsys
):
    input_file = _write_input_file(tmp_path)
    params_file = tmp_path / "params.json"
    params_file.write_text("{}", encoding="utf-8")
    comparisons_dir = tmp_path / "comparisons"
    existing_run = comparisons_dir / "20260406_120000_comparison"
    fresh_run = comparisons_dir / "20260406_121500_comparison"
    (existing_run / "99_Metadata").mkdir(parents=True)
    (existing_run / "99_Metadata" / "run_snapshot.json").write_text(
        json.dumps({"run_id": "run-1", "provenance": {"input_file": str(input_file)}}),
        encoding="utf-8",
    )
    (existing_run / "99_Metadata" / "comparison_params.normalized.json").write_text(
        json.dumps({"edge_method": "tracing"}),
        encoding="utf-8",
    )
    observed = {}

    monkeypatch.setattr(cli_module, "load_parameters", lambda _path: {"edge_method": "tracing"})
    monkeypatch.setattr(cli_module, "list_runs", lambda _base_dir: [{"path": existing_run}])
    monkeypatch.setattr(cli_module, "_build_fresh_output_dir", lambda _base_dir=None: fresh_run)

    def fake_orchestrate(*, output_dir, **_kwargs):
        observed["output_dir"] = output_dir
        return 0

    monkeypatch.setattr(cli_module, "orchestrate_comparison", fake_orchestrate)
    _run_cli(
        monkeypatch,
        [
            "--input",
            str(input_file),
            "--params",
            str(params_file),
            "--skip-matlab",
            "--resume-latest",
            "--output-dir",
            str(comparisons_dir),
        ],
    )

    assert cli_module.main() == 0
    assert observed["output_dir"] == fresh_run
    assert "missing reusable Python checkpoints" in capsys.readouterr().out


def test_cli_resume_latest_falls_back_to_fresh_run_root_when_params_mismatch(
    cli_module, tmp_path, monkeypatch, capsys
):
    input_file = _write_input_file(tmp_path)
    params_file = tmp_path / "params.json"
    params_file.write_text("{}", encoding="utf-8")
    comparisons_dir = tmp_path / "comparisons"
    existing_run = comparisons_dir / "20260406_120000_comparison"
    fresh_run = comparisons_dir / "20260406_121500_comparison"
    (existing_run / "99_Metadata").mkdir(parents=True)
    (existing_run / "99_Metadata" / "run_snapshot.json").write_text(
        json.dumps({"run_id": "run-1", "provenance": {"input_file": str(input_file)}}),
        encoding="utf-8",
    )
    (existing_run / "99_Metadata" / "comparison_params.normalized.json").write_text(
        json.dumps({"edge_method": "different"}),
        encoding="utf-8",
    )
    observed = {}

    monkeypatch.setattr(cli_module, "load_parameters", lambda _path: {"edge_method": "tracing"})
    monkeypatch.setattr(cli_module, "list_runs", lambda _base_dir: [{"path": existing_run}])
    monkeypatch.setattr(cli_module, "_build_fresh_output_dir", lambda _base_dir=None: fresh_run)

    def fake_orchestrate(*, output_dir, **_kwargs):
        observed["output_dir"] = output_dir
        return 0

    monkeypatch.setattr(cli_module, "orchestrate_comparison", fake_orchestrate)
    _run_cli(
        monkeypatch,
        [
            "--input",
            str(input_file),
            "--params",
            str(params_file),
            "--skip-matlab",
            "--resume-latest",
            "--output-dir",
            str(comparisons_dir),
        ],
    )

    assert cli_module.main() == 0
    assert observed["output_dir"] == fresh_run
    assert "recorded comparison parameters do not match" in capsys.readouterr().out


def test_cli_resume_latest_falls_back_to_fresh_run_root_when_input_mismatches(
    cli_module, tmp_path, monkeypatch, capsys
):
    input_file = _write_input_file(tmp_path)
    params_file = tmp_path / "params.json"
    params_file.write_text("{}", encoding="utf-8")
    comparisons_dir = tmp_path / "comparisons"
    existing_run = comparisons_dir / "20260406_120000_comparison"
    fresh_run = comparisons_dir / "20260406_121500_comparison"
    (existing_run / "99_Metadata").mkdir(parents=True)
    (existing_run / "99_Metadata" / "run_snapshot.json").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "provenance": {"input_file": str(tmp_path / "different_input.tif")},
            }
        ),
        encoding="utf-8",
    )
    observed = {}

    monkeypatch.setattr(cli_module, "load_parameters", lambda _path: {"edge_method": "tracing"})
    monkeypatch.setattr(cli_module, "list_runs", lambda _base_dir: [{"path": existing_run}])
    monkeypatch.setattr(cli_module, "_build_fresh_output_dir", lambda _base_dir=None: fresh_run)

    def fake_orchestrate(*, output_dir, **_kwargs):
        observed["output_dir"] = output_dir
        return 0

    monkeypatch.setattr(cli_module, "orchestrate_comparison", fake_orchestrate)
    _run_cli(
        monkeypatch,
        [
            "--input",
            str(input_file),
            "--params",
            str(params_file),
            "--skip-matlab",
            "--resume-latest",
            "--output-dir",
            str(comparisons_dir),
        ],
    )

    assert cli_module.main() == 0
    assert observed["output_dir"] == fresh_run
    assert "not compatible with the current input" in capsys.readouterr().out


def test_cli_runs_standalone_comparison_with_explicit_python_result_source(
    cli_module, tmp_path, monkeypatch
):
    matlab_dir = tmp_path / "matlab_results"
    python_dir = tmp_path / "python_results"
    output_dir = tmp_path / "comparison_output"
    matlab_dir.mkdir()
    python_dir.mkdir()
    observed = {}

    def fake_run_standalone_comparison(
        *, matlab_dir, python_dir, output_dir, project_root, python_result_source, comparison_depth
    ):
        observed["matlab_dir"] = matlab_dir
        observed["python_dir"] = python_dir
        observed["output_dir"] = output_dir
        observed["project_root"] = project_root
        observed["python_result_source"] = python_result_source
        observed["comparison_depth"] = comparison_depth
        return 0

    monkeypatch.setattr(cli_module, "run_standalone_comparison", fake_run_standalone_comparison)
    _run_cli(
        monkeypatch,
        [
            "--standalone-matlab-dir",
            str(matlab_dir),
            "--standalone-python-dir",
            str(python_dir),
            "--output-dir",
            str(output_dir),
            "--python-result-source",
            "network-json-only",
            "--comparison-depth",
            "shallow",
        ],
    )

    assert cli_module.main() == 0
    assert observed["matlab_dir"] == matlab_dir
    assert observed["python_dir"] == python_dir
    assert observed["output_dir"] == output_dir
    assert observed["python_result_source"] == "network-json-only"
    assert observed["comparison_depth"] == "shallow"
