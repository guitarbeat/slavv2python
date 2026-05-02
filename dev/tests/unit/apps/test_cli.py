"""Tests for the SLAVV CLI entry point (source.apps.cli)."""

import json

import numpy as np
import pytest
from dev.tests.support.network_builders import (
    build_authoritative_network_json_payload,
    build_network_object,
)
from source.apps.cli import (
    _build_cli_parser,
    _build_export_artifacts,
    _build_pipeline_parameters,
    _expand_export_formats,
    _load_exported_network_json,
    _require_existing_file,
    main,
)
from source.runtime import RunContext


class TestBuildParser:
    """Parser construction and basic argument parsing."""

    def test_help_flag(self, capsys):
        parser = _build_cli_parser()
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["--help"])
        assert exc.value.code == 0
        captured = capsys.readouterr()
        assert "SLAVV" in captured.out

    def test_version_flag(self):
        parser = _build_cli_parser()
        args = parser.parse_args(["--version"])
        assert args.version is True

    def test_run_subcommand_requires_input(self):
        parser = _build_cli_parser()
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["run"])
        assert exc.value.code != 0

    def test_run_subcommand_defaults(self):
        parser = _build_cli_parser()
        args = parser.parse_args(["run", "-i", "volume.tif"])
        assert args.input == "volume.tif"
        assert args.output == "./slavv_output"
        assert args.pipeline_profile == "paper"
        assert args.energy_storage_format is None
        assert args.energy_method is None
        assert args.energy_projection_mode is None
        assert args.edge_method is None
        assert args.vessel_radius is None
        assert args.microns_per_voxel is None
        assert args.export == []
        assert args.verbose is False

    def test_run_subcommand_full_options(self):
        parser = _build_cli_parser()
        args = parser.parse_args(
            [
                "run",
                "-i",
                "vol.tif",
                "-o",
                "out/",
                "--energy-storage-format",
                "zarr",
                "--energy-method",
                "frangi",
                "--energy-projection-mode",
                "paper",
                "--edge-method",
                "watershed",
                "--vessel-radius",
                "2.0",
                "--microns-per-voxel",
                "0.5",
                "0.5",
                "1.0",
                "--export",
                "csv",
                "json",
                "-v",
            ]
        )
        assert args.energy_storage_format == "zarr"
        assert args.energy_method == "frangi"
        assert args.energy_projection_mode == "paper"
        assert args.edge_method == "watershed"
        assert args.vessel_radius == 2.0
        assert args.microns_per_voxel == [0.5, 0.5, 1.0]
        assert args.export == ["csv", "json"]
        assert args.verbose is True

    def test_run_subcommand_accepts_simpleitk_energy_method(self):
        parser = _build_cli_parser()
        args = parser.parse_args(
            ["run", "-i", "vol.tif", "--energy-method", "simpleitk_objectness"]
        )

        assert args.energy_method == "simpleitk_objectness"

    def test_run_subcommand_accepts_cupy_energy_method(self):
        parser = _build_cli_parser()
        args = parser.parse_args(["run", "-i", "vol.tif", "--energy-method", "cupy_hessian"])

        assert args.energy_method == "cupy_hessian"

    def test_run_subcommand_accepts_paper_energy_projection_mode(self):
        parser = _build_cli_parser()
        args = parser.parse_args(["run", "-i", "vol.tif", "--energy-projection-mode", "paper"])

        assert args.energy_projection_mode == "paper"

    def test_info_subcommand(self):
        parser = _build_cli_parser()
        args = parser.parse_args(["info"])
        assert args.command == "info"

    def test_status_subcommand(self):
        parser = _build_cli_parser()
        args = parser.parse_args(["status", "--run-dir", "run_dir"])
        assert args.command == "status"
        assert args.run_dir == "run_dir"


class TestArgsToParameters:
    """Verify CLI args convert correctly to SLAVV parameter dicts."""

    def test_default_conversion(self):
        parser = _build_cli_parser()
        args = parser.parse_args(["run", "-i", "test.tif"])
        params = _build_pipeline_parameters(args)
        assert params["pipeline_profile"] == "paper"
        assert params["energy_method"] == "hessian"
        assert params["energy_projection_mode"] == "paper"
        assert params["energy_storage_format"] == "auto"
        assert params["edge_method"] == "tracing"
        assert params["radius_of_smallest_vessel_in_microns"] == 1.5
        assert params["microns_per_voxel"] == [1.0, 1.0, 1.0]


class TestCliHelpers:
    """Focused tests for shared CLI helper paths."""

    def test_expand_export_formats_preserves_explicit_selection(self):
        assert _expand_export_formats(["csv", "json"]) == ["csv", "json"]

    def test_expand_export_formats_expands_all(self):
        assert _expand_export_formats(["all"]) == ["csv", "json", "casx", "vmv", "mat"]

    def test_build_export_artifacts_skips_csv(self, tmp_path):
        artifacts = _build_export_artifacts(str(tmp_path), ["csv", "json", "mat"])

        assert artifacts == {
            "json": str(tmp_path / "network.json"),
            "mat": str(tmp_path / "network.mat"),
        }

    def test_require_existing_file_uses_consistent_error(self):
        with pytest.raises(SystemExit) as exc:
            _require_existing_file("missing-file-12345.tif", label="input file")

        assert exc.value.code == 1


class TestMainEntryPoint:
    """Integration-level tests of the main() function."""

    def test_no_args_shows_help(self, capsys):
        """Calling with no arguments prints help and exits 0."""
        with pytest.raises(SystemExit) as exc:
            main([])
        assert exc.value.code == 0
        captured = capsys.readouterr()
        assert "SLAVV" in captured.out or "usage" in captured.out.lower()

    def test_version_prints(self, capsys):
        main(["--version"])
        captured = capsys.readouterr()
        assert "slavv" in captured.out

    def test_info_prints_version(self, capsys):
        main(["info"])
        captured = capsys.readouterr()
        assert "slavv" in captured.out.lower()

    def test_run_missing_file_exits(self):
        with pytest.raises(SystemExit) as exc:
            main(["run", "-i", "nonexistent_file_12345.tif"])
        assert exc.value.code == 1

    def test_status_prints_run_snapshot(self, capsys, tmp_path):
        run_dir = tmp_path / "run"
        context = RunContext(
            run_dir=run_dir,
            input_fingerprint="input-a",
            params_fingerprint="params-a",
            target_stage="network",
        )
        context.mark_preprocess_complete()
        context.stage("energy").begin(detail="Energy running", units_total=4, units_completed=2)

        main(["status", "--run-dir", str(run_dir)])

        captured = capsys.readouterr()
        assert "Run ID:" in captured.out
        assert "Target progress:" in captured.out
        assert "energy" in captured.out

    def test_status_missing_snapshot_is_read_only(self, capsys, tmp_path):
        with pytest.raises(SystemExit) as exc:
            main(["status", "--run-dir", str(tmp_path)])

        captured = capsys.readouterr()
        assert exc.value.code == 1
        assert "no run snapshot found" in captured.err
        assert not list(tmp_path.iterdir())


def test_load_exported_network_json_preserves_parameters(tmp_path):
    path = tmp_path / "network.json"
    payload = build_authoritative_network_json_payload(
        network=build_network_object(
            vertices=[[1, 2, 3], [2, 2, 3]],
            edges=[[0, 1]],
            radii=[1.5, 1.5],
        ),
        parameters={"microns_per_voxel": [0.5, 0.5, 2.0], "pipeline_profile": "paper"}
    )
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = _load_exported_network_json(str(path))

    np.testing.assert_array_equal(result["vertices"]["positions"], np.array([[1, 2, 3], [2, 2, 3]]))
    assert result["parameters"]["microns_per_voxel"] == [0.5, 0.5, 2.0]
    assert result["network"]["strands"] == [[0, 1]]
    assert result["metadata"]["pipeline_profile"] == "paper"
    assert tuple(result["image_shape"]) == (4, 4, 4)


def test_analyze_command_prints_statistics_for_exported_json(capsys, tmp_path):
    path = tmp_path / "network.json"
    path.write_text(
        json.dumps(
            build_authoritative_network_json_payload(
                network=build_network_object(
                    vertices=[[0, 0, 0], [1, 0, 0], [2, 0, 0]],
                    edges=[[0, 1], [1, 2]],
                    radii=[1.0, 1.0, 1.0],
                ),
                parameters={"microns_per_voxel": [1.0, 1.0, 1.0], "pipeline_profile": "paper"}
            )
        ),
        encoding="utf-8",
    )

    main(["analyze", "-i", str(path)])

    captured = capsys.readouterr()
    assert "Topological Features:" in captured.out
    assert "Vertices: 3" in captured.out
    assert "Total Edge Length: 2.00 um" in captured.out


def test_plot_command_writes_html_for_authoritative_json(tmp_path):
    path = tmp_path / "network.json"
    path.write_text(
        json.dumps(build_authoritative_network_json_payload()),
        encoding="utf-8",
    )
    out_path = tmp_path / "plots.html"

    main(["plot", "-i", str(path), "-o", str(out_path)])

    assert out_path.exists()
    assert "html" in out_path.read_text(encoding="utf-8").lower()
