"""Tests for the SLAVV CLI entry point (slavv.apps.cli)."""

import pytest

from slavv.apps.cli import _args_to_parameters, _build_parser, main
from slavv.runtime import RunContext


class TestBuildParser:
    """Parser construction and basic argument parsing."""

    def test_help_flag(self, capsys):
        parser = _build_parser()
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["--help"])
        assert exc.value.code == 0
        captured = capsys.readouterr()
        assert "SLAVV" in captured.out

    def test_version_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["--version"])
        assert args.version is True

    def test_run_subcommand_requires_input(self):
        parser = _build_parser()
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["run"])
        assert exc.value.code != 0

    def test_run_subcommand_defaults(self):
        parser = _build_parser()
        args = parser.parse_args(["run", "-i", "volume.tif"])
        assert args.input == "volume.tif"
        assert args.output == "./slavv_output"
        assert args.energy_method == "hessian"
        assert args.edge_method == "tracing"
        assert args.vessel_radius == 1.5
        assert args.microns_per_voxel == [1.0, 1.0, 1.0]
        assert args.export == []
        assert args.verbose is False

    def test_run_subcommand_full_options(self):
        parser = _build_parser()
        args = parser.parse_args(
            [
                "run",
                "-i",
                "vol.tif",
                "-o",
                "out/",
                "--energy-method",
                "frangi",
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
        assert args.energy_method == "frangi"
        assert args.edge_method == "watershed"
        assert args.vessel_radius == 2.0
        assert args.microns_per_voxel == [0.5, 0.5, 1.0]
        assert args.export == ["csv", "json"]
        assert args.verbose is True

    def test_info_subcommand(self):
        parser = _build_parser()
        args = parser.parse_args(["info"])
        assert args.command == "info"

    def test_status_subcommand(self):
        parser = _build_parser()
        args = parser.parse_args(["status", "--run-dir", "run_dir"])
        assert args.command == "status"
        assert args.run_dir == "run_dir"


class TestArgsToParameters:
    """Verify CLI args convert correctly to SLAVV parameter dicts."""

    def test_default_conversion(self):
        parser = _build_parser()
        args = parser.parse_args(["run", "-i", "test.tif"])
        params = _args_to_parameters(args)
        assert params["energy_method"] == "hessian"
        assert params["edge_method"] == "tracing"
        assert params["radius_of_smallest_vessel_in_microns"] == 1.5
        assert params["microns_per_voxel"] == [1.0, 1.0, 1.0]


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
