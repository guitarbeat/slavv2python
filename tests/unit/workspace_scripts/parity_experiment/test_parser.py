"""Unit tests for the developer parity experiment runner parser."""

from __future__ import annotations

import importlib

parity_experiment = importlib.import_module("scripts.parity_experiment")


def test_build_parser_rerun_python_defaults():
    parser = parity_experiment.build_parser()

    args = parser.parse_args(
        [
            "rerun-python",
            "--source-run-root",
            "source-run",
            "--dest-run-root",
            "dest-run",
        ]
    )

    assert args.command == "rerun-python"
    assert args.rerun_from == "edges"
    assert args.params_file is None
    assert args.input is None


def test_build_parser_prove_exact_defaults():
    parser = parity_experiment.build_parser()

    args = parser.parse_args(
        [
            "prove-exact",
            "--source-run-root",
            "source-run",
            "--dest-run-root",
            "dest-run",
        ]
    )

    assert args.command == "prove-exact"
    assert args.stage == "all"
    assert args.report_path is None
    assert args.oracle_root is None


def test_build_parser_fail_fast_defaults():
    parser = parity_experiment.build_parser()

    args = parser.parse_args(
        [
            "fail-fast",
            "--source-run-root",
            "source-run",
            "--dest-run-root",
            "dest-run",
        ]
    )

    assert args.command == "fail-fast"
    assert args.force is False
    assert args.debug_maps is False
    assert args.oracle_root is None


def test_build_parser_promote_commands():
    parser = parity_experiment.build_parser()

    dataset_args = parser.parse_args(
        [
            "promote-dataset",
            "--dataset-file",
            "input.tif",
            "--experiment-root",
            "live-parity",
        ]
    )
    oracle_args = parser.parse_args(
        [
            "promote-oracle",
            "--matlab-batch-dir",
            "matlab-batch",
            "--oracle-root",
            "oracle-root",
        ]
    )
    init_args = parser.parse_args(
        [
            "init-exact-run",
            "--dataset-root",
            "dataset-root",
            "--oracle-root",
            "oracle-root",
            "--dest-run-root",
            "dest-run",
        ]
    )
    report_args = parser.parse_args(
        [
            "promote-report",
            "--run-root",
            "run-root",
        ]
    )
    normalize_args = parser.parse_args(
        [
            "normalize-recordings",
            "--run-root",
            "run-root",
        ]
    )
    diagnose_args = parser.parse_args(
        [
            "diagnose-gaps",
            "--run-root",
            "run-root",
        ]
    )

    assert dataset_args.command == "promote-dataset"
    assert dataset_args.experiment_root == "live-parity"
    assert oracle_args.command == "promote-oracle"
    assert oracle_args.oracle_root == "oracle-root"
    assert init_args.command == "init-exact-run"
    assert init_args.stop_after == "vertices"
    assert init_args.energy_storage_format == "npy"
    assert report_args.command == "promote-report"
    assert report_args.report_root is None
    assert normalize_args.command == "normalize-recordings"
    assert normalize_args.run_root == "run-root"
    assert diagnose_args.command == "diagnose-gaps"
    assert diagnose_args.limit == 10
