"""Unit tests for the developer parity experiment runner validation and params."""

from __future__ import annotations

import importlib

import pytest
from dev.tests.support.run_state_builders import (
    materialize_checkpoint_surface,
    materialize_run_snapshot,
)

from .helpers import (
    _build_source_run_root,
    _materialize_exact_matlab_batch,
    _write_json,
)

parity_experiment = importlib.import_module("dev.scripts.cli.parity_experiment")


def test_validate_source_run_surface_accepts_required_artifacts(tmp_path):
    run_root = _build_source_run_root(tmp_path)

    surface = parity_experiment.validate_source_run_surface(run_root)

    assert surface.run_root == run_root.resolve()
    assert surface.checkpoints_dir == run_root.resolve() / parity_experiment.CHECKPOINTS_DIR
    assert surface.comparison_report_path.is_file()
    assert surface.validated_params_path.is_file()
    assert surface.run_snapshot_path is None


def test_validate_source_run_surface_reports_missing_artifacts(tmp_path):
    run_root = tmp_path / "source-run"
    run_root.mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValueError, match="missing required artifacts"):
        parity_experiment.validate_source_run_surface(run_root)


def test_resolve_input_file_uses_run_snapshot_provenance(tmp_path):
    repo_root = tmp_path / "repo"
    input_file = repo_root / "data" / "slavv_test_volume.tif"
    input_file.parent.mkdir(parents=True, exist_ok=True)
    input_file.write_bytes(b"tiff")

    run_root = _build_source_run_root(tmp_path)
    materialize_run_snapshot(
        run_root,
        {
            "run_id": "run-1",
            "provenance": {
                "input_file": "data/slavv_test_volume.tif",
            },
        },
    )
    surface = parity_experiment.validate_source_run_surface(run_root)

    resolved = parity_experiment.resolve_input_file(
        surface,
        None,
        repo_root=repo_root,
    )

    assert resolved == input_file.resolve()


def test_resolve_input_file_requires_snapshot_or_explicit_input(tmp_path):
    run_root = _build_source_run_root(tmp_path)
    surface = parity_experiment.validate_source_run_surface(run_root)

    with pytest.raises(ValueError, match="no --input was provided"):
        parity_experiment.resolve_input_file(surface, None)


def test_load_params_file_uses_source_default_and_override(tmp_path):
    run_root = _build_source_run_root(tmp_path)
    surface = parity_experiment.validate_source_run_surface(run_root)
    override = _write_json(tmp_path / "override_params.json", {"edge_method": "tracing"})

    default_params = parity_experiment.load_params_file(surface, None)
    override_params = parity_experiment.load_params_file(surface, str(override))

    assert default_params == {"number_of_edges_per_vertex": 4}
    assert override_params == {"edge_method": "tracing"}


def test_load_params_file_rejects_python_only_parity_controls_on_exact_route(tmp_path):
    run_root = _build_source_run_root(tmp_path)
    surface = parity_experiment.validate_source_run_surface(run_root)
    override = _write_json(
        tmp_path / "override_params.json",
        {
            "comparison_exact_network": True,
            "edge_method": "tracing",
            "energy_method": "hessian",
            "direction_method": "hessian",
            "energy_projection_mode": "matlab",
            "discrete_tracing": False,
            "parity_watershed_candidate_mode": "all_contacts",
        },
    )

    with pytest.raises(ValueError, match="disallowed Python-only parity keys"):
        parity_experiment.load_params_file(surface, str(override))


def test_validate_exact_proof_source_surface_accepts_exact_compatible_artifacts(tmp_path):
    run_root = _build_source_run_root(tmp_path)
    _materialize_exact_matlab_batch(run_root)
    _write_json(
        run_root / "99_Metadata" / "validated_params.json",
        {"comparison_exact_network": True},
    )
    materialize_checkpoint_surface(
        run_root,
        stages=("energy",),
        payloads={"energy": {"energy_origin": "python_native_hessian"}},
    )

    surface = parity_experiment.validate_exact_proof_source_surface(run_root)

    assert surface.run_root == run_root.resolve()
    assert surface.oracle_surface.oracle_root.parent.name == "oracles"
    assert surface.oracle_surface.manifest_path is not None
    assert surface.oracle_surface.manifest_path.is_file()
    assert surface.matlab_batch_dir.name == "batch_260421-151654"
    assert set(surface.matlab_vector_paths) == {"energy", "vertices", "edges", "network"}


def test_validate_exact_proof_source_surface_requires_exact_route_gate(tmp_path):
    run_root = _build_source_run_root(tmp_path)
    _materialize_exact_matlab_batch(run_root)
    materialize_checkpoint_surface(
        run_root,
        stages=("energy",),
        payloads={"energy": {"energy_origin": "python_native_hessian"}},
    )

    with pytest.raises(ValueError, match="comparison_exact_network"):
        parity_experiment.validate_exact_proof_source_surface(run_root)


def test_validate_exact_proof_source_surface_requires_exact_compatible_energy_origin(tmp_path):
    run_root = _build_source_run_root(tmp_path)
    _materialize_exact_matlab_batch(run_root)
    _write_json(
        run_root / "99_Metadata" / "validated_params.json",
        {"comparison_exact_network": True},
    )
    materialize_checkpoint_surface(
        run_root,
        stages=("energy",),
        payloads={"energy": {"energy_origin": "matlab_batch_hdf5"}},
    )

    with pytest.raises(ValueError, match="exact-compatible"):
        parity_experiment.validate_exact_proof_source_surface(run_root)


def test_load_exact_params_file_rejects_python_only_parity_controls(tmp_path):
    run_root = _build_source_run_root(tmp_path)
    _materialize_exact_matlab_batch(run_root)
    _write_json(
        run_root / "99_Metadata" / "validated_params.json",
        {
            "comparison_exact_network": True,
            "edge_method": "tracing",
            "energy_method": "hessian",
            "direction_method": "hessian",
            "energy_projection_mode": "matlab",
            "discrete_tracing": False,
            "parity_candidate_salvage_mode": "auto",
        },
    )
    materialize_checkpoint_surface(
        run_root,
        stages=("energy",),
        payloads={"energy": {"energy_origin": "python_native_hessian"}},
    )

    surface = parity_experiment.validate_exact_proof_source_surface(run_root)

    with pytest.raises(ValueError, match="disallowed Python-only parity keys"):
        parity_experiment.load_exact_params_file(surface)
