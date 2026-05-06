"""Tests for initialization commands in the parity experiment runner."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from source.analysis.parity.constants import (
    CHECKPOINTS_DIR,
    EXPERIMENT_PROVENANCE_PATH,
    METADATA_DIR,
    RUN_MANIFEST_PATH,
    RUN_SNAPSHOT_PATH,
    VALIDATED_PARAMS_PATH,
)

from .support import _build_experiment_root, _materialize_exact_matlab_batch

parity_experiment = importlib.import_module("dev.scripts.cli.parity_experiment")


@pytest.mark.integration
def test_init_exact_run_bootstraps_source_surface_from_dataset_and_oracle(tmp_path, monkeypatch):
    experiment_root = _build_experiment_root(tmp_path)
    dataset_file = tmp_path / "input.tif"
    dataset_file.write_bytes(b"tiff-payload")
    dataset_hash = parity_experiment.fingerprint_file(dataset_file)
    dataset_root = experiment_root / "datasets" / dataset_hash

    parity_experiment.main(
        [
            "promote-dataset",
            "--dataset-file",
            str(dataset_file),
            "--experiment-root",
            str(experiment_root),
        ]
    )

    matlab_batch_dir = (
        tmp_path / "matlab-source" / "01_Input" / "matlab_results" / "batch_260421-151654"
    )
    _materialize_exact_matlab_batch(tmp_path / "matlab-source")
    settings_dir = matlab_batch_dir / "settings"
    settings_dir.mkdir(parents=True, exist_ok=True)
    savemat(
        settings_dir / "energy_260421-151654.mat",
        {
            "microns_per_voxel": np.array([0.5, 0.5, 1.0], dtype=np.float64),
            "radius_of_smallest_vessel_in_microns": 1.5,
            "radius_of_largest_vessel_in_microns": 40.0,
            "sample_index_of_refraction": 1.33,
            "numerical_aperture": 0.95,
            "excitation_wavelength_in_microns": 0.95,
            "scales_per_octave": 6,
            "max_voxels_per_node_energy": 1000000,
            "gaussian_to_ideal_ratio": 0.5,
            "spherical_to_annular_ratio": 0.5,
            "approximating_PSF": 1,
        },
    )
    savemat(
        settings_dir / "vertices_260421.mat",
        {
            "space_strel_apothem": 1,
            "energy_upper_bound": 0,
            "max_voxels_per_node": 6000,
            "length_dilation_ratio": 1,
        },
    )
    savemat(
        settings_dir / "edges_260421.mat",
        {
            "max_edge_length_per_origin_radius": 30,
            "space_strel_apothem_edges": 1,
            "number_of_edges_per_vertex": 4,
        },
    )
    savemat(
        settings_dir / "network_260421.mat",
        {
            "sigma_strand_smoothing": 1,
        },
    )

    oracle_root = experiment_root / "oracles" / "oracle-a"
    parity_experiment.main(
        [
            "promote-oracle",
            "--matlab-batch-dir",
            str(matlab_batch_dir),
            "--oracle-root",
            str(oracle_root),
            "--dataset-file",
            str(dataset_file),
            "--oracle-id",
            "oracle-a",
        ]
    )

    def _fake_load_tiff(_path):
        return np.ones((2, 2, 2), dtype=np.uint16)

    class FakeProcessor:
        def process_image(self, image, parameters, *, run_dir=None, stop_after=None, **_kwargs):
            assert tuple(image.shape) == (2, 2, 2)
            assert parameters["comparison_exact_network"] is True
            checkpoint_dir = Path(run_dir) / CHECKPOINTS_DIR
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            from joblib import dump

            dump(
                {
                    "energy_origin": "python_native_hessian",
                    "energy": np.zeros((2, 2, 2), dtype=np.float32),
                    "scale_indices": np.zeros((2, 2, 2), dtype=np.int16),
                    "lumen_radius_microns": np.array([1.0], dtype=np.float32),
                    "lumen_radius_pixels_axes": np.ones((1, 3), dtype=np.float32),
                },
                checkpoint_dir / "checkpoint_energy.pkl",
            )
            if stop_after == "vertices":
                dump(
                    {
                        "positions": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
                        "scales": np.array([0], dtype=np.int16),
                        "energies": np.array([-1.0], dtype=np.float32),
                        "count": 1,
                    },
                    checkpoint_dir / "checkpoint_vertices.pkl",
                )
            return {"parameters": parameters}

    monkeypatch.setattr("source.io.tiff.load_tiff_volume", _fake_load_tiff)
    monkeypatch.setattr("source.core.pipeline.SLAVVProcessor", FakeProcessor)

    dest_run_root = experiment_root / "runs" / "seed-a"
    parity_experiment.main(
        [
            "init-exact-run",
            "--dataset-root",
            str(dataset_root),
            "--oracle-root",
            str(oracle_root),
            "--dest-run-root",
            str(dest_run_root),
        ]
    )

    params = json.loads((dest_run_root / VALIDATED_PARAMS_PATH).read_text(encoding="utf-8"))
    assert params["comparison_exact_network"] is True
    assert params["microns_per_voxel"] == [0.5, 0.5, 1.0]
    assert params["step_size_per_origin_radius"] == 1.0
    assert params["max_edge_energy"] == 0.0
    assert params["edge_number_tolerance"] == 2
    assert params["distance_tolerance"] == 3.0
    assert params["radius_tolerance"] == 0.5
    assert params["energy_tolerance"] == 1.0
    assert (dest_run_root / "00_Refs" / "dataset_manifest.json").is_file()
    assert (dest_run_root / "00_Refs" / "oracle_manifest.json").is_file()
    provenance = json.loads(
        (dest_run_root / EXPERIMENT_PROVENANCE_PATH).read_text(encoding="utf-8")
    )
    assert provenance["dataset_hash"] == dataset_hash
    assert provenance["oracle_id"] == "oracle-a"
    assert provenance["oracle_size_of_image"] == [2, 2, 2]
    assert provenance["input_axis_permutation"] is None
    run_manifest = json.loads((dest_run_root / RUN_MANIFEST_PATH).read_text(encoding="utf-8"))
    assert run_manifest["command"] == "init-exact-run"
    assert run_manifest["oracle_id"] == "oracle-a"
    surface = parity_experiment.validate_exact_proof_source_surface(dest_run_root)
    assert surface.oracle_surface.oracle_root == oracle_root.resolve()


@pytest.mark.integration
def test_init_exact_run_reorients_input_volume_to_match_oracle_energy_shape(tmp_path, monkeypatch):
    experiment_root = _build_experiment_root(tmp_path)
    dataset_file = tmp_path / "input.tif"
    dataset_file.write_bytes(b"tiff")
    dataset_hash = parity_experiment._materialize_dataset_record(
        experiment_root,
        dataset_hash=None,
        dataset_file=dataset_file,
    )
    dataset_root = experiment_root / "datasets" / dataset_hash
    matlab_batch_dir = (
        tmp_path / "matlab-source" / "01_Input" / "matlab_results" / "batch_260421-151654"
    )
    matlab_batch_dir.parent.mkdir(parents=True, exist_ok=True)
    _materialize_exact_matlab_batch(tmp_path / "matlab-source")
    settings_dir = matlab_batch_dir / "settings"
    settings_dir.mkdir(parents=True, exist_ok=True)
    savemat(
        settings_dir / "energy_260421.mat",
        {
            "microns_per_voxel": np.array([0.5, 0.5, 1.0], dtype=np.float64),
            "radius_of_smallest_vessel_in_microns": 1.0,
            "radius_of_largest_vessel_in_microns": 2.0,
            "sample_index_of_refraction": 1.33,
            "numerical_aperture": 0.95,
            "scales_per_octave": 1.0,
            "max_voxels_per_node_energy": 1000000,
            "gaussian_to_ideal_ratio": 0.5,
            "spherical_to_annular_ratio": 0.5,
            "approximating_PSF": 0,
            "pixels_per_sigma_PSF": np.array([1.0, 1.0, 1.0], dtype=np.float64),
            "lumen_radius_in_microns_range": np.array([1.0], dtype=np.float64),
            "lumen_radius_in_pixels_range": np.array([[2.0, 2.0, 1.0]], dtype=np.float64),
            "excitation_wavelength_in_microns": 0.9,
        },
    )
    savemat(
        settings_dir / "vertices_260421.mat",
        {
            "space_strel_apothem": 1,
            "energy_upper_bound": 0,
            "max_voxels_per_node": 6000,
            "length_dilation_ratio": 1,
        },
    )
    savemat(
        settings_dir / "edges_260421.mat",
        {
            "max_edge_length_per_origin_radius": 30,
            "space_strel_apothem_edges": 1,
            "number_of_edges_per_vertex": 4,
        },
    )
    savemat(
        settings_dir / "network_260421.mat",
        {
            "sigma_strand_smoothing": 1,
        },
    )
    savemat(
        matlab_batch_dir / "data" / "energy_260421.mat",
        {
            "size_of_image": np.array([4, 5, 3], dtype=np.uint16),
            "intensity_limits": np.array([0, 1], dtype=np.uint16),
            "energy_runtime_in_seconds": np.array([1.0], dtype=np.float64),
        },
    )
    oracle_root = experiment_root / "oracles" / "oracle-a"
    parity_experiment.main(
        [
            "promote-oracle",
            "--matlab-batch-dir",
            str(matlab_batch_dir),
            "--oracle-root",
            str(oracle_root),
            "--dataset-file",
            str(dataset_file),
            "--oracle-id",
            "oracle-a",
        ]
    )

    def _fake_load_tiff(_path):
        return np.ones((3, 4, 5), dtype=np.uint16)

    class FakeProcessor:
        def process_image(self, image, parameters, *, run_dir=None, stop_after=None, **_kwargs):
            assert tuple(image.shape) == (4, 5, 3)
            checkpoint_dir = Path(run_dir) / CHECKPOINTS_DIR
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            from joblib import dump

            dump(
                {
                    "energy_origin": "python_native_hessian",
                    "energy": np.zeros((4, 5, 3), dtype=np.float32),
                    "scale_indices": np.zeros((4, 5, 3), dtype=np.int16),
                    "lumen_radius_microns": np.array([1.0], dtype=np.float32),
                    "lumen_radius_pixels_axes": np.ones((1, 3), dtype=np.float32),
                },
                checkpoint_dir / "checkpoint_energy.pkl",
            )
            return {"parameters": parameters}

    monkeypatch.setattr("source.io.tiff.load_tiff_volume", _fake_load_tiff)
    monkeypatch.setattr("source.core.pipeline.SLAVVProcessor", FakeProcessor)

    dest_run_root = experiment_root / "runs" / "seed-b"
    parity_experiment.main(
        [
            "init-exact-run",
            "--dataset-root",
            str(dataset_root),
            "--oracle-root",
            str(oracle_root),
            "--dest-run-root",
            str(dest_run_root),
            "--stop-after",
            "energy",
        ]
    )

    provenance = json.loads(
        (dest_run_root / EXPERIMENT_PROVENANCE_PATH).read_text(encoding="utf-8")
    )
    assert provenance["oracle_size_of_image"] == [4, 5, 3]
    assert provenance["input_axis_permutation"] == [1, 2, 0]


@pytest.mark.integration
def test_init_exact_run_can_finalize_existing_completed_seed(tmp_path, monkeypatch):
    experiment_root = _build_experiment_root(tmp_path)
    dataset_file = tmp_path / "input.tif"
    dataset_file.write_bytes(b"tiff")
    dataset_hash = parity_experiment._materialize_dataset_record(
        experiment_root,
        dataset_hash=None,
        dataset_file=dataset_file,
    )
    dataset_root = experiment_root / "datasets" / dataset_hash
    matlab_batch_dir = (
        tmp_path / "matlab-source" / "01_Input" / "matlab_results" / "batch_260421-151654"
    )
    matlab_batch_dir.parent.mkdir(parents=True, exist_ok=True)
    _materialize_exact_matlab_batch(tmp_path / "matlab-source")
    settings_dir = matlab_batch_dir / "settings"
    settings_dir.mkdir(parents=True, exist_ok=True)
    savemat(
        settings_dir / "energy_260421.mat",
        {
            "microns_per_voxel": np.array([0.5, 0.5, 1.0], dtype=np.float64),
            "radius_of_smallest_vessel_in_microns": 1.5,
            "radius_of_largest_vessel_in_microns": 40.0,
            "sample_index_of_refraction": 1.33,
            "numerical_aperture": 0.95,
            "excitation_wavelength_in_microns": 0.95,
            "scales_per_octave": 6,
            "max_voxels_per_node_energy": 1000000,
            "gaussian_to_ideal_ratio": 0.5,
            "spherical_to_annular_ratio": 0.5,
            "approximating_PSF": 1,
        },
    )
    savemat(
        settings_dir / "vertices_260421.mat",
        {
            "space_strel_apothem": 1,
            "energy_upper_bound": 0,
            "max_voxels_per_node": 6000,
            "length_dilation_ratio": 1,
        },
    )
    savemat(
        settings_dir / "edges_260421.mat",
        {
            "max_edge_length_per_origin_radius": 30,
            "space_strel_apothem_edges": 1,
            "number_of_edges_per_vertex": 4,
        },
    )
    savemat(settings_dir / "network_260421.mat", {"sigma_strand_smoothing": 1})
    oracle_root = experiment_root / "oracles" / "oracle-a"
    parity_experiment.main(
        [
            "promote-oracle",
            "--matlab-batch-dir",
            str(matlab_batch_dir),
            "--oracle-root",
            str(oracle_root),
            "--dataset-file",
            str(dataset_file),
            "--oracle-id",
            "oracle-a",
        ]
    )

    def _fake_load_tiff(_path):
        return np.ones((2, 2, 2), dtype=np.uint16)

    class FirstProcessor:
        def process_image(self, image, parameters, *, run_dir=None, stop_after=None, **_kwargs):
            checkpoint_dir = Path(run_dir) / CHECKPOINTS_DIR
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            from joblib import dump

            dump(
                {
                    "energy_origin": "python_native_hessian",
                    "energy": np.zeros((2, 2, 2), dtype=np.float32),
                    "scale_indices": np.zeros((2, 2, 2), dtype=np.int16),
                    "lumen_radius_microns": np.array([1.0], dtype=np.float32),
                    "lumen_radius_pixels_axes": np.ones((1, 3), dtype=np.float32),
                },
                checkpoint_dir / "checkpoint_energy.pkl",
            )
            snapshot_payload = {
                "status": "completed",
                "target_stage": stop_after,
                "current_stage": stop_after,
                "provenance": {
                    "source": "pipeline",
                    "layout": "structured",
                    "stop_after": stop_after,
                    "image_shape": list(image.shape),
                },
                "input_fingerprint": dataset_hash,
            }
            (Path(run_dir) / METADATA_DIR).mkdir(parents=True, exist_ok=True)
            (Path(run_dir) / RUN_SNAPSHOT_PATH).write_text(
                json.dumps(snapshot_payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            return {"parameters": parameters}

    monkeypatch.setattr("source.io.tiff.load_tiff_volume", _fake_load_tiff)
    monkeypatch.setattr("source.core.pipeline.SLAVVProcessor", FirstProcessor)

    dest_run_root = experiment_root / "runs" / "seed-c"
    parity_experiment.main(
        [
            "init-exact-run",
            "--dataset-root",
            str(dataset_root),
            "--oracle-root",
            str(oracle_root),
            "--dest-run-root",
            str(dest_run_root),
            "--stop-after",
            "energy",
        ]
    )

    class SecondProcessor:
        def process_image(self, *_args, **_kwargs):
            raise AssertionError("completed seed should finalize without rerunning process_image")

    monkeypatch.setattr("source.core.pipeline.SLAVVProcessor", SecondProcessor)
    parity_experiment.main(
        [
            "init-exact-run",
            "--dataset-root",
            str(dataset_root),
            "--oracle-root",
            str(oracle_root),
            "--dest-run-root",
            str(dest_run_root),
            "--stop-after",
            "energy",
        ]
    )

    run_manifest = json.loads((dest_run_root / RUN_MANIFEST_PATH).read_text(encoding="utf-8"))
    assert run_manifest["kind"] == "parity_source_run"
    assert run_manifest["command"] == "init-exact-run"
    snapshot_payload = json.loads((dest_run_root / RUN_SNAPSHOT_PATH).read_text(encoding="utf-8"))
    assert snapshot_payload["provenance"]["input_file"] == str(
        dataset_root / "01_Input" / dataset_file.name
    )
    assert snapshot_payload["provenance"]["oracle_id"] == "oracle-a"


@pytest.mark.integration
def test_init_exact_run_rejects_existing_active_seed(tmp_path):
    experiment_root = _build_experiment_root(tmp_path)
    dataset_file = tmp_path / "input.tif"
    dataset_file.write_bytes(b"tiff")
    dataset_hash = parity_experiment._materialize_dataset_record(
        experiment_root,
        dataset_hash=None,
        dataset_file=dataset_file,
    )
    dataset_root = experiment_root / "datasets" / dataset_hash
    matlab_batch_dir = (
        tmp_path / "matlab-source" / "01_Input" / "matlab_results" / "batch_260421-151654"
    )
    matlab_batch_dir.parent.mkdir(parents=True, exist_ok=True)
    _materialize_exact_matlab_batch(tmp_path / "matlab-source")
    settings_dir = matlab_batch_dir / "settings"
    settings_dir.mkdir(parents=True, exist_ok=True)
    savemat(
        settings_dir / "energy_260421.mat",
        {
            "microns_per_voxel": np.array([0.5, 0.5, 1.0], dtype=np.float64),
            "radius_of_smallest_vessel_in_microns": 1.5,
            "radius_of_largest_vessel_in_microns": 40.0,
            "sample_index_of_refraction": 1.33,
            "numerical_aperture": 0.95,
            "excitation_wavelength_in_microns": 0.95,
            "scales_per_octave": 6,
            "max_voxels_per_node_energy": 1000000,
            "gaussian_to_ideal_ratio": 0.5,
            "spherical_to_annular_ratio": 0.5,
            "approximating_PSF": 1,
        },
    )
    savemat(
        settings_dir / "vertices_260421.mat",
        {
            "space_strel_apothem": 1,
            "energy_upper_bound": 0,
            "max_voxels_per_node": 6000,
            "length_dilation_ratio": 1,
        },
    )
    savemat(
        settings_dir / "edges_260421.mat",
        {
            "max_edge_length_per_origin_radius": 30,
            "space_strel_apothem_edges": 1,
            "number_of_edges_per_vertex": 4,
        },
    )
    savemat(settings_dir / "network_260421.mat", {"sigma_strand_smoothing": 1})
    oracle_root = experiment_root / "oracles" / "oracle-a"
    parity_experiment.main(
        [
            "promote-oracle",
            "--matlab-batch-dir",
            str(matlab_batch_dir),
            "--oracle-root",
            str(oracle_root),
            "--dataset-file",
            str(dataset_file),
            "--oracle-id",
            "oracle-a",
        ]
    )

    dest_run_root = experiment_root / "runs" / "seed-d"
    parity_experiment.ensure_dest_run_layout(dest_run_root)
    (dest_run_root / EXPERIMENT_PROVENANCE_PATH).write_text(
        json.dumps(
            {
                "bootstrap_kind": "init-exact-run",
                "dataset_hash": dataset_hash,
                "oracle_id": "oracle-a",
                "stop_after": "vertices",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (dest_run_root / RUN_SNAPSHOT_PATH).write_text(
        json.dumps({"status": "running"}, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="still active"):
        parity_experiment.main(
            [
                "init-exact-run",
                "--dataset-root",
                str(dataset_root),
                "--oracle-root",
                str(oracle_root),
                "--dest-run-root",
                str(dest_run_root),
            ]
        )
