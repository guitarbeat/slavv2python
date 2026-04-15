from __future__ import annotations

import joblib
import numpy as np
import pytest
from h5py import File
from scipy.io import savemat

from slavv.io.matlab_bridge import import_matlab_batch, load_matlab_batch_params


def test_import_matlab_batch_loads_hdf5_energy_and_pipeline_vertices(tmp_path):
    batch_dir = tmp_path / "batch_260326-120000"
    data_dir = batch_dir / "data"
    settings_dir = batch_dir / "settings"
    vectors_dir = batch_dir / "vectors"
    checkpoint_dir = tmp_path / "checkpoints"
    data_dir.mkdir(parents=True)
    settings_dir.mkdir()
    vectors_dir.mkdir()

    energy_h5_path = data_dir / "energy_260326-120000_sample"
    with File(energy_h5_path, "w") as handle:
        dataset = np.zeros((2, 2, 3, 4), dtype=np.float32)
        dataset[0] = np.array(
            [
                [[1, 1, 2, 2], [1, 2, 2, 3], [1, 1, 1, 1]],
                [[2, 2, 2, 2], [2, 3, 3, 3], [1, 1, 2, 2]],
            ],
            dtype=np.float32,
        )
        dataset[1] = np.linspace(-6.0, -1.0, num=24, dtype=np.float32).reshape(2, 3, 4)
        handle.create_dataset("d", data=dataset)

    savemat(
        settings_dir / "energy_260326-120000.mat",
        {
            "microns_per_voxel": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "pixels_per_sigma_PSF": np.array([0.5, 0.75, 1.0], dtype=np.float32),
            "lumen_radius_in_microns_range": np.array([1.5, 2.5, 4.0], dtype=np.float32),
            "lumen_radius_in_pixels_range": np.array(
                [[1.0, 1.0, 1.0], [2.0, 2.5, 3.0], [4.0, 4.5, 5.0]],
                dtype=np.float32,
            ),
        },
    )
    savemat(
        vectors_dir / "vertices_260326-120100.mat",
        {
            "vertex_space_subscripts": np.array([[2, 3, 4, 1], [3, 4, 5, 2]], dtype=np.uint16),
            "vertex_scale_subscripts": np.array([1, 2], dtype=np.uint16),
        },
    )

    written = import_matlab_batch(batch_dir, checkpoint_dir, stages=["energy", "vertices"])

    assert set(written) == {"energy", "vertices"}

    energy_checkpoint = joblib.load(checkpoint_dir / "checkpoint_energy.pkl")
    vertices_checkpoint = joblib.load(checkpoint_dir / "checkpoint_vertices.pkl")

    assert energy_checkpoint["energy_origin"] == "matlab_batch_hdf5"
    assert energy_checkpoint["energy_source"] == "matlab_batch_hdf5"
    assert energy_checkpoint["energy"].shape == (2, 3, 4)
    assert energy_checkpoint["scale_indices"][0, 0, 0] == 0
    assert energy_checkpoint["scale_indices"][0, 0, 2] == 1
    assert energy_checkpoint["scale_indices"][1, 1, 1] == 2
    assert np.allclose(energy_checkpoint["microns_per_sigma_PSF"], [0.5, 1.5, 3.0])
    assert energy_checkpoint["lumen_radius_pixels_axes"].shape == (3, 3)

    assert vertices_checkpoint["count"] == 2
    assert vertices_checkpoint["positions"].shape == (2, 3)
    assert vertices_checkpoint["scales"].tolist() == [0, 1]
    assert np.allclose(vertices_checkpoint["radii_microns"], [1.5, 2.5])
    assert np.allclose(
        vertices_checkpoint["radii_pixels"],
        [1.0, np.cbrt(2.0 * 2.5 * 3.0)],
    )
    assert np.allclose(vertices_checkpoint["energies"], [0.0, 0.0])


def test_import_matlab_batch_prefers_curated_vertices_when_present(tmp_path):
    batch_dir = tmp_path / "batch_260326-130000"
    data_dir = batch_dir / "data"
    settings_dir = batch_dir / "settings"
    vectors_dir = batch_dir / "vectors"
    checkpoint_dir = tmp_path / "checkpoints"
    data_dir.mkdir(parents=True)
    settings_dir.mkdir()
    vectors_dir.mkdir()

    with File(data_dir / "energy_260326-130000_sample", "w") as handle:
        handle.create_dataset("d", data=np.zeros((2, 2, 2, 2), dtype=np.float32))
    savemat(
        settings_dir / "energy_260326-130000.mat",
        {
            "microns_per_voxel": np.array([1.0, 1.0, 1.0], dtype=np.float32),
            "pixels_per_sigma_PSF": np.array([1.0, 1.0, 1.0], dtype=np.float32),
            "lumen_radius_in_microns_range": np.array([1.0, 2.0], dtype=np.float32),
            "lumen_radius_in_pixels_range": np.array(
                [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=np.float32
            ),
        },
    )
    savemat(
        vectors_dir / "vertices_260326-130100.mat",
        {
            "vertex_space_subscripts": np.array([[2, 2, 2, 1]], dtype=np.uint16),
            "vertex_scale_subscripts": np.array([1], dtype=np.uint16),
        },
    )
    savemat(
        vectors_dir / "curated_vertices_260326-130200.mat",
        {
            "vertex_space_subscripts": np.array([[3, 3, 3, 1], [4, 4, 4, 2]], dtype=np.uint16),
            "vertex_scale_subscripts": np.array([1, 2], dtype=np.uint16),
        },
    )

    import_matlab_batch(batch_dir, checkpoint_dir, stages=["energy", "vertices"])
    vertices_checkpoint = joblib.load(checkpoint_dir / "checkpoint_vertices.pkl")

    assert vertices_checkpoint["count"] == 2


def test_load_matlab_batch_params_maps_edge_influence_settings(tmp_path):
    batch_dir = tmp_path / "batch_260326-140000"
    settings_dir = batch_dir / "settings"
    settings_dir.mkdir(parents=True)

    savemat(
        settings_dir / "energy_260326-140000.mat",
        {
            "microns_per_voxel": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "radius_of_smallest_vessel_in_microns": np.array(1.5, dtype=np.float32),
            "radius_of_largest_vessel_in_microns": np.array(50.0, dtype=np.float32),
        },
    )
    savemat(
        settings_dir / "vertices_260326-140010.mat",
        {
            "space_strel_apothem": np.array(1, dtype=np.int16),
            "length_dilation_ratio": np.array(1.0, dtype=np.float32),
            "max_voxels_per_node": np.array(6000, dtype=np.int32),
        },
    )
    savemat(
        settings_dir / "edges_260326-140020.mat",
        {
            "number_of_edges_per_vertex": np.array(4, dtype=np.int16),
            "space_strel_apothem_edges": np.array(1, dtype=np.int16),
            "max_edge_length_per_origin_radius": np.array(60.0, dtype=np.float32),
            "length_dilation_ratio_vertices": np.array(2.0, dtype=np.float32),
            "length_dilation_ratio_edges": np.array(2.0 / 3.0, dtype=np.float32),
        },
    )

    params = load_matlab_batch_params(batch_dir)

    assert params["microns_per_voxel"] == [1.0, 2.0, 3.0]
    assert params["max_voxels_per_node"] == 6000
    assert params["number_of_edges_per_vertex"] == 4
    assert params["space_strel_apothem_edges"] == 1
    assert params["max_edge_length_per_origin_radius"] == 60.0
    assert params["sigma_per_influence_vertices"] == 2.0
    assert params["sigma_per_influence_edges"] == pytest.approx(2.0 / 3.0)
