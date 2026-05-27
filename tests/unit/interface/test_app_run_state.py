from __future__ import annotations

import numpy as np

from slavv_python.schema.app_run import AppRunState, rebuild_network_for_curation
from slavv_python.schema.results import EdgeSet, EnergyResult, PipelineResult, VertexSet


def test_app_run_state_mapping_compatibility():
    pipeline = PipelineResult(
        parameters={"microns_per_voxel": [1.0, 1.0, 1.0]},
        vertices=VertexSet.from_dict(
            {
                "positions": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
                "scales": np.array([0], dtype=np.int16),
            }
        ),
    )
    app_run = AppRunState(pipeline=pipeline, image_shape=(4, 4, 4))
    assert "vertices" in app_run
    assert len(app_run["vertices"]["positions"]) == 1


def test_rebuild_network_for_curation_returns_typed_pipeline():
    vertices = VertexSet.from_dict(
        {
            "positions": np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32),
            "scales": np.array([0, 0], dtype=np.int16),
        }
    )
    edges = EdgeSet.from_dict(
        {
            "traces": [np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)],
            "connections": np.array([[0, 1]], dtype=np.int32),
            "energies": np.array([-1.0], dtype=np.float32),
        }
    )
    app_run = AppRunState(
        pipeline=PipelineResult(
            parameters={"microns_per_voxel": [1.0, 1.0, 1.0]},
            energy_data=EnergyResult.from_dict(
                {
                    "energy": np.zeros((3, 3, 3), dtype=np.float32),
                    "scale_indices": np.zeros((3, 3, 3), dtype=np.int16),
                }
            ),
            vertices=vertices,
            edges=edges,
        )
    )
    updated = rebuild_network_for_curation(
        app_run,
        vertices.to_dict(),
        edges.to_dict(),
    )
    assert updated.pipeline.network is not None
    assert updated.pipeline.vertices is not None
    assert updated.pipeline.edges is not None
