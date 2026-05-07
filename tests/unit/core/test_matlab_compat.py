from __future__ import annotations

import numpy as np

from slavv_python.core.matlab_compat import stages, vectorize_v200


def test_get_energy_v202_delegates_to_energy_facade(monkeypatch):
    calls: list[tuple[np.ndarray, dict[str, object], object]] = []

    def fake_calculate_energy_field(image, params, get_chunking_lattice_func=None):
        calls.append((image, params, get_chunking_lattice_func))
        return {"energy": True}

    monkeypatch.setattr(stages, "calculate_energy_field", fake_calculate_energy_field)

    image = np.zeros((2, 2, 2), dtype=np.float32)
    params = {"energy_method": "hessian"}
    result = stages.get_energy_v202(image, params)

    assert result == {"energy": True}
    assert calls == [(image, params, None)]


def test_choose_edges_v200_delegates_to_workflow_chooser(monkeypatch):
    calls: list[tuple[object, ...]] = []

    def fake_choose(*args):
        calls.append(args)
        return {"chosen": True}

    monkeypatch.setattr(stages, "choose_edges_for_workflow", fake_choose)

    candidates = {"connections": np.zeros((0, 2), dtype=np.int32)}
    vertex_positions = np.zeros((0, 3), dtype=np.float32)
    vertex_scales = np.zeros((0,), dtype=np.int16)
    lumen_radius_microns = np.array([1.0], dtype=np.float32)
    lumen_radius_pixels_axes = np.ones((1, 3), dtype=np.float32)
    params = {"comparison_exact_network": True}

    result = stages.choose_edges_v200(
        candidates,
        vertex_positions,
        vertex_scales,
        lumen_radius_microns,
        lumen_radius_pixels_axes,
        (4, 4, 4),
        params,
    )

    assert result == {"chosen": True}
    assert calls == [
        (
            candidates,
            vertex_positions,
            vertex_scales,
            lumen_radius_microns,
            lumen_radius_pixels_axes,
            (4, 4, 4),
            params,
        )
    ]


def test_vectorize_v200_delegates_to_slavv_processor():
    image = np.zeros((2, 2, 2), dtype=np.float32)
    params = {"comparison_exact_network": True}
    calls: list[tuple[object, ...]] = []

    class FakeProcessor:
        def __init__(self):
            self.energy_data = {"energy": True}
            self.vertices = {"vertices": True}
            self.edges = {"edges": True}
            self.network = {"network": True}

        def run(
            self,
            image_arg,
            params_arg,
            *,
            progress_callback=None,
            event_callback=None,
            run_dir=None,
            stop_after=None,
            force_rerun_from=None,
        ):
            calls.append(
                (
                    image_arg,
                    params_arg,
                    progress_callback,
                    event_callback,
                    run_dir,
                    stop_after,
                    force_rerun_from,
                )
            )
            return {
                "parameters": params_arg,
                "energy_data": self.energy_data,
                "vertices": self.vertices,
                "edges": self.edges,
                "network": self.network,
            }

    processor = FakeProcessor()
    result = vectorize_v200(
        image,
        params,
        processor=processor,
        run_dir="run-dir",
        stop_after="edges",
        force_rerun_from="vertices",
    )

    assert result["energy_data"] == {"energy": True}
    assert result["vertices"] == {"vertices": True}
    assert result["edges"] == {"edges": True}
    assert result["network"] == {"network": True}
    assert calls == [(image, params, None, None, "run-dir", "edges", "vertices")]
