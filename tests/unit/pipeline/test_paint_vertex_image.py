"""Unit tests for vertex-volume painting occupancy and strel caching."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from slavv_python.pipeline.vertices import painting
from slavv_python.pipeline.vertices.painting import paint_vertex_image

if TYPE_CHECKING:
    import pytest


def test_paint_vertex_image_cache_matches_per_vertex_baseline() -> None:
    """Reused radii must not change occupancy versus painting one vertex at a time."""
    image_shape = (28, 28, 16)
    positions = np.array(
        [
            [8.0, 8.0, 8.0],
            [20.0, 8.0, 8.0],
            [8.0, 20.0, 8.0],
        ],
        dtype=np.float64,
    )
    scales = np.zeros((3,), dtype=np.int16)
    radii = np.array(
        [
            [2.0, 2.0, 1.0],
            [2.0, 2.0, 1.0],
            [2.0, 2.0, 1.0],
        ],
        dtype=np.float64,
    )

    cached = paint_vertex_image(positions, scales, radii, image_shape)

    baseline = np.zeros(image_shape, dtype=np.uint16)
    for i in range(len(positions)):
        single = paint_vertex_image(
            positions[i : i + 1],
            scales[i : i + 1],
            radii[i : i + 1],
            image_shape,
        )
        occupied = single > 0
        baseline[occupied] = np.uint16(i + 1)

    assert cached.dtype == np.uint16
    assert np.array_equal(cached, baseline)
    assert int(np.count_nonzero(cached == 0)) < int(np.prod(image_shape))


def test_paint_vertex_image_distinct_radii_paint_distinct_bodies() -> None:
    """Different radius keys must still produce different occupancy footprints."""
    image_shape = (24, 24, 16)
    positions = np.array(
        [
            [8.0, 8.0, 8.0],
            [16.0, 16.0, 8.0],
        ],
        dtype=np.float64,
    )
    scales = np.zeros((2,), dtype=np.int16)
    radii = np.array(
        [
            [1.0, 1.0, 1.0],
            [3.0, 3.0, 1.5],
        ],
        dtype=np.float64,
    )

    painted = paint_vertex_image(positions, scales, radii, image_shape)
    small_count = int(np.count_nonzero(painted == 1))
    large_count = int(np.count_nonzero(painted == 2))
    assert small_count > 0
    assert large_count > small_count


def test_paint_vertex_image_calls_ellipsoid_once_per_unique_radius(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The strel cache must generate each unique ellipsoid once, not once per vertex."""
    calls: list[tuple[object, ...]] = []
    real_ellipsoid = painting.ellipsoid

    def _counting_ellipsoid(*args: object, **kwargs: object) -> np.ndarray:
        calls.append(args)
        return real_ellipsoid(*args, **kwargs)

    monkeypatch.setattr(painting, "ellipsoid", _counting_ellipsoid)

    image_shape = (24, 24, 16)
    positions = np.array(
        [
            [6.0, 6.0, 8.0],
            [18.0, 6.0, 8.0],
            [6.0, 18.0, 8.0],
            [18.0, 18.0, 8.0],
        ],
        dtype=np.float64,
    )
    scales = np.zeros((4,), dtype=np.int16)
    shared = np.array([1.5, 1.5, 1.0], dtype=np.float64)
    other = np.array([2.5, 2.5, 1.0], dtype=np.float64)
    radii = np.vstack([shared, shared, other, shared])

    painted = paint_vertex_image(positions, scales, radii, image_shape)

    assert len(calls) == 2
    assert int(np.count_nonzero(painted == 1)) > 0
    assert int(np.count_nonzero(painted == 4)) > 0
