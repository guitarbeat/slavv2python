import numpy as np

from source.analysis import (
    evaluate_registration,
    get_edge_metric,
    get_edges_for_vertex,
    register_vector_sets,
    resample_vectors,
    smooth_edge_traces,
    transform_vector_set,
)


def test_resample_vectors_straight_line():
    trace = np.array([[0.0, 0.0], [10.0, 0.0]])
    resampled = resample_vectors(trace, step=1.0)

    assert len(resampled) == 11
    expected = np.column_stack((np.linspace(0, 10, 11), np.zeros(11)))
    np.testing.assert_allclose(resampled, expected, atol=1e-6)


def test_resample_vectors_curved():
    trace = np.array([[0.0, 0.0], [5.0, 0.0], [5.0, 5.0]])
    resampled = resample_vectors(trace, step=2.0)

    assert len(resampled) == 6
    np.testing.assert_allclose(resampled[0], [0, 0], atol=1e-6)
    np.testing.assert_allclose(resampled[-1], [5, 5], atol=1e-6)
    np.testing.assert_allclose(resampled[2], [4, 0], atol=1e-6)
    np.testing.assert_allclose(resampled[3], [5, 1], atol=1e-6)


def test_resample_vectors_zero_length_trace_returns_single_point():
    trace = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])

    resampled = resample_vectors(trace, step=1.0)

    np.testing.assert_allclose(resampled, np.array([[1.0, 1.0]]))


def test_smooth_edge_traces():
    trace = np.array([[0, 0], [1, 1], [2, 0], [3, 1], [4, 0]], dtype=float)
    smoothed = smooth_edge_traces([trace], sigma=1.0)

    assert len(smoothed) == 1
    smoothed_trace = smoothed[0]
    assert smoothed_trace[1, 1] < 1.0
    assert smoothed_trace[3, 1] < 1.0
    assert smoothed_trace[2, 1] > 0.0


def test_transform_vector_set_scale_translate():
    points = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    transformed = transform_vector_set(
        points,
        scale=[2.0, 2.0, 2.0],
        translate=[1.0, 1.0, 1.0],
    )

    expected = np.array([[3, 1, 1], [1, 3, 1], [1, 1, 3]], dtype=float)
    np.testing.assert_allclose(transformed, expected, atol=1e-6)


def test_register_vector_sets_rigid():
    source = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float)
    target = source + np.array([10, 0, 0])

    transform, error = register_vector_sets(source, target, method="rigid", return_error=True)

    source_homogeneous = np.c_[source, np.ones(4)]
    source_registered = (source_homogeneous @ transform.T)[:, :3]
    np.testing.assert_allclose(source_registered, target, atol=1e-6)
    assert error < 1e-6


def test_evaluate_registration_empty_sets_return_zero_score_and_empty_matches():
    score, before_to_after, after_to_before = evaluate_registration(
        np.empty((0, 3), dtype=float),
        np.empty((0, 3), dtype=float),
    )

    assert score == 0.0
    assert before_to_after.size == 0
    assert after_to_before.size == 0


def test_get_edge_metric_supports_length_and_energy_summary_methods():
    energy = np.arange(27, dtype=float).reshape(3, 3, 3)
    trace = np.array(
        [
            [-1.0, -1.0, -1.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [5.0, 5.0, 5.0],
        ],
        dtype=float,
    )
    expected_samples = np.array([0.0, 0.0, 13.0, 26.0], dtype=float)

    assert np.isclose(get_edge_metric(trace, method="length"), 6 * np.sqrt(3))
    assert np.isclose(get_edge_metric(trace, energy, method="mean_energy"), expected_samples.mean())
    assert np.isclose(get_edge_metric(trace, energy, method="min_energy"), expected_samples.min())
    assert np.isclose(get_edge_metric(trace, energy, method="max_energy"), expected_samples.max())
    assert np.isclose(
        get_edge_metric(trace, energy, method="median_energy"),
        np.median(expected_samples),
    )


def test_get_edges_for_vertex():
    connections = np.array([[0, 1], [1, 2], [0, 2]])

    assert sorted(get_edges_for_vertex(connections, 0)) == [0, 2]
    assert sorted(get_edges_for_vertex(connections, 1)) == [0, 1]
    assert len(get_edges_for_vertex(connections, 3)) == 0
