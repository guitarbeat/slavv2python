"""Runtime-oriented helpers for MATLAB comparison execution."""

from pathlib import Path

from slavv.evaluation.comparison import discover_matlab_artifacts


def test_discover_matlab_artifacts_returns_empty_for_missing_output(tmp_path: Path):
    missing = tmp_path / "missing_output"

    assert discover_matlab_artifacts(missing) == {}


def test_discover_matlab_artifacts_prefers_latest_batch_and_network_file(tmp_path: Path):
    older = tmp_path / "batch_260323-180000"
    newer = tmp_path / "batch_260323-190000"
    (older / "vectors").mkdir(parents=True)
    newer_vectors = newer / "vectors"
    newer_vectors.mkdir(parents=True)
    (newer_vectors / "network_260323-190100_sample.mat").write_text("", encoding="utf-8")

    artifacts = discover_matlab_artifacts(tmp_path)

    assert artifacts["batch_folder"] == str(newer)
    assert artifacts["vectors_dir"] == str(newer_vectors)
    assert artifacts["network_mat"] == str(newer_vectors / "network_260323-190100_sample.mat")


def test_discover_matlab_artifacts_handles_partial_batch_without_network(tmp_path: Path):
    batch_folder = tmp_path / "batch_260323-200000"
    vectors_dir = batch_folder / "vectors"
    vectors_dir.mkdir(parents=True)

    artifacts = discover_matlab_artifacts(tmp_path)

    assert artifacts["batch_folder"] == str(batch_folder)
    assert artifacts["vectors_dir"] == str(vectors_dir)
    assert "network_mat" not in artifacts
