"""Shared MATLAB-engine fixtures for stretch Energy isolation tests."""

from __future__ import annotations

from collections.abc import Iterator  # noqa: TC003

import pytest

from slavv_python.pipeline.energy.matlab_engine_backend import (
    MatlabEngineInfraError,
    default_vectorization_root,
    resolve_matlab_root,
)
from slavv_python.pipeline.energy.matlab_engine_host import (
    MatlabEnginePy37Worker,
    resolve_python37_executable,
)


@pytest.fixture(scope="session")
def stretch_py37_worker() -> Iterator[MatlabEnginePy37Worker]:
    """Long-lived py37 MATLAB worker, or skip as incomplete_infra."""
    python37 = resolve_python37_executable()
    if python37 is None:
        pytest.skip("incomplete_infra: isolated Python 3.7 stretch env missing")
    try:
        resolve_matlab_root()
    except MatlabEngineInfraError as exc:
        pytest.skip(f"incomplete_infra: {exc}")
    vectorization_root = default_vectorization_root()
    if not (vectorization_root / "source").is_dir():
        pytest.skip("incomplete_infra: Vectorization-Public source missing")
    worker = MatlabEnginePy37Worker(python37, vectorization_root=vectorization_root)
    try:
        worker.start()
    except MatlabEngineInfraError as exc:
        pytest.skip(f"incomplete_infra: {exc}")
    try:
        yield worker
    finally:
        worker.quit()
