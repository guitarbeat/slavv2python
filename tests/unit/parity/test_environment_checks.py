from __future__ import annotations

from typing import TYPE_CHECKING

from slavv.parity.environment_checks import Validator

if TYPE_CHECKING:
    from pathlib import Path


def test_check_vectorization_public_uses_source_subdirectory(tmp_path: Path):
    repo_path = tmp_path / "external" / "Vectorization-Public" / "source"
    repo_path.mkdir(parents=True)
    (repo_path / "vectorize_V200.m").write_text("% test", encoding="utf-8")

    validator = Validator(project_root=tmp_path)

    assert validator.check_vectorization_public() is True
    assert validator.errors == []
