"""Unit tests for workspace management utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from slavv_python.runtime.workspace import WorkspaceAuditor, find_experiment_root, find_repo_root

if TYPE_CHECKING:
    from pathlib import Path


def test_find_repo_root_discovery(tmp_path: Path):
    """Test that repo root is found by looking for pyproject.toml."""
    # Create a dummy repo structure
    repo = tmp_path / "my_repo"
    repo.mkdir()
    (repo / "pyproject.toml").touch()

    sub = repo / "a" / "b" / "c"
    sub.mkdir(parents=True)

    assert find_repo_root(sub) == repo


def test_find_repo_root_failure(tmp_path: Path):
    """Test that it raises RuntimeError if no pyproject.toml is found."""
    with patch("pathlib.Path.exists", return_value=False):
        with pytest.raises(RuntimeError, match="Could not find repository root"):
            find_repo_root(tmp_path)


def test_find_experiment_root(tmp_path: Path):
    """Test that experiment root (workspace) is correctly identified."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").touch()
    workspace = repo / "workspace"
    workspace.mkdir()

    assert find_experiment_root(repo) == workspace


def test_auditor_detects_violations(tmp_path: Path):
    """Test that WorkspaceAuditor identifies non-standard items."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").touch()
    (repo / "README.md").touch()

    # Create valid folder
    (repo / "source").mkdir()

    # Create violation folder
    (repo / "legacy_stuff").mkdir()

    # Create violation file
    (repo / "orphaned_file.txt").touch()

    auditor = WorkspaceAuditor(repo)
    results = auditor.audit_root()

    assert any("legacy_stuff" in v for v in results)
    assert any("orphaned_file.txt" in v for v in results)


def test_auditor_detects_misplaced_egg_info(tmp_path: Path):
    """Test that WorkspaceAuditor identifies egg-info in slavv_python/."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").touch()

    source = repo / "source"
    source.mkdir()

    bad_egg = source / "slavv_python.egg-info"
    bad_egg.mkdir()
    (bad_egg / "PKG-INFO").touch()

    auditor = WorkspaceAuditor(repo)
    results = auditor.audit_source()

    assert any("slavv_python.egg-info" in v for v in results)
