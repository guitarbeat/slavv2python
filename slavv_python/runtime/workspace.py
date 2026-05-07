"""Repository awareness and structural validation for the SLAVV workspace."""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import ClassVar

logger = logging.getLogger(__name__)


class FolderRole(Enum):
    """Canonical roles for repository directories."""

    PACKAGE_ROOT = "PACKAGE_ROOT"  # slavv_python/
    DEV_ROOT = "DEV_ROOT"  # workspace/
    DATASETS = "DATASETS"  # workspace/datasets/
    ORACLES = "ORACLES"  # workspace/oracles/
    RUNS = "RUNS"  # workspace/runs/
    REPORTS = "REPORTS"  # workspace/reports/
    DOCS = "DOCS"  # docs/


def find_repo_root(start_path: Path | str | None = None) -> Path:
    """Find the repository root by looking for pyproject.toml."""
    current = Path(start_path or Path.cwd()).resolve()
    for parent in [current, *list(current.parents)]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not find repository root (missing pyproject.toml)")


def find_experiment_root(repo_root: Path | None = None) -> Path:
    """Find the canonical experiment root (usually repo/workspace)."""
    root = repo_root or find_repo_root()
    dev_path = root / "workspace"
    if not dev_path.exists():
        raise RuntimeError(f"Experiment root 'workspace' not found in {root}")
    return dev_path


class WorkspaceAuditor:
    """Audits the repository for structural violations."""

    CANONICAL_ROOT_FOLDERS: ClassVar[set[str]] = {
        "slavv_python",
        "workspace",
        "docs",
        "external",
        ".github",
        ".git",
    }

    CANONICAL_ROOT_FILES: ClassVar[set[str]] = {
        "pyproject.toml",
        "README.md",
        "LICENSE",
        "CHANGELOG.md",
        "ANTIGRAVITY.md",
        ".gitignore",
    }

    def __init__(self, repo_root: Path | None = None):
        self.root = repo_root or find_repo_root()

    def audit_root(self) -> list[str]:
        """Check for non-standard folders and files in the repository root."""
        violations = []

        # Check for unexpected top-level items
        for item in self.root.iterdir():
            name = item.name
            if name.startswith((".", "__")) and name not in self.CANONICAL_ROOT_FOLDERS:
                continue

            if (
                item.is_dir()
                and name not in self.CANONICAL_ROOT_FOLDERS
                and name != "slavv_python.egg-info"
            ):
                violations.append(f"Non-standard root directory: {name}")
            elif (
                item.is_file()
                and name not in self.CANONICAL_ROOT_FILES
                and not name.endswith((".ini", ".yaml", ".yml", ".md"))
                and name not in {".gitmodules", ".sourcery.yaml"}
            ):
                violations.append(f"Non-standard root file: {name}")

        return violations

    def audit_source(self) -> list[str]:
        """Check for misplaced build artifacts in slavv_python/."""
        violations = []
        source_dir = self.root / "slavv_python"
        if not source_dir.exists():
            return ["Missing slavv_python directory"]

        for item in source_dir.rglob("*.egg-info"):
            violations.append(f"Misplaced build artifact in slavv_python: {item.relative_to(self.root)}")

        return violations

    def run_full_audit(self) -> dict[str, list[str]]:
        """Run all audit checks."""
        return {
            "root": self.audit_root(),
            "source": self.audit_source(),
        }
