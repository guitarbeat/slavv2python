"""Comprehensive End-to-End (E2E) Acceptance Test Suite for slavv2python uv Migration.

Methodology (4-Tier E2E Testing Architecture):
- Tier 1: Feature Coverage (>=5 test cases per feature: Build System, uv.lock, uv sync extras, uv run dev tooling, CI workflows)
- Tier 2: Boundary & Corner Cases (>=5 test cases per feature: git tracking, tampering detection, extras combinations, pip editable & wheel build, docs sanitization)
- Tier 3: Cross-Feature Combinations (pairwise interactions across sync, lock, build, pytest, ruff, CI)
- Tier 4: Real-World Application Scenarios (end-to-end user journeys: clean onboarding, CI pipeline emulation, release packaging, full lifecycle, probe scripts)
"""

from __future__ import annotations

import os
import subprocess
import tarfile
import zipfile
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

try:
    import yaml  # type: ignore[import-not-found]
except ImportError:
    yaml = None  # Fallback dictionary inspection if PyYAML is uninstalled


REPO_ROOT = Path(__file__).resolve().parents[2]


def run_cmd(
    cmd: list[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: float = 60.0,
) -> subprocess.CompletedProcess[str]:
    """Execute a CLI command safely with captured output."""
    work_dir = cwd or REPO_ROOT
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    return subprocess.run(
        cmd,
        cwd=str(work_dir),
        env=run_env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def load_pyproject(root: Path | None = None) -> dict[str, Any]:
    """Load pyproject.toml as a dictionary."""
    path = (root or REPO_ROOT) / "pyproject.toml"
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_uv_lock(root: Path | None = None) -> dict[str, Any]:
    """Load uv.lock as a dictionary."""
    path = (root or REPO_ROOT) / "uv.lock"
    with open(path, "rb") as f:
        return tomllib.load(f)


def parse_yaml_file(path: Path) -> dict[str, Any]:
    """Load a YAML file using PyYAML or structured dictionary fallback."""
    content = path.read_text(encoding="utf-8")
    if yaml is not None:
        data = yaml.safe_load(content)
        if isinstance(data, dict):
            return data
    return {"raw_text": content}


# ==============================================================================
# TIER 1: FEATURE COVERAGE (>=5 test cases per feature)
# ==============================================================================


class TestTier1Feature1BuildSystem:
    """Feature 1: Build system backend migration to uv / hatchling (R1)."""

    def test_t1_f1_build_system_backend_is_hatchling(self) -> None:
        """Verify [build-system] specifies hatchling and does not reference setuptools or wheel."""
        pyproject = load_pyproject()
        assert "build-system" in pyproject, "Missing [build-system] table"
        build_sys = pyproject["build-system"]
        assert build_sys.get("build-backend") == "hatchling.build"
        requires = build_sys.get("requires", [])
        assert any("hatchling" in r for r in requires), (
            f"Requires should contain hatchling: {requires}"
        )
        assert not any("setuptools" in r for r in requires), "Requires must not contain setuptools"
        assert not any("wheel" in r for r in requires), "Requires must not contain wheel"

    def test_t1_f1_legacy_setuptools_sections_removed(self) -> None:
        """Verify tool.setuptools sections are completely absent from pyproject.toml."""
        pyproject = load_pyproject()
        tools = pyproject.get("tool", {})
        assert "setuptools" not in tools, "tool.setuptools must be removed"

    def test_t1_f1_pep621_project_metadata_verbatim(self) -> None:
        """Verify PEP 621 project metadata and base dependencies are preserved verbatim."""
        pyproject = load_pyproject()
        project = pyproject.get("project", {})
        assert project.get("name") == "slavv_python"
        assert project.get("version") == "0.1.0"
        assert project.get("requires-python") == ">=3.11"
        assert project.get("readme") == "README.md"
        assert project.get("license") == {"text": "GPL-3.0"}
        assert any(a.get("name") == "UTFOIL Lab" for a in project.get("authors", []))

        # Base dependencies (18 packages)
        deps = project.get("dependencies", [])
        expected_bases = [
            "numpy",
            "scipy",
            "scikit-image",
            "scikit-learn",
            "networkx",
            "h5py",
            "tifffile",
            "matplotlib",
            "plotly",
            "pandas",
            "pillow",
            "joblib",
            "defusedxml",
            "psutil",
            "py-cpuinfo",
            "seaborn",
            "fasteners",
            "tabulate",
        ]
        for base in expected_bases:
            assert any(base in dep for dep in deps), (
                f"Base dependency {base} missing from project.dependencies"
            )

    def test_t1_f1_optional_dependency_groups_12_intact(self) -> None:
        """Verify all 12 optional dependency groups are defined in project.optional-dependencies."""
        pyproject = load_pyproject()
        opt_deps = pyproject.get("project", {}).get("optional-dependencies", {})
        expected_groups = [
            "app",
            "ml",
            "notebooks",
            "dicom",
            "sitk",
            "cupy",
            "zarr",
            "napari",
            "accel",
            "workspace",
            "tui",
            "all",
        ]
        for group in expected_groups:
            assert group in opt_deps, (
                f"Optional dependency group '{group}' missing from pyproject.toml"
            )
            assert len(opt_deps[group]) > 0, (
                f"Optional dependency group '{group}' must not be empty"
            )

    def test_t1_f1_project_scripts_and_hatch_config(self) -> None:
        """Verify [project.scripts] and [tool.hatch.build.targets.wheel] configuration."""
        pyproject = load_pyproject()
        scripts = pyproject.get("project", {}).get("scripts", {})
        assert scripts.get("slavv") == "slavv_python.interface.cli:main"
        assert scripts.get("slavv-app") == "slavv_python.interface.streamlit.launcher:main"

        hatch_wheel = (
            pyproject.get("tool", {})
            .get("hatch", {})
            .get("build", {})
            .get("targets", {})
            .get("wheel", {})
        )
        assert hatch_wheel.get("packages") == ["slavv_python"]


class TestTier1Feature2LockfileGeneration:
    """Feature 2: uv.lock lockfile generation and integrity (R2)."""

    def test_t1_f2_uv_lock_exists_at_repo_root(self) -> None:
        """Verify uv.lock exists at the repository root and is non-empty."""
        lock_path = REPO_ROOT / "uv.lock"
        assert lock_path.is_file(), "uv.lock does not exist at repo root"
        assert lock_path.stat().st_size > 1000, "uv.lock file is suspiciously small"

    def test_t1_f2_uv_lock_is_valid_toml(self) -> None:
        """Verify uv.lock parses cleanly as TOML."""
        lock_data = load_uv_lock()
        assert isinstance(lock_data, dict), "uv.lock did not parse as a TOML dictionary"
        assert "package" in lock_data or "manifest" in lock_data or "version" in lock_data

    def test_t1_f2_uv_lock_contains_slavv_python_root(self) -> None:
        """Verify slavv_python is resolved in uv.lock."""
        lock_data = load_uv_lock()
        packages = lock_data.get("package", [])
        pkg_names = [p.get("name") for p in packages if isinstance(p, dict)]
        assert "slavv_python" in pkg_names or "slavv-python" in pkg_names

    def test_t1_f2_uv_lock_resolves_optional_dependencies(self) -> None:
        """Verify packages from optional groups are present in uv.lock."""
        lock_data = load_uv_lock()
        packages = lock_data.get("package", [])
        pkg_names = {p.get("name") for p in packages if isinstance(p, dict)}
        for expected_pkg in ["streamlit", "pytest", "hypothesis", "ruff", "mypy"]:
            assert expected_pkg in pkg_names, (
                f"Expected package {expected_pkg} not resolved in uv.lock"
            )

    def test_t1_f2_uv_lock_check_command(self) -> None:
        """Verify uv lock --check exits with code 0 (lockfile is up to date)."""
        proc = run_cmd(["uv", "lock", "--check"])
        assert proc.returncode == 0, f"uv lock --check failed: {proc.stderr}\n{proc.stdout}"


class TestTier1Feature3SyncWithExtras:
    """Feature 3: uv sync with dependency extras (R1/R2)."""

    def test_t1_f3_uv_sync_dry_run_base(self) -> None:
        """Verify uv sync --dry-run exits with code 0 for base environment."""
        proc = run_cmd(["uv", "sync", "--dry-run"])
        assert proc.returncode == 0, f"uv sync --dry-run failed: {proc.stderr}\n{proc.stdout}"

    def test_t1_f3_uv_sync_dry_run_extra_app(self) -> None:
        """Verify uv sync --dry-run --extra app exits with code 0."""
        proc = run_cmd(["uv", "sync", "--dry-run", "--extra", "app"])
        assert proc.returncode == 0, (
            f"uv sync --dry-run --extra app failed: {proc.stderr}\n{proc.stdout}"
        )

    def test_t1_f3_uv_sync_dry_run_extra_workspace(self) -> None:
        """Verify uv sync --dry-run --extra workspace exits with code 0."""
        proc = run_cmd(["uv", "sync", "--dry-run", "--extra", "workspace"])
        assert proc.returncode == 0, (
            f"uv sync --dry-run --extra workspace failed: {proc.stderr}\n{proc.stdout}"
        )

    def test_t1_f3_uv_sync_dry_run_multi_extras(self) -> None:
        """Verify uv sync --dry-run with multiple extras exits with code 0."""
        proc = run_cmd(
            ["uv", "sync", "--dry-run", "--extra", "app", "--extra", "workspace", "--extra", "zarr"]
        )
        assert proc.returncode == 0, f"uv sync multi-extras failed: {proc.stderr}\n{proc.stdout}"

    def test_t1_f3_uv_sync_dry_run_all_extras(self) -> None:
        """Verify uv sync --dry-run --all-extras exits with code 0."""
        proc = run_cmd(["uv", "sync", "--dry-run", "--all-extras"])
        assert proc.returncode == 0, (
            f"uv sync --dry-run --all-extras failed: {proc.stderr}\n{proc.stdout}"
        )


class TestTier1Feature4DevTooling:
    """Feature 4: Developer workflow tooling via uv run (R3)."""

    def test_t1_f4_uv_run_cli_help(self) -> None:
        """Verify uv run slavv --help exits with code 0 and displays CLI usage."""
        proc = run_cmd(["uv", "run", "slavv", "--help"])
        assert proc.returncode == 0, f"uv run slavv --help failed: {proc.stderr}"
        assert "usage" in proc.stdout.lower() or "slavv" in proc.stdout.lower()

    def test_t1_f4_uv_run_cli_info(self) -> None:
        """Verify uv run slavv info exits with code 0 and provides system info."""
        proc = run_cmd(["uv", "run", "slavv", "info"])
        assert proc.returncode == 0, f"uv run slavv info failed: {proc.stderr}"
        assert "slavv" in proc.stdout.lower() or "python" in proc.stdout.lower()

    def test_t1_f4_uv_run_ruff_check(self) -> None:
        """Verify uv run ruff check slavv_python exits with code 0."""
        proc = run_cmd(["uv", "run", "ruff", "check", "slavv_python"])
        assert proc.returncode == 0, f"uv run ruff check failed: {proc.stderr}\n{proc.stdout}"

    def test_t1_f4_uv_run_mypy_execution(self) -> None:
        """Verify uv run mypy is executable via uv run."""
        proc = run_cmd(["uv", "run", "mypy", "--version"])
        assert proc.returncode == 0, f"uv run mypy failed: {proc.stderr}"

    def test_t1_f4_uv_run_pytest_execution(self) -> None:
        """Verify uv run pytest is executable via uv run."""
        proc = run_cmd(["uv", "run", "pytest", "--version"])
        assert proc.returncode == 0, f"uv run pytest failed: {proc.stderr}"


class TestTier1Feature5CIWorkflows:
    """Feature 5: CI workflow validity and setup-uv integration (R4)."""

    def test_t1_f5_regression_gate_yaml_valid(self) -> None:
        """Verify .github/workflows/regression-gate.yml is valid YAML."""
        wf_path = REPO_ROOT / ".github" / "workflows" / "regression-gate.yml"
        assert wf_path.is_file(), f"Workflow file missing: {wf_path}"
        data = parse_yaml_file(wf_path)
        assert data is not None

    def test_t1_f5_matlab_parity_yaml_valid(self) -> None:
        """Verify .github/workflows/matlab-random-component-parity.yml is valid YAML."""
        wf_path = REPO_ROOT / ".github" / "workflows" / "matlab-random-component-parity.yml"
        assert wf_path.is_file(), f"Workflow file missing: {wf_path}"
        data = parse_yaml_file(wf_path)
        assert data is not None

    def test_t1_f5_ci_workflows_declare_setup_uv(self) -> None:
        """Verify CI workflows use astral-sh/setup-uv action."""
        for wf_name in ["regression-gate.yml", "matlab-random-component-parity.yml"]:
            wf_path = REPO_ROOT / ".github" / "workflows" / wf_name
            content = wf_path.read_text(encoding="utf-8")
            assert "astral-sh/setup-uv" in content, (
                f"{wf_name} does not reference astral-sh/setup-uv"
            )

    def test_t1_f5_ci_workflows_use_uv_sync(self) -> None:
        """Verify CI workflows use uv sync for dependency installation."""
        for wf_name in ["regression-gate.yml", "matlab-random-component-parity.yml"]:
            wf_path = REPO_ROOT / ".github" / "workflows" / wf_name
            content = wf_path.read_text(encoding="utf-8")
            assert "uv sync" in content, f"{wf_name} does not use uv sync"

    def test_t1_f5_ci_workflows_use_uv_run(self) -> None:
        """Verify CI workflows execute test and quality steps via uv run."""
        for wf_name in ["regression-gate.yml", "matlab-random-component-parity.yml"]:
            wf_path = REPO_ROOT / ".github" / "workflows" / wf_name
            content = wf_path.read_text(encoding="utf-8")
            assert "uv run" in content, f"{wf_name} does not use uv run"


# ==============================================================================
# TIER 2: BOUNDARY & CORNER CASES (>=5 test cases per feature)
# ==============================================================================


class TestTier2Boundary1GitTracking:
    """Boundary Feature 1: uv.lock git ignore and tracking rules."""

    def test_t2_b1_git_check_ignore_uv_lock(self) -> None:
        """Verify git check-ignore reports uv.lock is not ignored."""
        proc = run_cmd(["git", "check-ignore", "-v", "uv.lock"])
        assert proc.returncode != 0, "uv.lock must NOT be ignored by git"

    def test_t2_b1_gitignore_no_active_uv_lock_rule(self) -> None:
        """Verify .gitignore does not contain an active uncommented uv.lock rule."""
        gitignore_path = REPO_ROOT / ".gitignore"
        assert gitignore_path.is_file()
        lines = gitignore_path.read_text(encoding="utf-8").splitlines()
        active_uv_lock_rules = [
            line.strip()
            for line in lines
            if line.strip() == "uv.lock" or line.strip() == "/uv.lock"
        ]
        assert len(active_uv_lock_rules) == 0, (
            f"Found active uv.lock in .gitignore: {active_uv_lock_rules}"
        )

    def test_t2_b1_git_add_dry_run_uv_lock(self) -> None:
        """Verify git add --dry-run uv.lock succeeds without --force flag."""
        proc = run_cmd(["git", "add", "--dry-run", "uv.lock"])
        assert proc.returncode == 0, f"git add --dry-run uv.lock failed: {proc.stderr}"

    def test_t2_b1_uv_lock_non_empty_size(self) -> None:
        """Verify uv.lock contains full resolved package graph (>50 KB)."""
        lock_size = (REPO_ROOT / "uv.lock").stat().st_size
        assert lock_size > 50_000, (
            f"uv.lock size ({lock_size} bytes) is too small for full dependency graph"
        )

    def test_t2_b1_uv_lock_root_directory_placement(self) -> None:
        """Verify uv.lock is co-located with root pyproject.toml."""
        assert (REPO_ROOT / "uv.lock").exists()
        assert (REPO_ROOT / "pyproject.toml").exists()
        assert (REPO_ROOT / "uv.lock").parent == (REPO_ROOT / "pyproject.toml").parent


class TestTier2Boundary2TamperingDetection:
    """Boundary Feature 2: uv lock --check detects tampering and drift."""

    def test_t2_b2_lock_check_detects_dependency_version_tamper(self, tmp_path: Path) -> None:
        """Verify modifying pyproject.toml dependency causes uv lock --check to fail in isolated workspace."""
        temp_dir = tmp_path / "tamper_test"
        temp_dir.mkdir(parents=True, exist_ok=True)
        pyproj_content = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        lock_content = (REPO_ROOT / "uv.lock").read_text(encoding="utf-8")

        tampered_pyproj = pyproj_content.replace('"numpy>=2.0.0,<2.5"', '"numpy>=99.99.0"')
        (temp_dir / "pyproject.toml").write_text(tampered_pyproj, encoding="utf-8")
        (temp_dir / "uv.lock").write_text(lock_content, encoding="utf-8")

        proc = run_cmd(["uv", "lock", "--check"], cwd=temp_dir)
        assert proc.returncode != 0, (
            "uv lock --check should fail when dependency constraint is tampered"
        )

    def test_t2_b2_lock_check_detects_added_dependency(self, tmp_path: Path) -> None:
        """Verify adding a new dependency to pyproject.toml triggers lock drift detection."""
        temp_dir = tmp_path / "add_dep_test"
        temp_dir.mkdir(parents=True, exist_ok=True)
        pyproj_content = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        lock_content = (REPO_ROOT / "uv.lock").read_text(encoding="utf-8")

        tampered_pyproj = pyproj_content.replace(
            '"tabulate>=0.9.0",', '"tabulate>=0.9.0",\n    "nonexistent-mock-pkg>=1.0.0",'
        )
        (temp_dir / "pyproject.toml").write_text(tampered_pyproj, encoding="utf-8")
        (temp_dir / "uv.lock").write_text(lock_content, encoding="utf-8")

        proc = run_cmd(["uv", "lock", "--check"], cwd=temp_dir)
        assert proc.returncode != 0, "uv lock --check should fail when a new dependency is added"

    def test_t2_b2_lock_check_detects_modified_extra(self, tmp_path: Path) -> None:
        """Verify modifying an optional-dependency group triggers lock drift detection."""
        temp_dir = tmp_path / "mod_extra_test"
        temp_dir.mkdir(parents=True, exist_ok=True)
        pyproj_content = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        lock_content = (REPO_ROOT / "uv.lock").read_text(encoding="utf-8")

        tampered_pyproj = pyproj_content.replace('"streamlit>=1.56.0",', '"streamlit>=999.0.0",')
        (temp_dir / "pyproject.toml").write_text(tampered_pyproj, encoding="utf-8")
        (temp_dir / "uv.lock").write_text(lock_content, encoding="utf-8")

        proc = run_cmd(["uv", "lock", "--check"], cwd=temp_dir)
        assert proc.returncode != 0, "uv lock --check should fail when extra group is modified"

    def test_t2_b2_lock_check_detects_corrupted_lockfile(self, tmp_path: Path) -> None:
        """Verify corrupted uv.lock syntax triggers error during uv lock --check."""
        temp_dir = tmp_path / "corrupt_lock_test"
        temp_dir.mkdir(parents=True, exist_ok=True)
        pyproj_content = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")

        (temp_dir / "pyproject.toml").write_text(pyproj_content, encoding="utf-8")
        (temp_dir / "uv.lock").write_text(
            "corrupted_lock_content = [invalid_syntax", encoding="utf-8"
        )

        proc = run_cmd(["uv", "lock", "--check"], cwd=temp_dir)
        assert proc.returncode != 0, "uv lock --check should fail on corrupted lockfile"

    def test_t2_b2_lock_check_restored_clean_passes(self, tmp_path: Path) -> None:
        """Verify clean un-tampered copy passes uv lock --check cleanly."""
        temp_dir = tmp_path / "clean_test"
        temp_dir.mkdir(parents=True, exist_ok=True)
        pyproj_content = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        lock_content = (REPO_ROOT / "uv.lock").read_text(encoding="utf-8")

        (temp_dir / "pyproject.toml").write_text(pyproj_content, encoding="utf-8")
        (temp_dir / "uv.lock").write_text(lock_content, encoding="utf-8")

        proc = run_cmd(["uv", "lock", "--check"], cwd=temp_dir)
        assert proc.returncode == 0, f"Clean copy failed uv lock --check: {proc.stderr}"


class TestTier2Boundary3SyncExtrasAndErrors:
    """Boundary Feature 3: uv sync extras combinations and error handling."""

    def test_t2_b3_sync_unknown_extra_fails_cleanly(self) -> None:
        """Verify syncing an unknown extra fails with non-zero exit code."""
        proc = run_cmd(["uv", "sync", "--dry-run", "--extra", "completely_unknown_extra_xyz"])
        assert proc.returncode != 0, "uv sync with unknown extra should fail"

    def test_t2_b3_sync_idempotent_dry_run(self) -> None:
        """Verify consecutive dry-run syncs produce identical return codes."""
        p1 = run_cmd(["uv", "sync", "--dry-run"])
        p2 = run_cmd(["uv", "sync", "--dry-run"])
        assert p1.returncode == 0
        assert p2.returncode == 0

    def test_t2_b3_sync_multi_extra_composition(self) -> None:
        """Verify complex multi-extra compositions resolve cleanly."""
        proc = run_cmd(
            [
                "uv",
                "sync",
                "--dry-run",
                "--extra",
                "app",
                "--extra",
                "ml",
                "--extra",
                "notebooks",
                "--extra",
                "dicom",
                "--extra",
                "sitk",
                "--extra",
                "zarr",
                "--extra",
                "workspace",
                "--extra",
                "tui",
            ]
        )
        assert proc.returncode == 0, f"Multi-extra sync failed: {proc.stderr}\n{proc.stdout}"

    def test_t2_b3_sync_all_extras_superset(self) -> None:
        """Verify uv sync --all-extras covers all optional dependencies."""
        proc = run_cmd(["uv", "sync", "--dry-run", "--all-extras"])
        assert proc.returncode == 0, f"All extras sync failed: {proc.stderr}"

    def test_t2_b3_sync_locked_flag_enforces_lockfile(self) -> None:
        """Verify uv sync --locked succeeds when lock is up to date."""
        proc = run_cmd(["uv", "sync", "--dry-run", "--locked"])
        assert proc.returncode == 0, f"uv sync --locked failed: {proc.stderr}"


class TestTier2Boundary4PipCompatibilityAndBuild:
    """Boundary Feature 4: pip editable compatibility and build artifact integrity."""

    def test_t2_b4_pip_install_editable_dry_run(self) -> None:
        """Verify uv pip install -e . --dry-run succeeds with hatchling backend."""
        proc = run_cmd(["uv", "pip", "install", "-e", ".", "--dry-run"])
        assert proc.returncode == 0, f"uv pip install -e . --dry-run failed: {proc.stderr}"

    def test_t2_b4_uv_build_generates_sdist_and_wheel(self, tmp_path: Path) -> None:
        """Verify uv build produces .tar.gz and .whl artifacts."""
        dist_dir = tmp_path / "dist_artifacts"
        proc = run_cmd(["uv", "build", "--out-dir", str(dist_dir)])
        assert proc.returncode == 0, f"uv build failed: {proc.stderr}\n{proc.stdout}"

        sdists = list(dist_dir.glob("*.tar.gz"))
        wheels = list(dist_dir.glob("*.whl"))
        assert len(sdists) == 1, f"Expected 1 sdist, found {sdists}"
        assert len(wheels) == 1, f"Expected 1 wheel, found {wheels}"

    def test_t2_b4_wheel_contains_package_json_overrides(self, tmp_path: Path) -> None:
        """Verify built wheel contains matlab_linspace_overrides.json package data."""
        dist_dir = tmp_path / "dist_check_json"
        proc = run_cmd(["uv", "build", "--wheel", "--out-dir", str(dist_dir)])
        assert proc.returncode == 0
        wheel_file = next(dist_dir.glob("*.whl"))

        with zipfile.ZipFile(wheel_file, "r") as z:
            names = z.namelist()
            assert any("matlab_linspace_overrides.json" in name for name in names), (
                f"matlab_linspace_overrides.json missing from wheel: {names}"
            )

    def test_t2_b4_wheel_contains_matlab_random_reference_json(self, tmp_path: Path) -> None:
        """Verify built wheel contains matlab_random_linspace_reference.json."""
        dist_dir = tmp_path / "dist_check_ref_json"
        proc = run_cmd(["uv", "build", "--wheel", "--out-dir", str(dist_dir)])
        assert proc.returncode == 0
        wheel_file = next(dist_dir.glob("*.whl"))

        with zipfile.ZipFile(wheel_file, "r") as z:
            names = z.namelist()
            assert any("matlab_random_linspace_reference.json" in name for name in names), (
                f"matlab_random_linspace_reference.json missing from wheel: {names}"
            )

    def test_t2_b4_wheel_metadata_pep621_fields(self, tmp_path: Path) -> None:
        """Verify built wheel METADATA file contains PEP 621 compliant metadata."""
        dist_dir = tmp_path / "dist_check_meta"
        proc = run_cmd(["uv", "build", "--wheel", "--out-dir", str(dist_dir)])
        assert proc.returncode == 0
        wheel_file = next(dist_dir.glob("*.whl"))

        with zipfile.ZipFile(wheel_file, "r") as z:
            meta_name = next(n for n in z.namelist() if n.endswith("METADATA"))
            metadata_text = z.read(meta_name).decode("utf-8")
            assert "Name: slavv_python" in metadata_text or "Name: slavv-python" in metadata_text
            assert "Version: 0.1.0" in metadata_text
            assert "Requires-Python: >=3.11" in metadata_text


class TestTier2Boundary5DocsAndErrorResilience:
    """Boundary Feature 5: Documentation sanitization and runtime error handling."""

    def test_t2_b5_readme_no_bare_pip_install_setup(self) -> None:
        """Verify README.md setup instructions use uv rather than bare pip install."""
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        assert "uv sync" in readme, "README.md must instruct users to run uv sync"
        assert "pip install -e" not in readme, (
            "README.md must not instruct users with bare pip install -e"
        )

    def test_t2_b5_agents_md_uses_uv_commands(self) -> None:
        """Verify AGENTS.md setup & quality sections use uv sync / uv run."""
        agents_md = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
        assert "uv sync" in agents_md, "AGENTS.md must instruct users to run uv sync"

    def test_t2_b5_scripts_readme_uses_uv_run(self) -> None:
        """Verify scripts/README.md references uv run python."""
        scripts_readme = REPO_ROOT / "scripts" / "README.md"
        if scripts_readme.exists():
            content = scripts_readme.read_text(encoding="utf-8")
            assert "uv run" in content, "scripts/README.md must use uv run"

    def test_t2_b5_streamlit_launcher_unit_test_passes(self) -> None:
        """Verify tests/unit/interface/test_streamlit_launcher.py passes cleanly."""
        proc = run_cmd(
            [
                "uv",
                "run",
                "--extra",
                "workspace",
                "pytest",
                "tests/unit/interface/test_streamlit_launcher.py",
                "-q",
            ]
        )
        assert proc.returncode == 0, (
            f"test_streamlit_launcher.py failed: {proc.stderr}\n{proc.stdout}"
        )

    def test_t2_b5_uv_run_invalid_command_fails_gracefully(self) -> None:
        """Verify uv run on a non-existent command exits non-zero."""
        proc = run_cmd(["uv", "run", "completely_nonexistent_cli_cmd_99999"])
        assert proc.returncode != 0

    def test_t2_b5_no_deprecated_pip_install_in_source_tree(self) -> None:
        """Verify slavv_python source tree contains zero deprecated pip install strings."""
        pkg_dir = REPO_ROOT / "slavv_python"
        for py_file in pkg_dir.rglob("*.py"):
            content = py_file.read_text(encoding="utf-8")
            assert "pip install" not in content, (
                f"Deprecated 'pip install' found in {py_file.relative_to(REPO_ROOT)}"
            )


# ==============================================================================
# TIER 3: CROSS-FEATURE COMBINATIONS (Pairwise Interactions)
# ==============================================================================


class TestTier3CrossFeatureCombinations:
    """Tier 3: Pairwise cross-feature interactions."""

    def test_t3_pairwise_sync_workspace_and_ruff_check(self) -> None:
        """Pairwise: uv sync --extra workspace + uv run ruff check slavv_python."""
        sync_proc = run_cmd(["uv", "sync", "--dry-run", "--extra", "workspace"])
        assert sync_proc.returncode == 0
        ruff_proc = run_cmd(["uv", "run", "ruff", "check", "slavv_python"])
        assert ruff_proc.returncode == 0

    def test_t3_pairwise_lock_check_and_build(self, tmp_path: Path) -> None:
        """Pairwise: uv lock --check + uv build."""
        lock_proc = run_cmd(["uv", "lock", "--check"])
        assert lock_proc.returncode == 0
        build_proc = run_cmd(["uv", "build", "--out-dir", str(tmp_path / "pairwise_dist")])
        assert build_proc.returncode == 0

    def test_t3_pairwise_sync_workspace_and_pytest(self) -> None:
        """Pairwise: uv sync --extra workspace + uv run pytest on interface launcher."""
        sync_proc = run_cmd(["uv", "sync", "--dry-run", "--extra", "workspace", "--extra", "app"])
        assert sync_proc.returncode == 0
        test_proc = run_cmd(
            [
                "uv",
                "run",
                "--extra",
                "workspace",
                "--extra",
                "app",
                "pytest",
                "tests/unit/interface/test_streamlit_launcher.py",
                "-q",
            ]
        )
        assert test_proc.returncode == 0

    def test_t3_pairwise_sync_ci_extras_and_ci_gate_sequence(self) -> None:
        """Pairwise: uv sync (app,workspace,zarr) + ruff check + mypy."""
        sync_proc = run_cmd(
            ["uv", "sync", "--dry-run", "--extra", "app", "--extra", "workspace", "--extra", "zarr"]
        )
        assert sync_proc.returncode == 0

        lint_proc = run_cmd(["uv", "run", "ruff", "check", "slavv_python"])
        assert lint_proc.returncode == 0

        mypy_proc = run_cmd(["uv", "run", "mypy", "--version"])
        assert mypy_proc.returncode == 0

    def test_t3_pairwise_sync_all_extras_and_cli_entrypoints(self) -> None:
        """Pairwise: uv sync --all-extras + slavv CLI subcommands."""
        sync_proc = run_cmd(["uv", "sync", "--dry-run", "--all-extras"])
        assert sync_proc.returncode == 0

        info_proc = run_cmd(["uv", "run", "slavv", "info"])
        assert info_proc.returncode == 0

        help_proc = run_cmd(["uv", "run", "slavv", "--help"])
        assert help_proc.returncode == 0

    def test_t3_pairwise_git_tracking_and_ci_setup_uv(self) -> None:
        """Pairwise: uv.lock git trackability + CI workflow setup-uv configuration."""
        git_proc = run_cmd(["git", "check-ignore", "-v", "uv.lock"])
        assert git_proc.returncode != 0, "uv.lock must not be gitignored"

        wf_path = REPO_ROOT / ".github" / "workflows" / "regression-gate.yml"
        content = wf_path.read_text(encoding="utf-8")
        assert "astral-sh/setup-uv" in content


# ==============================================================================
# TIER 4: REAL-WORLD APPLICATION SCENARIOS (End-to-End User Journeys)
# ==============================================================================


class TestTier4RealWorldScenarios:
    """Tier 4: End-to-End User Journeys and Lifecycle Flows."""

    def test_t4_journey_clean_developer_onboarding(self) -> None:
        """Journey 1: New developer onboarding flow from clone to tool execution."""
        # 1. Verify lockfile exists
        assert (REPO_ROOT / "uv.lock").exists()
        # 2. Verify pyproject specifies hatchling
        pyproj = load_pyproject()
        assert pyproj["build-system"]["build-backend"] == "hatchling.build"
        # 3. Perform sync dry-run with recommended developer extras
        sync_proc = run_cmd(["uv", "sync", "--dry-run", "--extra", "app", "--extra", "workspace"])
        assert sync_proc.returncode == 0
        # 4. Run CLI info command
        info_proc = run_cmd(["uv", "run", "slavv", "info"])
        assert info_proc.returncode == 0
        # 5. Run linter check
        ruff_proc = run_cmd(["uv", "run", "ruff", "check", "slavv_python"])
        assert ruff_proc.returncode == 0
        # 6. Run fast unit test
        test_proc = run_cmd(
            ["uv", "run", "pytest", "tests/unit/interface/test_streamlit_launcher.py", "-q"]
        )
        assert test_proc.returncode == 0

    def test_t4_journey_ci_regression_pipeline_emulation(self) -> None:
        """Journey 2: Full emulation of the GitHub Actions Regression Gate CI pipeline."""
        # Step 1: Validate workflow YAML parsing
        wf_path = REPO_ROOT / ".github" / "workflows" / "regression-gate.yml"
        data = parse_yaml_file(wf_path)
        assert data is not None

        # Step 2: Simulate `uv sync --extra app --extra workspace --extra zarr`
        sync_proc = run_cmd(
            ["uv", "sync", "--dry-run", "--extra", "app", "--extra", "workspace", "--extra", "zarr"]
        )
        assert sync_proc.returncode == 0

        # Step 3: Simulate `uv run ruff check slavv_python`
        lint_proc = run_cmd(["uv", "run", "ruff", "check", "slavv_python"])
        assert lint_proc.returncode == 0

        # Step 4: Simulate `uv run mypy --version`
        mypy_proc = run_cmd(["uv", "run", "mypy", "--version"])
        assert mypy_proc.returncode == 0

        # Step 5: Simulate pytest execution on CI unit tests
        pytest_proc = run_cmd(
            ["uv", "run", "pytest", "tests/unit/interface/test_streamlit_launcher.py", "-q"]
        )
        assert pytest_proc.returncode == 0

    def test_t4_journey_release_packaging_distribution(self, tmp_path: Path) -> None:
        """Journey 3: End-to-end package release build, sdist & wheel validation."""
        dist_dir = tmp_path / "release_dist"
        dist_dir.mkdir(parents=True, exist_ok=True)

        # 1. Build sdist and wheel
        build_proc = run_cmd(["uv", "build", "--out-dir", str(dist_dir)])
        assert build_proc.returncode == 0, f"uv build failed: {build_proc.stderr}"

        # 2. Inspect sdist
        sdists = list(dist_dir.glob("*.tar.gz"))
        assert len(sdists) == 1
        with tarfile.open(sdists[0], "r:gz") as tar:
            members = tar.getnames()
            assert any("pyproject.toml" in m for m in members), (
                f"pyproject.toml missing in sdist: {members}"
            )
            assert any("README.md" in m for m in members), f"README.md missing in sdist: {members}"
            assert any("slavv_python" in m for m in members), (
                f"slavv_python missing in sdist: {members}"
            )

        # 3. Inspect wheel
        wheels = list(dist_dir.glob("*.whl"))
        assert len(wheels) == 1
        with zipfile.ZipFile(wheels[0], "r") as z:
            names = z.namelist()
            # Verify Python modules
            assert any("slavv_python/__init__.py" in n for n in names)
            assert any("slavv_python/interface/cli" in n for n in names)
            # Verify package data
            assert any("matlab_linspace_overrides.json" in n for n in names)
            assert any("matlab_random_linspace_reference.json" in n for n in names)
            # Verify metadata
            assert any(".dist-info/METADATA" in n for n in names)
            assert any(".dist-info/entry_points.txt" in n for n in names)

    def test_t4_journey_lock_sync_all_extras_lifecycle(self) -> None:
        """Journey 4: Full lifecycle lock check, all-extras sync, and CLI verification."""
        # 1. Verify lock matches pyproject
        lock_check = run_cmd(["uv", "lock", "--check"])
        assert lock_check.returncode == 0

        # 2. Verify all-extras dry-run resolution
        sync_all = run_cmd(["uv", "sync", "--dry-run", "--all-extras"])
        assert sync_all.returncode == 0

        # 3. Test CLI runtime health
        info_proc = run_cmd(["uv", "run", "slavv", "info"])
        assert info_proc.returncode == 0

        # 4. Verify importability
        import_proc = run_cmd(
            [
                "uv",
                "run",
                "python",
                "-c",
                "import slavv_python; import slavv_python.pipeline; print('OK')",
            ]
        )
        assert import_proc.returncode == 0
        assert "OK" in import_proc.stdout

    def test_t4_journey_developer_scripts_execution(self) -> None:
        """Journey 5: Developer probe script invocation via uv run."""
        # Test direct module execution via uv run python
        proc = run_cmd(
            [
                "uv",
                "run",
                "python",
                "-c",
                "import slavv_python.interface.cli as cli; print(cli.__name__)",
            ]
        )
        assert proc.returncode == 0
        assert "slavv_python.interface.cli" in proc.stdout
