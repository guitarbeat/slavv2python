"""
Stage-isolated network gate validation and execution.

This module provides fast-fail validation and execution for the stage-isolated
network gate workflow, which imports exact MATLAB edges and reruns only Python
network assembly to isolate parity issues.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class NetworkGateValidation:
    """Pre-execution validation for stage-isolated network gate."""

    has_matlab_edges: bool
    has_matlab_vertices: bool
    has_matlab_energy: bool
    matlab_edges_fingerprint: str
    matlab_vertices_fingerprint: str
    validation_passed: bool
    validation_errors: list[str] = field(default_factory=list)
    validation_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


def _compute_file_fingerprint(file_path: Path) -> str:
    """
    Compute SHA-256 fingerprint of a file.

    Args:
        file_path: Path to the file to fingerprint

    Returns:
        Fingerprint string in format "sha256:hexdigest"
    """
    if not file_path.exists():
        return ""

    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            # Read in chunks to handle large files efficiently
            for chunk in iter(lambda: f.read(8192), b""):
                sha256_hash.update(chunk)
        return f"sha256:{sha256_hash.hexdigest()}"
    except Exception as e:
        logger.warning(f"Failed to compute fingerprint for {file_path}: {e}")
        return ""


def validate_network_gate_artifacts(run_root: Path) -> NetworkGateValidation:
    """
    Fast-fail validation before network gate execution.

    Checks for required MATLAB edge artifacts and generates fingerprints
    for provenance tracking.

    Args:
        run_root: Root directory of the comparison run

    Returns:
        NetworkGateValidation with validation status and any errors
    """
    from .run_layout import resolve_run_layout

    layout = resolve_run_layout(run_root)
    matlab_dir = layout["matlab_dir"]

    validation_errors: list[str] = []

    # Check for MATLAB edges artifact
    matlab_edges_path = matlab_dir / "edges.mat"
    has_matlab_edges = matlab_edges_path.exists()
    if not has_matlab_edges:
        validation_errors.append(
            f"Missing MATLAB edges artifact: {matlab_edges_path}. "
            "Run imported-MATLAB edge rerun first."
        )

    # Check for MATLAB vertices artifact
    matlab_vertices_path = matlab_dir / "vertices.mat"
    has_matlab_vertices = matlab_vertices_path.exists()
    if not has_matlab_vertices:
        validation_errors.append(
            f"Missing MATLAB vertices artifact: {matlab_vertices_path}. "
            "Required for network gate validation."
        )

    # Check for MATLAB energy artifact
    matlab_energy_path = matlab_dir / "energy.mat"
    has_matlab_energy = matlab_energy_path.exists()
    if not has_matlab_energy:
        validation_errors.append(
            f"Missing MATLAB energy artifact: {matlab_energy_path}. "
            "Required for network gate validation."
        )

    # Compute fingerprints for existing artifacts
    matlab_edges_fingerprint = (
        _compute_file_fingerprint(matlab_edges_path) if has_matlab_edges else ""
    )
    matlab_vertices_fingerprint = (
        _compute_file_fingerprint(matlab_vertices_path) if has_matlab_vertices else ""
    )

    validation_passed = not validation_errors

    return NetworkGateValidation(
        has_matlab_edges=has_matlab_edges,
        has_matlab_vertices=has_matlab_vertices,
        has_matlab_energy=has_matlab_energy,
        matlab_edges_fingerprint=matlab_edges_fingerprint,
        matlab_vertices_fingerprint=matlab_vertices_fingerprint,
        validation_passed=validation_passed,
        validation_errors=validation_errors,
    )


@dataclass
class NetworkGateExecution:
    """Execution metadata for stage-isolated network gate."""

    validation: NetworkGateValidation
    started_at: str
    completed_at: str
    elapsed_seconds: float
    comparison_exact_network_forced: bool
    parity_achieved: bool
    vertices_match: bool
    edges_match: bool
    strands_match: bool
    python_network_fingerprint: str
    proof_artifact_json_path: str = ""
    proof_artifact_markdown_path: str = ""
    proof_index_path: str = ""
    peak_memory_mb: float | None = None
    cpu_time_seconds: float | None = None
    execution_errors: list[str] = field(default_factory=list)


def execute_stage_isolated_network_gate(
    run_root: Path,
    *,
    input_file: Path,
    params: dict[str, Any],
) -> NetworkGateExecution:
    """
    Execute stage-isolated network gate with timing and validation.

    Process:
    1. Validate required artifacts (fast-fail)
    2. Force comparison_exact_network=True
    3. Import MATLAB edges/vertices/energy artifacts
    4. Rerun Python from network stage
    5. Compare results and record parity status
    6. Persist execution metadata

    Args:
        run_root: Root directory of the comparison run
        input_file: Path to the input TIFF volume
        params: Pipeline parameters

    Returns:
        NetworkGateExecution with timing and parity status
    """
    import copy
    import json
    import time

    import psutil

    from slavv.core import SLAVVProcessor
    from slavv.io import load_tiff_volume
    from slavv.io.matlab_bridge import import_matlab_batch
    from slavv.utils.safe_unpickle import safe_load

    from .metrics import compare_results
    from .proof_artifacts import generate_proof_artifact, maintain_proof_artifact_index
    from .run_layout import resolve_run_layout

    # Start timing
    started_at = datetime.now().isoformat()
    start_time = time.time()
    start_cpu_time = time.process_time()

    # Step 1: Validate required artifacts (fast-fail)
    validation = validate_network_gate_artifacts(run_root)
    if not validation.validation_passed:
        elapsed_seconds = time.time() - start_time
        return NetworkGateExecution(
            validation=validation,
            started_at=started_at,
            completed_at=datetime.now().isoformat(),
            elapsed_seconds=elapsed_seconds,
            comparison_exact_network_forced=False,
            parity_achieved=False,
            vertices_match=False,
            edges_match=False,
            strands_match=False,
            python_network_fingerprint="",
            execution_errors=validation.validation_errors,
        )

    layout = resolve_run_layout(run_root)
    matlab_dir = layout["matlab_dir"]
    python_dir = layout["python_dir"]
    metadata_dir = layout["metadata_dir"]

    execution_errors: list[str] = []

    try:
        # Step 2: Force comparison_exact_network=True mode
        params_for_python = copy.deepcopy(params)
        params_for_python["comparison_exact_network"] = True
        params_for_python["python_parity_rerun_from"] = "network"

        # Step 3: Import MATLAB edges/vertices/energy artifacts
        checkpoint_dir = python_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        matlab_batch_folder = next(
            (
                item
                for item in matlab_dir.iterdir()
                if item.is_dir() and item.name.startswith("batch_")
            ),
            None,
        )
        if matlab_batch_folder is None:
            execution_errors.append("Could not find MATLAB batch folder in matlab_dir")
            elapsed_seconds = time.time() - start_time
            return NetworkGateExecution(
                validation=validation,
                started_at=started_at,
                completed_at=datetime.now().isoformat(),
                elapsed_seconds=elapsed_seconds,
                comparison_exact_network_forced=True,
                parity_achieved=False,
                vertices_match=False,
                edges_match=False,
                strands_match=False,
                python_network_fingerprint="",
                execution_errors=execution_errors,
            )

        logger.info(f"Importing MATLAB artifacts from {matlab_batch_folder}")
        imported = import_matlab_batch(
            str(matlab_batch_folder),
            checkpoint_dir,
            stages=["energy", "vertices", "edges"],
        )
        logger.info(f"Imported stages: {list(imported.keys())}")

        # Step 4: Rerun Python from network stage
        logger.info("Loading input image")
        image = load_tiff_volume(str(input_file))

        logger.info("Initializing Python processor")
        processor = SLAVVProcessor()

        logger.info("Running Python pipeline from network stage")
        results = processor.process_image(
            image,
            params_for_python,
            run_dir=str(run_root),
            force_rerun_from="network",
        )

        # Step 5: Compare results and record parity status
        logger.info("Loading MATLAB results for comparison")
        matlab_edges_path = checkpoint_dir / "checkpoint_edges.pkl"
        matlab_vertices_path = checkpoint_dir / "checkpoint_vertices.pkl"
        matlab_network_path = checkpoint_dir / "checkpoint_network.pkl"

        if not matlab_edges_path.exists():
            execution_errors.append(f"MATLAB edges checkpoint not found: {matlab_edges_path}")
        if not matlab_vertices_path.exists():
            execution_errors.append(f"MATLAB vertices checkpoint not found: {matlab_vertices_path}")

        if execution_errors:
            elapsed_seconds = time.time() - start_time
            return NetworkGateExecution(
                validation=validation,
                started_at=started_at,
                completed_at=datetime.now().isoformat(),
                elapsed_seconds=elapsed_seconds,
                comparison_exact_network_forced=True,
                parity_achieved=False,
                vertices_match=False,
                edges_match=False,
                strands_match=False,
                python_network_fingerprint="",
                execution_errors=execution_errors,
            )

        matlab_edges = safe_load(matlab_edges_path)
        matlab_vertices = safe_load(matlab_vertices_path)
        matlab_network = safe_load(matlab_network_path) if matlab_network_path.exists() else None

        # Build comparison structure
        matlab_results = {
            "success": True,
            "elapsed_time": 0.0,
            "results": {
                "edges": matlab_edges,
                "vertices": matlab_vertices,
                "network": matlab_network,
            },
        }

        python_results = {
            "success": True,
            "elapsed_time": time.time() - start_time,
            "results": results,
        }

        comparison = compare_results(matlab_results, python_results, None)

        # Extract parity status
        vertices_match = comparison.get("vertices", {}).get("exact_match", False)
        edges_match = comparison.get("edges", {}).get("exact_match", False)
        network_comparison = comparison.get("network", {})
        strands_match = network_comparison.get("strands_exact_match", False)
        parity_achieved = vertices_match and edges_match and strands_match

        # Compute Python network fingerprint
        python_network_fingerprint = ""
        if "network" in results:
            network_json = json.dumps(results["network"], sort_keys=True)
            python_network_fingerprint = (
                f"sha256:{hashlib.sha256(network_json.encode()).hexdigest()}"
            )

        elapsed_seconds = time.time() - start_time
        cpu_time_seconds = time.process_time() - start_cpu_time
        peak_memory_mb = None
        try:
            peak_memory_mb = psutil.Process().memory_info().rss / (1024 * 1024)
        except (OSError, psutil.Error):
            peak_memory_mb = None

        # Step 6: Persist execution metadata
        execution_metadata = {
            "validation": {
                "has_matlab_edges": validation.has_matlab_edges,
                "has_matlab_vertices": validation.has_matlab_vertices,
                "has_matlab_energy": validation.has_matlab_energy,
                "matlab_edges_fingerprint": validation.matlab_edges_fingerprint,
                "matlab_vertices_fingerprint": validation.matlab_vertices_fingerprint,
                "validation_passed": validation.validation_passed,
                "validation_errors": validation.validation_errors,
                "validation_timestamp": validation.validation_timestamp,
            },
            "started_at": started_at,
            "completed_at": datetime.now().isoformat(),
            "elapsed_seconds": elapsed_seconds,
            "comparison_exact_network_forced": True,
            "matlab_batch_folder": str(matlab_batch_folder),
            "parity_achieved": parity_achieved,
            "vertices_match": vertices_match,
            "edges_match": edges_match,
            "strands_match": strands_match,
            "python_network_fingerprint": python_network_fingerprint,
            "peak_memory_mb": peak_memory_mb,
            "cpu_time_seconds": cpu_time_seconds,
            "execution_errors": execution_errors,
        }

        metadata_dir.mkdir(parents=True, exist_ok=True)
        execution_file = metadata_dir / "network_gate_execution.json"
        with open(execution_file, "w", encoding="utf-8") as f:
            json.dump(execution_metadata, f, indent=2)

        logger.info(f"Network gate execution metadata persisted to {execution_file}")

        proof_artifact_json_path = ""
        proof_artifact_markdown_path = ""
        proof_index_path = ""
        if parity_achieved:
            try:
                proof = generate_proof_artifact(execution_file, run_root=run_root)
                maintain_proof_artifact_index(run_root, proof)
                proof_artifact_json_path = proof.artifact_json_path
                proof_artifact_markdown_path = proof.artifact_markdown_path
                proof_index_path = str(
                    resolve_run_layout(run_root)["analysis_dir"] / "proof_artifact_index.json"
                )
            except (OSError, ValueError) as exc:
                logger.warning("Failed to generate proof artifacts for %s: %s", run_root, exc)

        return NetworkGateExecution(
            validation=validation,
            started_at=started_at,
            completed_at=datetime.now().isoformat(),
            elapsed_seconds=elapsed_seconds,
            comparison_exact_network_forced=True,
            parity_achieved=parity_achieved,
            vertices_match=vertices_match,
            edges_match=edges_match,
            strands_match=strands_match,
            python_network_fingerprint=python_network_fingerprint,
            proof_artifact_json_path=proof_artifact_json_path,
            proof_artifact_markdown_path=proof_artifact_markdown_path,
            proof_index_path=proof_index_path,
            peak_memory_mb=peak_memory_mb,
            cpu_time_seconds=cpu_time_seconds,
            execution_errors=execution_errors,
        )

    except Exception as e:
        elapsed_seconds = time.time() - start_time
        error_message = f"Network gate execution failed: {e}"
        logger.exception(error_message)
        execution_errors.append(error_message)

        return NetworkGateExecution(
            validation=validation,
            started_at=started_at,
            completed_at=datetime.now().isoformat(),
            elapsed_seconds=elapsed_seconds,
            comparison_exact_network_forced=True,
            parity_achieved=False,
            vertices_match=False,
            edges_match=False,
            strands_match=False,
            python_network_fingerprint="",
            execution_errors=execution_errors,
        )
