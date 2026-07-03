"""Property 16: Oracle Artifact Completeness.

# Feature: matlab-python-parity, Property 16: Oracle Artifact Completeness

For any valid oracle root directory, ``ensure_oracle_artifacts`` (or
``inspect_oracle_artifact``) shall successfully load exactly one artifact surface
for each of the four gated stages (energy, vertices, edges, network), and no stage
surface shall be None / not ready.

This test builds a minimal fixture oracle directory under ``tmp_path`` with stub
joblib-serialized payloads for all four stages, then asserts:

1. Every ``OracleArtifactStatus.ready`` is ``True``.
2. All four stages return an ``OracleArtifactStatus`` (i.e. the result is not None).
3. No exception propagates during loading (guaranteeing the loader does not raise
   on a complete, well-formed oracle root).

Validates: Requirements 10.2
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pytest

from slavv_python.analytics.parity.oracle.oracle_artifacts import (
    OracleArtifactStatus,
    inspect_oracle_artifact,
)
from slavv_python.analytics.parity.proof.exact_proof_contract import EXACT_STAGE_ORDER

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STAGES: tuple[str, ...] = EXACT_STAGE_ORDER  # ("energy", "vertices", "edges", "network")

# Relative path where oracle_artifacts.py expects normalized pkl files:
#   oracle_root / NORMALIZED_DIR / "oracle" / f"{stage}.pkl"
# NORMALIZED_DIR = Path("03_Analysis") / "normalized"
_ARTIFACT_REL = Path("03_Analysis") / "normalized" / "oracle"


def _make_stub_oracle_root(root: Path) -> Path:
    """Create a minimal oracle directory with one stub pkl artifact per stage.

    Each artifact is a dict with a single numpy array keyed by stage name so the
    loader can open the file and produce a non-empty ``summary`` dict.  The exact
    payload shape does not matter — the loader only needs to be able to call
    ``joblib.load`` without raising.

    Returns ``root`` for convenience.
    """
    artifact_dir = root / _ARTIFACT_REL
    artifact_dir.mkdir(parents=True, exist_ok=True)

    for stage in _STAGES:
        stub_payload = {stage: np.zeros((2, 2, 2), dtype=np.float64)}
        joblib.dump(stub_payload, artifact_dir / f"{stage}.pkl")

    return root


# ---------------------------------------------------------------------------
# Property 16: Oracle Artifact Completeness
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_all_four_stage_artifacts_are_present_and_ready(tmp_path: Path) -> None:
    """All four stage artifact surfaces must be non-None and ready after loading.

    Build a complete fixture oracle root, then inspect each of the four stages.
    Each returned ``OracleArtifactStatus`` must satisfy ``.ready == True`` — that
    is, the file must exist *and* be loadable without error.
    """
    oracle_root = _make_stub_oracle_root(tmp_path / "oracle")

    results: dict[str, OracleArtifactStatus] = {}
    for stage in _STAGES:
        status = inspect_oracle_artifact(oracle_root, stage)
        results[stage] = status

    # --- property assertion: every result is non-None -------------------------
    for stage in _STAGES:
        assert results[stage] is not None, (
            f"inspect_oracle_artifact returned None for stage '{stage}'; "
            "expected an OracleArtifactStatus object."
        )

    # --- property assertion: every surface is ready ---------------------------
    for stage in _STAGES:
        status = results[stage]
        assert status.ready, (
            f"OracleArtifactStatus for stage '{stage}' is not ready. "
            f"exists={status.exists}, readable={status.readable}, "
            f"error={status.error!r}"
        )


@pytest.mark.unit
def test_loader_raises_no_exception_for_complete_oracle_root(tmp_path: Path) -> None:
    """The loader must not raise any exception when all four artifacts are present.

    Wraps all four ``inspect_oracle_artifact`` calls in a try/except to produce a
    clean failure message if any unexpected exception propagates.
    """
    oracle_root = _make_stub_oracle_root(tmp_path / "oracle")

    for stage in _STAGES:
        try:
            status = inspect_oracle_artifact(oracle_root, stage)
        except Exception as exc:  # noqa: BLE001
            pytest.fail(
                f"inspect_oracle_artifact raised an unexpected exception for stage "
                f"'{stage}': {type(exc).__name__}: {exc}"
            )
        else:
            assert status is not None, (
                f"inspect_oracle_artifact returned None for stage '{stage}'"
            )


@pytest.mark.unit
def test_exactly_four_stages_loaded(tmp_path: Path) -> None:
    """Exactly four stage surfaces are returned — one per gated stage.

    Verifies both the count and the exact stage names in the canonical order.
    """
    oracle_root = _make_stub_oracle_root(tmp_path / "oracle")

    statuses = [inspect_oracle_artifact(oracle_root, stage) for stage in _STAGES]

    assert len(statuses) == 4, (
        f"Expected 4 stage surfaces, got {len(statuses)}: {[s.stage for s in statuses]}"
    )

    returned_stages = [s.stage for s in statuses]
    assert returned_stages == list(_STAGES), (
        f"Stage names do not match EXACT_STAGE_ORDER. "
        f"Expected {list(_STAGES)}, got {returned_stages}"
    )


@pytest.mark.unit
def test_ensure_oracle_artifacts_no_repair_all_ready(tmp_path: Path) -> None:
    """``ensure_oracle_artifacts`` with ``repair=False`` returns all four stages ready.

    This exercises the public API entry-point to confirm it handles a complete
    oracle root without attempting to materialize missing artifacts.
    """
    from slavv_python.analytics.parity.oracle.oracle_artifacts import ensure_oracle_artifacts

    oracle_root = _make_stub_oracle_root(tmp_path / "oracle")

    statuses = ensure_oracle_artifacts(oracle_root, repair=False)

    assert set(statuses.keys()) == set(_STAGES), (
        f"ensure_oracle_artifacts returned stages {set(statuses.keys())} "
        f"but expected {set(_STAGES)}"
    )

    for stage, status in statuses.items():
        assert status is not None, (
            f"ensure_oracle_artifacts returned None status for stage '{stage}'"
        )
        assert status.ready, (
            f"ensure_oracle_artifacts status for stage '{stage}' is not ready. "
            f"exists={status.exists}, readable={status.readable}, error={status.error!r}"
        )
