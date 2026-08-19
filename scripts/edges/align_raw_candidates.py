"""Align the fresh MATLAB raw watershed candidates (batch indexing) to the
oracle/Python canonical vertex indexing via the verified coordinate transform
(Python [Z,Y,X] 0-based -> MATLAB [Y,X,Z] 1-based: reorder then +1), then compare
MATLAB's emission order of the tied crop pair (4043,6281)/(4212,6281) with Python's.

Run:
  .venv\\Scripts\\python.exe scripts/align_matlab_raw_candidates.py
"""
from __future__ import annotations

import scipy.io
from pathlib import Path

import h5py
import numpy as np

from slavv_python.utils.safe_unpickle import safe_load

REPO = Path(__file__).resolve().parents[1]
BATCH = REPO / "workspace/oracles/180709_E_crop_M_v2/01_Input/matlab_results/batch_260624-105705"
STANDALONE_DUMP = REPO / "workspace/scratch/matlab_edge_dump/raw_watershed_candidates.mat"
PY_CAND = REPO / "workspace/runs/oracle_180709_E/crop_M_exact_v3/04_Edges/candidates.pkl"
ORACLE_VERTS = REPO / "workspace/oracles/180709_E_crop_M_v2/03_Analysis/normalized/oracle/vertices.pkl"


def canonical_positions(positions_zxy: np.ndarray) -> np.ndarray:
    """Oracle/Python 0-based [Z,Y,X] -> MATLAB 1-based [Y,X,Z] integer positions."""
    yxz = positions_zxy[:, [1, 2, 0]] + 1.0
    return np.round(yxz).astype(np.int64)


def position_to_index_map(positions: np.ndarray) -> dict[tuple[int, int, int], int]:
    return {tuple(p.tolist()): i for i, p in enumerate(positions)}


def main() -> None:
    # Canonical vertex positions (oracle == Python) in 0-based [Z,Y,X].
    canon_pos = np.asarray(
        safe_load(ORACLE_VERTS)["positions"], dtype=np.float64
    )
    canon_pos_yxz = canonical_positions(canon_pos)
    canon_map = position_to_index_map(canon_pos_yxz)
    n_canon = len(canon_pos)

    # MATLAB standalone raw candidates, in BATCH vertex indexing (v7.3 HDF5).
    with h5py.File(str(STANDALONE_DUMP), "r") as f:
        matl_e2v = np.asarray(f["edges2vertices"][()], dtype=np.int64).T  # (N,2), 1-based
    print(f"MATLAB standalone raw candidates: {matl_e2v.shape[0]} edges")

    # Batch curated vertex positions in [Y,X,Z] 1-based.
    sv = scipy.io.loadmat(str(BATCH / "vectors" / "curated_vertices_260624-105705_180709_E_crop_M.mat"))
    batch_pos = np.asarray(sv["vertex_space_subscripts"], dtype=np.int64)  # [Y,X,Z] 1-based
    assert batch_pos.shape[0] == n_canon, (batch_pos.shape[0], n_canon)

    # Map batch index -> canonical index via identical position.
    batch_to_canon = np.full(batch_pos.shape[0] + 1, -1, dtype=np.int64)
    matched = 0
    for bidx in range(batch_pos.shape[0]):
        cidx = canon_map.get(tuple(batch_pos[bidx].tolist()), -1)
        batch_to_canon[bidx + 1] = cidx  # MATLAB is 1-based
        if cidx >= 0:
            matched += 1
    print(f"batch->canonical mapped: {matched}/{batch_pos.shape[0]}")

    # Remap MATLAB edges2vertices into canonical indexing.
    remapped = batch_to_canon[matl_e2v]
    bad = int(np.sum(remapped < 0))
    print(f"MATLAB edges with unmapped vertices: {bad}")
    remapped = remapped[remapped[:, 0] >= 0]

    # Python emission order from crop_M_exact_v3 candidates.pkl (canonical indexing).
    py = safe_load(PY_CAND)
    py_e2v = np.asarray(py["connections"], dtype=np.int64)
    print(f"Python crop candidates: {py_e2v.shape[0]} edges")

    target = 6281
    py_order = [tuple(sorted([int(a), int(b)])) for a, b in py_e2v]
    mat_order = [tuple(sorted([int(a), int(b)])) for a, b in remapped]

    def find(pair, order):
        for i, p in enumerate(order):
            if p == pair:
                return i
        return None

    for pair in [(4043, 6281), (4212, 6281)]:
        pi = find(pair, py_order)
        mi = find(pair, mat_order)
        print(
            f"pair {pair}: Python emission row={pi}  MATLAB emission row={mi}  "
            f"({'SAME' if (pi is not None and mi is not None and pi == mi) else 'DIFFERENT'})"
        )

    # Show the local emission-order context around the pair in both.
    for label, order in [("PYTHON", py_order), ("MATLAB", mat_order)]:
        for pair in [(4043, 6281), (4212, 6281)]:
            i = find(pair, order)
            if i is None:
                continue
            lo = max(0, i - 3)
            hi = min(len(order), i + 4)
            ctx = order[lo:hi]
            print(f"  {label} ctx around {pair} (row {i}): {ctx}")


if __name__ == "__main__":
    main()
