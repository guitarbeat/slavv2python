"""One-shot helper: merge MATLAB linspace meshes into crop override table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.io

REPO_ROOT = Path(__file__).resolve().parents[2]
RANDOM_REFERENCE_PATHS = (
    REPO_ROOT / "slavv_python" / "pipeline" / "energy" / "matlab_random_linspace_reference.json",
    REPO_ROOT / "tests" / "support" / "fixtures" / "matlab_random_linspace_reference.json",
)


def export_overrides(manifest_path: Path, matlab_mat_path: Path) -> int:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data = scipy.io.loadmat(matlab_mat_path, squeeze_me=True, struct_as_record=False)
    records = data["results"].linspace_records
    if not isinstance(records, (list, np.ndarray)):
        records = [records]
    payload: dict[str, list[float]] = {}
    for context, record in zip(manifest["linspace_contexts"], records, strict=True):
        key = f"{context['offset']},{context['stride']},{context['count']},{context['local_start']}"
        payload[key] = [float(value) for value in np.asarray(record.values).reshape(-1)]
    encoded = json.dumps(payload, indent=2) + "\n"
    for path in RANDOM_REFERENCE_PATHS:
        path.write_text(encoded, encoding="utf-8")
    return len(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--matlab-mat", type=Path, required=True)
    args = parser.parse_args(argv)
    added = export_overrides(args.manifest, args.matlab_mat)
    print(
        f"wrote {added} random linspace reference keys to {len(RANDOM_REFERENCE_PATHS)} locations"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
