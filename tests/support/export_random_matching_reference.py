"""One-shot helper: export MATLAB matching-kernel meshes for the random corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.io

FIXTURE_PATH = Path(__file__).with_name("fixtures") / "matlab_random_matching_reference.json"


def _spacing_key(spacing_yxz: np.ndarray) -> str:
    return ",".join(f"{float(value):.17g}" for value in spacing_yxz.reshape(-1))


def export_reference(matlab_mat_path: Path, output_path: Path = FIXTURE_PATH) -> int:
    data = scipy.io.loadmat(matlab_mat_path, squeeze_me=True, struct_as_record=False)
    records = data["results"].records
    if not isinstance(records, (list, np.ndarray)):
        records = [records]
    kernels: dict[str, dict[str, object]] = {}
    for record in records:
        spacing = np.asarray(record.spacing_yxz, dtype=np.float64).reshape(-1)
        shape = [int(value) for value in np.asarray(record.shape_yxz).reshape(-1)]
        key = _spacing_key(spacing)
        kernels[key] = {
            "spacing_yxz": spacing.tolist(),
            "shape_yxz": shape,
            "values": [float(value) for value in np.asarray(record.values).reshape(-1)],
        }
    payload = {"version": 1, "kernels": kernels}
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return len(kernels)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matlab-mat", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=FIXTURE_PATH)
    args = parser.parse_args(argv)
    count = export_reference(args.matlab_mat, args.output)
    print(f"wrote {count} matching-kernel references to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
