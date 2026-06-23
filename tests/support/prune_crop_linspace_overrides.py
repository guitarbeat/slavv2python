"""Keep only crop probe-table keys in matlab_linspace_overrides.json."""

from __future__ import annotations

import json
from pathlib import Path

from tests.support.parity_harness import linspace_table_path

REPO_ROOT = Path(__file__).resolve().parents[2]
OVERRIDES_PATH = (
    REPO_ROOT / "slavv_python" / "pipeline" / "energy" / "matlab_linspace_overrides.json"
)


def main() -> int:
    table_path = linspace_table_path()
    if table_path is None:
        raise FileNotFoundError("crop linspace probe table fixture not available")
    table = json.loads(table_path.read_text(encoding="utf-8"))
    table_keys = {
        f"{row['offset']},{row['stride']},{row['count']},{row['local_start']}"
        for row in table["rows"]
    }
    overrides = json.loads(OVERRIDES_PATH.read_text(encoding="utf-8"))
    filtered = {key: values for key, values in overrides.items() if key in table_keys}
    OVERRIDES_PATH.write_text(json.dumps(filtered, indent=2) + "\n", encoding="utf-8")
    print(f"kept {len(filtered)} crop override keys; removed {len(overrides) - len(filtered)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
