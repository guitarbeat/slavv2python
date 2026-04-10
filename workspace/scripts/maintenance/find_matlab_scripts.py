from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SEARCH_ROOTS = (
    REPO_ROOT / "external" / "Vectorization-Public",
    REPO_ROOT / "workspace",
)
MATLAB_FUNCTION_RE = re.compile(r"^function\b", re.IGNORECASE)


def _is_matlab_function(path: Path) -> bool:
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        content = path.read_text(encoding="latin-1")
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("%"):
            continue
        return MATLAB_FUNCTION_RE.match(stripped) is not None
    return False


def find_scripts() -> list[Path]:
    matlab_files = sorted(
        path for root in SEARCH_ROOTS for path in root.rglob("*.m") if path.is_file()
    )
    return [path for path in matlab_files if not _is_matlab_function(path)]


def main() -> None:
    print("--- SCRIPTS FOUND ---")
    for path in find_scripts():
        print(path.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
