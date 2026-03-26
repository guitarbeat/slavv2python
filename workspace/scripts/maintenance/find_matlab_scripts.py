from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SEARCH_ROOTS = (
    REPO_ROOT / "external" / "Vectorization-Public",
    REPO_ROOT / "workspace",
)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def _is_matlab_function(path: Path) -> bool:
    for line in _read_text(path).splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("%"):
            continue
        return stripped.startswith(("function ", "function["))
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
