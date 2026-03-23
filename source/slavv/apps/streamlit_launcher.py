"""Launch the Streamlit UI through Streamlit's real CLI entrypoint."""

from __future__ import annotations

import os
import subprocess
import sys
from contextlib import contextmanager
from importlib import resources, util
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

_APP_PACKAGE = "slavv.apps"
_APP_SCRIPT = "web_app.py"


@contextmanager
def _resolve_app_path() -> Iterator[Path]:
    """Yield a filesystem path to the packaged Streamlit app script."""
    app_resource = resources.files(_APP_PACKAGE).joinpath(_APP_SCRIPT)
    with resources.as_file(app_resource) as app_path:
        yield Path(app_path)


def _build_command(app_path: Path, argv: Sequence[str]) -> list[str]:
    """Build the delegated `python -m streamlit run ...` command."""
    return [sys.executable, "-m", "streamlit", "run", str(app_path), *argv]


def _build_env() -> dict[str, str]:
    """Use UTF-8 for the delegated Streamlit process on Windows consoles."""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def main(argv: Sequence[str] | None = None) -> int:
    """Run the packaged Streamlit application through Streamlit itself."""
    if util.find_spec("streamlit") is None:
        print(
            "slavv-app requires the optional Streamlit dependency. "
            'Install it with `pip install -e ".[app]"` or `pip install slavv[app]`.',
            file=sys.stderr,
        )
        return 1

    streamlit_args = list(sys.argv[1:] if argv is None else argv)

    with _resolve_app_path() as app_path:
        completed = subprocess.run(
            _build_command(app_path, streamlit_args),
            check=False,
            env=_build_env(),
        )

    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
