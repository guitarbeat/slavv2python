"""User-facing applications for SLAVV.

Submodules
----------
cli
    Command-line interface (``slavv`` command).
parity_cli
    Packaged MATLAB/Python parity comparison CLI.
web_app
    Streamlit web application.
"""

from __future__ import annotations

from .cli import main  # legacy package-level re-export; console scripts target slavv.apps.cli:main

__all__ = ["main"]
