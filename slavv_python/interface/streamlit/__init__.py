"""Grouped Streamlit app package for SLAVV."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .app import main

__all__ = ["main"]


def __getattr__(name: str):
    if name == "main":
        from .app import main

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
