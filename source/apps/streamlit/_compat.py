"""Compatibility helpers for grouped Streamlit wrapper modules."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType


def bind_legacy_module(module_name: str) -> tuple[ModuleType, list[str]]:
    """Return the imported legacy module plus its exported names."""
    legacy_module = import_module(module_name)
    exported_names = getattr(legacy_module, "__all__", None)
    if exported_names is None:
        exported_names = [name for name in vars(legacy_module) if not name.startswith("_")]
    return legacy_module, sorted(set(exported_names))
