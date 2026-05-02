"""Compatibility alias for legacy underscore energy backends."""

from __future__ import annotations

import sys

from source.core.energy_internal import energy_backends as _canonical_module

sys.modules[__name__] = _canonical_module
