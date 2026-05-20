from __future__ import annotations

import contextlib

# Stub for ML IO that is currently missing from the codebase
# but expected by unit tests.

@contextlib.contextmanager
def materialize_model_source(source):
    """Context manager that yields a local path to a model file."""
    yield source
