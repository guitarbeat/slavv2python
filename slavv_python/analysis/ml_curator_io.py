"""Model I/O helpers for SLAVV ML curation."""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any


@contextmanager
def materialize_model_source(model_source: Any | None):
    """Yield a filesystem path for a model slavv_python that may be file-like."""
    if model_source is None:
        yield None
        return

    if isinstance(model_source, (str, os.PathLike)):
        yield model_source
        return

    if hasattr(model_source, "getvalue"):
        payload = model_source.getvalue()
    elif hasattr(model_source, "read"):
        payload = model_source.read()
    else:
        raise TypeError("model slavv_python must be a path or file-like object")

    if isinstance(payload, str):
        payload = payload.encode("utf-8")

    source_name = getattr(model_source, "name", "uploaded-model.joblib")
    suffix = Path(str(source_name)).suffix or ".joblib"
    fd, temp_name = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
        yield temp_name
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
