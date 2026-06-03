from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any


@contextmanager
def materialize_model_source(source: Any | None):
    """Yield a filesystem path for a model source that may be file-like."""
    if source is None:
        yield None
        return

    if isinstance(source, (str, os.PathLike)):
        yield str(source)
        return

    if hasattr(source, "getvalue"):
        payload = source.getvalue()
    elif hasattr(source, "read"):
        payload = source.read()
    else:
        raise TypeError(f"Unsupported model source type: {type(source)}")

    if isinstance(payload, str):
        payload = payload.encode("utf-8")

    source_name = getattr(source, "name", "uploaded-model.joblib")
    suffix = Path(str(source_name)).suffix or ".joblib"
    fd, temp_path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
        yield temp_path
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
