from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any


@contextmanager
def materialize_model_source(source: Any):
    """
    Context manager that ensures a model source is available as a local file path.
    Supports Path objects, strings, and UploadedFile-like objects with getvalue().
    """
    if isinstance(source, (str, Path)):
        yield str(source)
        return

    # Handle UploadedFile or similar objects with getvalue()
    if hasattr(source, "getvalue"):
        fd, temp_path = tempfile.mkstemp(suffix=".joblib")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(source.getvalue())
            yield temp_path
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        return

    raise TypeError(f"Unsupported model source type: {type(source)}")
