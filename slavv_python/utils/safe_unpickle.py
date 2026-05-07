"""
Safe Unpickler Module for SLAVV

This module provides a secure mechanism for loading pickled data, restricting
imports to a strict whitelist to mitigate potential security vulnerabilities
from malicious pickle files.
"""

from __future__ import annotations

import bz2
import gzip
import logging
import lzma
import os
import pickle
import tempfile
from typing import Any, ClassVar, Union

from joblib.numpy_pickle import NumpyUnpickler

# Configure logging
logger = logging.getLogger(__name__)

# Default maximum file size (1GB)
DEFAULT_MAX_PICKLE_SIZE = 1024 * 1024 * 1024
_STREAM_CHUNK_SIZE = 1024 * 1024


class SafeNumpyUnpickler(NumpyUnpickler):
    """
    Custom unpickler that restricts global imports to a safe whitelist.
    Inherits from joblib.numpy_pickle.NumpyUnpickler to support joblib dumps.
    """

    SAFE_MODULES: ClassVar[set[str]] = {
        "numpy",
        "sklearn",
        "joblib",
        "pandas",
        "scipy",
        "_codecs",
        "copyreg",
        "collections",
        "types",
    }
    SAFE_BUILTIN_GLOBALS: ClassVar[set[str]] = {
        "bytearray",
        "complex",
        "frozenset",
        "range",
        "set",
        "slice",
    }

    def find_class(self, module, name):
        if module in {"builtins", "__builtin__"}:
            if name not in self.SAFE_BUILTIN_GLOBALS:
                raise pickle.UnpicklingError(f"Global '{module}.{name}' is forbidden")
            return super().find_class(module, name)

        # Allow safe modules and their submodules
        # e.g. 'numpy.core.multiarray' starts with 'numpy'
        is_safe = False
        if module in self.SAFE_MODULES:
            is_safe = True
        else:
            for m in self.SAFE_MODULES:
                if module.startswith(f"{m}."):
                    is_safe = True
                    break

        if not is_safe:
            raise pickle.UnpicklingError(f"Global '{module}.{name}' is forbidden")

        return super().find_class(module, name)


def _detect_opener(file_path: str):
    with open(file_path, "rb") as handle:
        header = handle.read(6)

    if header.startswith(b"\x1f\x8b"):
        return gzip.open
    if header.startswith(b"BZh"):
        return bz2.open
    if header.startswith((b"\xfd7zXZ", b"]\x00\x00\x80\x00")):
        return lzma.open
    return open


def _materialize_bounded_stream(file_path: str, opener, max_size: int) -> str:
    """Copy the pickle payload to a temp file while enforcing a post-decompression size cap."""
    fd, temp_name = tempfile.mkstemp(suffix=".pkl")
    total_size = 0
    try:
        with os.fdopen(fd, "wb") as temp_handle, opener(file_path, "rb") as source:
            while True:
                chunk = slavv_python.read(_STREAM_CHUNK_SIZE)
                if not chunk:
                    break
                total_size += len(chunk)
                if total_size > max_size:
                    raise ValueError(
                        f"Decompressed file size {total_size} exceeds limit of {max_size} bytes"
                    )
                temp_handle.write(chunk)
        return temp_name
    except Exception:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
        raise


def safe_load(file_path: Union[str, os.PathLike], max_size: int = DEFAULT_MAX_PICKLE_SIZE) -> Any:
    """
    Safely load a pickle file with size limits and restricted imports.
    Supports compressed files (gzip, bz2, lzma) if detected by extension or magic bytes.
    Also supports joblib dumps via NumpyUnpickler.

    Args:
        file_path: Path to the pickle file.
        max_size: Maximum allowed file size in bytes.

    Returns:
        Unpickled object.

    Raises:
        ValueError: If file exceeds size limit.
        pickle.UnpicklingError: If pickle is malformed or contains forbidden globals.
        FileNotFoundError: If file not found.
    """
    file_path = str(file_path)

    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    file_size = os.path.getsize(file_path)
    if file_size > max_size:
        logger.error(f"File size {file_size} exceeds limit of {max_size} bytes")
        raise ValueError(f"File size {file_size} exceeds limit of {max_size} bytes")

    try:
        opener = _detect_opener(file_path)
        materialized_path = (
            file_path
            if opener is open
            else _materialize_bounded_stream(file_path, opener, max_size)
        )
        try:
            with open(materialized_path, "rb") as f:
                # We must pass the filename and file handle to NumpyUnpickler
                # ensure_native_byte_order=False is required to verify correctly (as per memory)
                # mmap_mode=None (default) as we load everything to memory
                try:
                    # Try with positional arguments matching joblib 1.5.3 signature
                    return SafeNumpyUnpickler(materialized_path, f, False, None).load()
                except TypeError:
                    # Fallback for older joblib versions or different signature
                    # Some versions might not have ensure_native_byte_order
                    return SafeNumpyUnpickler(materialized_path, f).load()
        finally:
            if materialized_path != file_path and os.path.exists(materialized_path):
                os.unlink(materialized_path)

    except (pickle.UnpicklingError, AttributeError, ImportError, IndexError, TypeError) as e:
        logger.error(f"Unpickling error for {file_path}: {e}")
        raise pickle.UnpicklingError(f"Failed to safely unpickle {file_path}: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error loading {file_path}: {e}")
        raise e
