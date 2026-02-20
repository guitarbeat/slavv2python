"""
Safe Unpickler Module for SLAVV

This module provides a secure mechanism for loading pickled data, restricting
imports to a strict whitelist to mitigate potential security vulnerabilities
from malicious pickle files.
"""

import pickle
import io
import os
import sys
import logging
import joblib
import numpy as np
import gzip
import bz2
import lzma
from joblib.numpy_pickle import NumpyUnpickler
from typing import Any, List, Set, Union, Optional

# Configure logging
logger = logging.getLogger(__name__)

# Default maximum file size (1GB)
DEFAULT_MAX_PICKLE_SIZE = 1024 * 1024 * 1024

class SafeNumpyUnpickler(NumpyUnpickler):
    """
    Custom unpickler that restricts global imports to a safe whitelist.
    Inherits from joblib.numpy_pickle.NumpyUnpickler to support joblib dumps.
    """

    SAFE_MODULES = {
        'numpy',
        'sklearn',
        'joblib',
        'pandas',
        'scipy',
        '_codecs',
        'copyreg',
        'collections',
        'types',
        'builtins',
        '__builtin__'
    }

    def find_class(self, module, name):
        # Allow safe modules and their submodules
        # e.g. 'numpy.core.multiarray' starts with 'numpy'
        is_safe = False
        if module in self.SAFE_MODULES:
            is_safe = True
        else:
            for m in self.SAFE_MODULES:
                if module.startswith(m + '.'):
                    is_safe = True
                    break

        if not is_safe:
            raise pickle.UnpicklingError(f"Global '{module}.{name}' is forbidden")

        return super().find_class(module, name)


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
        # Detect compression
        with open(file_path, 'rb') as f:
            header = f.read(4)

        if header.startswith(b'\x1f\x8b'):  # gzip
            opener = gzip.open
        elif header.startswith(b'BZh'):     # bz2
            opener = bz2.open
        elif header.startswith(b'\xfd7zXZ') or header.startswith(b']\x00\x00\x80\x00'): # lzma/xz
            opener = lzma.open
        else:
            opener = open

        with opener(file_path, 'rb') as f:
            # We must pass the filename and file handle to NumpyUnpickler
            # ensure_native_byte_order=False is required to verify correctly (as per memory)
            # mmap_mode=None (default) as we load everything to memory
            try:
                # Try with positional arguments matching joblib 1.5.3 signature
                return SafeNumpyUnpickler(file_path, f, False, None).load()
            except TypeError:
                # Fallback for older joblib versions or different signature
                # Some versions might not have ensure_native_byte_order
                return SafeNumpyUnpickler(file_path, f).load()

    except (pickle.UnpicklingError, AttributeError, ImportError, IndexError, TypeError) as e:
        logger.error(f"Unpickling error for {file_path}: {e}")
        raise pickle.UnpicklingError(f"Failed to safely unpickle {file_path}: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error loading {file_path}: {e}")
        raise e
